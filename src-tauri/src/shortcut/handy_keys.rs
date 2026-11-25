//! Handy-keys based keyboard shortcut implementation
//!
//! This module provides an alternative to Tauri's global-shortcut plugin
//! using the handy-keys library for more control over keyboard events.
//!
//! ## Architecture
//!
//! The implementation uses a dedicated manager thread that owns the chords:
//!
//! ```text
//! ┌─────────────────┐     commands      ┌──────────────────────┐
//! │   Main Thread   │ ───────────────▶ │   Manager Thread     │
//! │                 │   (via channel)   │                      │
//! │ - register()    │  keys (forwarder) │ - matches chords     │
//! │ - unregister()  │ ───────────────▶ │ - waits for input    │
//! └─────────────────┘                   │ - dispatches actions │
//!                                       └──────────────────────┘
//! ```
//!
//! A forwarder thread waits on one blocking `KeyboardListener` and sends each
//! key, stamped on arrival, down the commands' channel, so the manager thread
//! sleeps until either comes. It matches chords as handy-keys' `HotkeyManager`
//! does, whose events can only be polled, and keeps the listener's blocking
//! set in step so a chord's keys are swallowed. Commands (register/unregister)
//! are sent via an mpsc channel and responses are synchronously awaited.
//!
//! ## Recording Mode
//!
//! For UI key capture, a separate `KeyboardListener` is created on-demand and
//! polled from a dedicated recording thread. Events are emitted to the frontend
//! via Tauri's event system.

use handy_keys::{BlockingHotkeys, Hotkey, KeyEvent, KeyboardListener};
use log::{debug, error, info};
use serde::Serialize;
use specta::Type;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tauri::{AppHandle, Emitter, Manager};

use crate::settings::{self, get_settings, ShortcutBinding};
use crate::TranscriptionCoordinator;

use super::handler::handle_shortcut_event;
use super::modifier_trigger::{ModifierTrigger, TriggerAction};

/// The binding whose modifiers can trigger on their own.
const TRANSCRIBE: &str = "transcribe";

/// Commands that can be sent to the hotkey manager thread
enum ManagerCommand {
    Register {
        binding_id: String,
        hotkey_string: String,
        response: Sender<Result<(), String>>,
    },
    Unregister {
        binding_id: String,
        response: Sender<Result<(), String>>,
    },
    /// Re-read the "modifier alone" setting.
    RefreshModifierTrigger,
    Shutdown,
}

/// What wakes the manager thread.
enum Input {
    /// A key from the listener, stamped when it arrived.
    Key(KeyEvent, std::time::Instant),
    Command(ManagerCommand),
}

/// The command side of the manager thread's channel. The forwarder keeps the
/// channel open, so dropping this tells the thread to stop.
struct Commands(Sender<Input>);

impl Commands {
    fn send(&self, cmd: ManagerCommand) -> Result<(), mpsc::SendError<Input>> {
        self.0.send(Input::Command(cmd))
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.send(ManagerCommand::Shutdown);
    }
}

/// The "modifier alone" trigger while it is on: the state machine reading
/// every key the listener sees.
struct ModifierTriggerRuntime {
    machine: ModifierTrigger,
    /// The transcribe binding, e.g. "option+space".
    hotkey_string: String,
}

impl ModifierTriggerRuntime {
    /// The trigger for the transcribe binding `hotkey_string`, if the setting
    /// is on and the binding has modifiers and a key.
    fn build(effects: &impl Effects, hotkey_string: &str) -> Option<Self> {
        let hold_threshold = effects.modifier_trigger()?;
        let chord: Hotkey = hotkey_string.parse().ok()?;
        let machine = ModifierTrigger::new(chord, hold_threshold)?;
        info!("Modifier-alone trigger on for '{hotkey_string}'");
        Some(Self {
            machine,
            hotkey_string: hotkey_string.to_string(),
        })
    }

    /// Tear the trigger down, releasing a press still held.
    fn stop(mut self, effects: &impl Effects) {
        if let Some(action) = self.machine.abandon(std::time::Instant::now()) {
            effects.trigger(&self.hotkey_string, action);
        }
    }

    /// Replace `current` with the trigger the settings now ask for. A change
    /// of hold threshold alone keeps the running trigger.
    fn rebuild(current: &mut Option<Self>, effects: &impl Effects, hotkey_string: Option<&str>) {
        let hold_threshold = effects.modifier_trigger();
        if let Some((trigger, hold_threshold)) = current
            .as_mut()
            .filter(|t| Some(t.hotkey_string.as_str()) == hotkey_string)
            .zip(hold_threshold)
        {
            trigger.machine.set_hold_threshold(hold_threshold);
            return;
        }
        if let Some(old) = current.take() {
            old.stop(effects);
        }
        *current = hotkey_string.and_then(|hotkey| Self::build(effects, hotkey));
    }

    /// Feed the machine a key the listener saw at `at`.
    fn on_key(&mut self, effects: &impl Effects, event: &KeyEvent, at: std::time::Instant) {
        // Only a modifier going down can start anything. Secure Input
        // hides the keys that would discard a press. With no model, every
        // Option keystroke would raise "no model selected"; leave that
        // notice to the shortcut itself.
        let can_start = event.key.is_some() || !event.is_key_down || effects.can_start();
        if let Some(action) = self.machine.on_key_event(event, at, can_start) {
            effects.trigger(&self.hotkey_string, action);
        }
    }
}

fn apply_trigger_action(app: &AppHandle, hotkey_string: &str, action: TriggerAction) {
    debug!("modifier-alone trigger: {action:?}");
    if let TriggerAction::Chord { pressed } = action {
        handle_shortcut_event(app, TRANSCRIBE, hotkey_string, pressed);
        return;
    }
    let Some(coordinator) = app.try_state::<TranscriptionCoordinator>() else {
        return;
    };
    match action {
        TriggerAction::Press { at } | TriggerAction::Release { at } => {
            let settings = get_settings(app);
            let pressed = matches!(action, TriggerAction::Press { .. });
            coordinator.send_trigger_input(
                TRANSCRIBE,
                hotkey_string,
                pressed,
                settings.modifier_alone_activation,
                std::time::Duration::from_millis(settings.hold_threshold_ms),
                at,
            );
        }
        TriggerAction::Discard { pressed_at } => coordinator.discard_press(TRANSCRIBE, pressed_at),
        TriggerAction::Rebase { pressed_at, at } => coordinator.rebase_press(
            TRANSCRIBE,
            pressed_at,
            at,
            get_settings(app).shortcut_activation,
        ),
        TriggerAction::Chord { .. } => {}
    }
}

/// What the manager thread does to the rest of the app: the app itself in
/// production, a recorder in tests.
trait Effects {
    /// A registered binding's chord went down or up.
    fn shortcut(&self, binding_id: &str, hotkey_string: &str, pressed: bool);
    /// The modifier trigger of the transcribe binding `hotkey_string` acts.
    fn trigger(&self, hotkey_string: &str, action: TriggerAction);
    /// The hold threshold, while the "modifier alone" trigger is on.
    fn modifier_trigger(&self) -> Option<std::time::Duration>;
    /// Whether a bare modifier going down may start dictation now.
    fn can_start(&self) -> bool;
    /// The thread woke up to look for work.
    fn woke(&self) {}
}

impl Effects for AppHandle {
    fn shortcut(&self, binding_id: &str, hotkey_string: &str, pressed: bool) {
        handle_shortcut_event(self, binding_id, hotkey_string, pressed);
    }

    fn trigger(&self, hotkey_string: &str, action: TriggerAction) {
        apply_trigger_action(self, hotkey_string, action);
    }

    fn modifier_trigger(&self) -> Option<std::time::Duration> {
        // Elsewhere a bare modifier can't pass through harmlessly: on Windows a
        // lone Alt tap opens the menu bar of the app in front.
        if !cfg!(target_os = "macos") {
            return None;
        }
        let settings = get_settings(self);
        settings
            .modifier_alone_trigger
            .then(|| std::time::Duration::from_millis(settings.hold_threshold_ms))
    }

    fn can_start(&self) -> bool {
        !crate::secure_input::is_enabled_now() && !get_settings(self).selected_model.is_empty()
    }
}

/// The registered chords, matched against the listener's keys as handy-keys'
/// `HotkeyManager` matches them, and kept in the listener's blocking set so
/// their keys are swallowed.
#[derive(Default)]
struct Chords {
    blocking: BlockingHotkeys,
    hotkeys: HashMap<u32, Hotkey>,
    next_id: u32,
    /// Chords down now: a key's autorepeat presses nothing again.
    pressed: HashSet<u32>,
}

impl Chords {
    fn register(&mut self, hotkey: Hotkey) -> Result<u32, String> {
        if self.hotkeys.values().any(|h| *h == hotkey) {
            return Err(format!("Hotkey already registered: {hotkey}"));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.hotkeys.insert(id, hotkey);
        if let Ok(mut blocking) = self.blocking.lock() {
            blocking.insert(hotkey);
        }
        Ok(id)
    }

    fn unregister(&mut self, id: u32) -> Result<(), String> {
        let hotkey = self
            .hotkeys
            .remove(&id)
            .ok_or_else(|| format!("Hotkey with id {id} not found"))?;
        if let Ok(mut blocking) = self.blocking.lock() {
            blocking.remove(&hotkey);
        }
        Ok(())
    }

    /// The chords `event` presses (true) or releases (false).
    fn process(&mut self, event: &KeyEvent) -> Vec<(u32, bool)> {
        let mut edges = Vec::new();
        for (&id, hotkey) in &self.hotkeys {
            let edge = if event.is_key_down {
                hotkey.modifiers.matches(event.modifiers)
                    && hotkey.key == event.key
                    && self.pressed.insert(id)
            } else {
                // A modifier event releases a chord only once its modifiers
                // no longer match: tapping Shift while holding a Cmd-only
                // chord releases nothing.
                let released = match event.key {
                    Some(_) => hotkey.key == event.key,
                    None => !hotkey.modifiers.matches(event.modifiers),
                };
                released && self.pressed.remove(&id)
            };
            if edge {
                edges.push((id, event.is_key_down));
            }
        }
        edges
    }

    /// Let every key through: the listener outlives the manager thread.
    fn release_keys(&self) {
        if let Ok(mut blocking) = self.blocking.lock() {
            blocking.clear();
        }
    }
}

/// Hand each key from `next` to the manager thread as it comes, stamped on
/// arrival. Ends when the listener stops, or at the first key after the
/// manager thread has.
fn forward_keys(mut next: impl FnMut() -> Option<KeyEvent>, tx: Sender<Input>) {
    while let Some(event) = next() {
        if tx
            .send(Input::Key(event, std::time::Instant::now()))
            .is_err()
        {
            break;
        }
    }
}

/// State for the handy-keys shortcut manager
pub struct HandyKeysState {
    /// Channel to send commands to the manager thread (wrapped in Mutex for Sync)
    command_sender: Mutex<Commands>,
    /// Handle to the manager thread (wrapped in Mutex for Sync, allows proper join on drop)
    thread_handle: Mutex<Option<JoinHandle<()>>>,
    /// Recording listener for UI key capture (only active during recording)
    recording_listener: Mutex<Option<KeyboardListener>>,
    /// Flag indicating if we're in recording mode
    is_recording: AtomicBool,
    /// The binding ID being recorded (if any)
    recording_binding_id: Mutex<Option<String>>,
    /// Flag to stop recording loop
    recording_running: Arc<AtomicBool>,
}

/// Key event sent to frontend during recording mode
#[derive(Debug, Clone, Serialize, Type)]
pub struct FrontendKeyEvent {
    /// Currently pressed modifier keys
    pub modifiers: Vec<String>,
    /// The key that was pressed (if any)
    pub key: Option<String>,
    /// Whether this is a key down event
    pub is_key_down: bool,
    /// The full hotkey string (e.g., "option+space")
    pub hotkey_string: String,
}

impl HandyKeysState {
    /// Create a new HandyKeysState
    pub fn new(app: AppHandle) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<Input>();

        // Start the manager thread
        let app_clone = app.clone();
        // Create the listener here rather than inside the thread so a startup
        // failure (e.g. missing Accessibility permission) reaches the caller,
        // which can then fall back to the Tauri implementation.
        let blocking: BlockingHotkeys = Arc::new(Mutex::new(HashSet::new()));
        let listener = KeyboardListener::new_with_blocking(Arc::clone(&blocking))
            .map_err(|e| format!("Failed to start the keyboard listener: {e}"))?;
        // Never joined: it waits for the next key, and the listener goes
        // with it.
        let key_tx = tx.clone();
        thread::spawn(move || forward_keys(|| listener.recv().ok(), key_tx));
        let thread_handle = thread::spawn(move || {
            Self::manager_thread(rx, app_clone, blocking);
        });

        Ok(Self {
            command_sender: Mutex::new(Commands(tx)),
            thread_handle: Mutex::new(Some(thread_handle)),
            recording_listener: Mutex::new(None),
            is_recording: AtomicBool::new(false),
            recording_binding_id: Mutex::new(None),
            recording_running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The main manager thread - waits for keys and commands, and handles
    /// each as it comes. `blocking` is the listener's set of keys to swallow.
    fn manager_thread(input: Receiver<Input>, effects: impl Effects, blocking: BlockingHotkeys) {
        info!("handy-keys manager thread started");

        // Maps binding IDs to chord ids and hotkey strings
        let mut chords = Chords {
            blocking,
            ..Default::default()
        };
        let mut binding_to_hotkey: HashMap<String, u32> = HashMap::new();
        let mut hotkey_to_binding: HashMap<u32, (String, String)> = HashMap::new(); // (binding_id, hotkey_string)
        let mut modifier_trigger: Option<ModifierTriggerRuntime> = None;

        loop {
            let next = input.recv();
            effects.woke();
            match next {
                Ok(Input::Key(event, at)) => {
                    // Raw events first: a modifier went down before the chord
                    // it is part of, so the trigger must see it before the
                    // chord's event.
                    if let Some(trigger) = modifier_trigger.as_mut() {
                        trigger.on_key(&effects, &event, at);
                    }

                    for (id, is_pressed) in chords.process(&event) {
                        if let Some((binding_id, hotkey_string)) = hotkey_to_binding.get(&id) {
                            debug!(
                                "handy-keys event: binding={}, hotkey={}, pressed={}",
                                binding_id, hotkey_string, is_pressed
                            );
                            if let Some(trigger) = modifier_trigger
                                .as_mut()
                                .filter(|_| binding_id == TRANSCRIBE)
                            {
                                if let Some(action) = trigger.machine.on_chord(is_pressed, at) {
                                    effects.trigger(hotkey_string, action);
                                }
                                continue;
                            }
                            effects.shortcut(binding_id, hotkey_string, is_pressed);
                        }
                    }
                }
                Ok(Input::Command(cmd)) => match cmd {
                    ManagerCommand::Register {
                        binding_id,
                        hotkey_string,
                        response,
                    } => {
                        let result = Self::do_register(
                            &mut chords,
                            &mut binding_to_hotkey,
                            &mut hotkey_to_binding,
                            &binding_id,
                            &hotkey_string,
                        );
                        if result.is_ok() && binding_id == TRANSCRIBE {
                            ModifierTriggerRuntime::rebuild(
                                &mut modifier_trigger,
                                &effects,
                                Some(&hotkey_string),
                            );
                        }
                        let _ = response.send(result);
                    }
                    ManagerCommand::Unregister {
                        binding_id,
                        response,
                    } => {
                        let result = Self::do_unregister(
                            &mut chords,
                            &mut binding_to_hotkey,
                            &mut hotkey_to_binding,
                            &binding_id,
                        );
                        if binding_id == TRANSCRIBE {
                            // Also while capturing a new shortcut: pressing
                            // Option then must not start dictation.
                            ModifierTriggerRuntime::rebuild(&mut modifier_trigger, &effects, None);
                        }
                        let _ = response.send(result);
                    }
                    ManagerCommand::RefreshModifierTrigger => {
                        let transcribe = hotkey_to_binding
                            .values()
                            .find(|(id, _)| id == TRANSCRIBE)
                            .map(|(_, hotkey)| hotkey.clone());
                        ModifierTriggerRuntime::rebuild(
                            &mut modifier_trigger,
                            &effects,
                            transcribe.as_deref(),
                        );
                    }
                    ManagerCommand::Shutdown => {
                        info!("handy-keys manager thread shutting down");
                        break;
                    }
                },
                Err(mpsc::RecvError) => {
                    info!("Command channel disconnected, shutting down");
                    break;
                }
            }
        }

        chords.release_keys();
        info!("handy-keys manager thread stopped");
    }

    /// Register a hotkey
    fn do_register(
        chords: &mut Chords,
        binding_to_hotkey: &mut HashMap<String, u32>,
        hotkey_to_binding: &mut HashMap<u32, (String, String)>,
        binding_id: &str,
        hotkey_string: &str,
    ) -> Result<(), String> {
        let hotkey: Hotkey = hotkey_string
            .parse()
            .map_err(|e| format!("Failed to parse hotkey '{}': {}", hotkey_string, e))?;

        let id = chords
            .register(hotkey)
            .map_err(|e| format!("Failed to register hotkey: {}", e))?;

        binding_to_hotkey.insert(binding_id.to_string(), id);
        hotkey_to_binding.insert(id, (binding_id.to_string(), hotkey_string.to_string()));

        debug!(
            "Registered handy-keys shortcut: {} -> {:?}",
            binding_id, hotkey
        );
        Ok(())
    }

    /// Unregister a hotkey
    fn do_unregister(
        chords: &mut Chords,
        binding_to_hotkey: &mut HashMap<String, u32>,
        hotkey_to_binding: &mut HashMap<u32, (String, String)>,
        binding_id: &str,
    ) -> Result<(), String> {
        if let Some(id) = binding_to_hotkey.remove(binding_id) {
            chords
                .unregister(id)
                .map_err(|e| format!("Failed to unregister hotkey: {}", e))?;
            hotkey_to_binding.remove(&id);
            debug!("Unregistered handy-keys shortcut: {}", binding_id);
        }
        Ok(())
    }

    /// Register a shortcut binding
    pub fn register(&self, binding: &ShortcutBinding) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.command_sender
            .lock()
            .map_err(|_| "Failed to lock command_sender")?
            .send(ManagerCommand::Register {
                binding_id: binding.id.clone(),
                hotkey_string: binding.current_binding.clone(),
                response: tx,
            })
            .map_err(|_| "Failed to send register command")?;

        rx.recv()
            .map_err(|_| "Failed to receive register response")?
    }

    /// Unregister a shortcut binding
    pub fn unregister(&self, binding: &ShortcutBinding) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.command_sender
            .lock()
            .map_err(|_| "Failed to lock command_sender")?
            .send(ManagerCommand::Unregister {
                binding_id: binding.id.clone(),
                response: tx,
            })
            .map_err(|_| "Failed to send unregister command")?;

        rx.recv()
            .map_err(|_| "Failed to receive unregister response")?
    }

    /// Apply a change to the "modifier alone" setting.
    pub fn refresh_modifier_trigger(&self) -> Result<(), String> {
        self.command_sender
            .lock()
            .map_err(|_| "Failed to lock command_sender")?
            .send(ManagerCommand::RefreshModifierTrigger)
            .map_err(|_| "Failed to send refresh command".to_string())
    }

    /// Start recording mode for a specific binding
    pub fn start_recording(&self, app: &AppHandle, binding_id: String) -> Result<(), String> {
        if self.is_recording.load(Ordering::SeqCst) {
            return Err("Already recording".into());
        }

        // Create a new keyboard listener for recording
        let listener = KeyboardListener::new()
            .map_err(|e| format!("Failed to create keyboard listener: {}", e))?;

        {
            let mut recording = self
                .recording_listener
                .lock()
                .map_err(|_| "Failed to lock recording_listener")?;
            *recording = Some(listener);
        }
        {
            let mut binding = self
                .recording_binding_id
                .lock()
                .map_err(|_| "Failed to lock recording_binding_id")?;
            *binding = Some(binding_id);
        }

        self.is_recording.store(true, Ordering::SeqCst);
        self.recording_running.store(true, Ordering::SeqCst);

        // Start a thread to emit key events to the frontend
        let app_clone = app.clone();
        let recording_running = Arc::clone(&self.recording_running);
        thread::spawn(move || {
            Self::recording_loop(app_clone, recording_running);
        });

        debug!("Started handy-keys recording mode");
        Ok(())
    }

    /// Recording loop - emits key events to frontend during recording
    fn recording_loop(app: AppHandle, running: Arc<AtomicBool>) {
        while running.load(Ordering::SeqCst) {
            let event = {
                let state = match app.try_state::<HandyKeysState>() {
                    Some(s) => s,
                    None => break,
                };
                let listener = state.recording_listener.lock().ok();
                listener.as_ref().and_then(|l| l.as_ref()?.try_recv())
            };

            if let Some(key_event) = event {
                // Convert to frontend-friendly format
                let frontend_event = FrontendKeyEvent {
                    modifiers: modifiers_to_strings(key_event.modifiers),
                    key: key_event.key.map(|k| k.to_string().to_lowercase()),
                    is_key_down: key_event.is_key_down,
                    hotkey_string: key_event
                        .as_hotkey()
                        .map(|h| h.to_handy_string())
                        .unwrap_or_default(),
                };

                // Emit to frontend
                if let Err(e) = app.emit("handy-keys-event", &frontend_event) {
                    error!("Failed to emit key event: {}", e);
                }
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        debug!("Recording loop ended");
    }

    /// Stop recording mode
    pub fn stop_recording(&self) -> Result<(), String> {
        self.is_recording.store(false, Ordering::SeqCst);
        self.recording_running.store(false, Ordering::SeqCst);

        {
            let mut recording = self
                .recording_listener
                .lock()
                .map_err(|_| "Failed to lock recording_listener")?;
            *recording = None;
        }
        {
            let mut binding = self
                .recording_binding_id
                .lock()
                .map_err(|_| "Failed to lock recording_binding_id")?;
            *binding = None;
        }

        debug!("Stopped handy-keys recording mode");
        Ok(())
    }
}

impl Drop for HandyKeysState {
    fn drop(&mut self) {
        // Signal recording to stop
        self.recording_running.store(false, Ordering::SeqCst);
        self.is_recording.store(false, Ordering::SeqCst);

        // Send shutdown command
        if let Ok(sender) = self.command_sender.lock() {
            let _ = sender.send(ManagerCommand::Shutdown);
        }

        // Wait for the manager thread to finish
        if let Ok(mut handle) = self.thread_handle.lock() {
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }
    }
}

/// Convert handy-keys Modifiers to a list of strings
fn modifiers_to_strings(modifiers: handy_keys::Modifiers) -> Vec<String> {
    let mut result = Vec::new();

    if modifiers.contains(handy_keys::Modifiers::CTRL) {
        result.push("ctrl".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::OPT) {
        #[cfg(target_os = "macos")]
        result.push("option".to_string());
        #[cfg(not(target_os = "macos"))]
        result.push("alt".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::SHIFT) {
        result.push("shift".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::CMD) {
        #[cfg(target_os = "macos")]
        result.push("command".to_string());
        #[cfg(not(target_os = "macos"))]
        result.push("super".to_string());
    }
    if modifiers.contains(handy_keys::Modifiers::FN) {
        result.push("fn".to_string());
    }

    result
}

/// Validate a shortcut string for the HandyKeys implementation.
/// HandyKeys is more permissive: allows modifier-only combos and the fn key.
pub fn validate_shortcut(raw: &str) -> Result<(), String> {
    if raw.trim().is_empty() {
        return Err("Shortcut cannot be empty".into());
    }
    // HandyKeys accepts modifier-only, key-only, and modifier+key combos
    // Just verify the string is parseable
    raw.parse::<Hotkey>()
        .map(|_| ())
        .map_err(|e| format!("Invalid shortcut for HandyKeys: {}", e))
}

/// Initialize handy-keys shortcuts
pub fn init_shortcuts(app: &AppHandle) -> Result<(), String> {
    let state = HandyKeysState::new(app.clone())?;

    let default_bindings = settings::get_default_settings().bindings;
    let user_settings = settings::load_or_create_app_settings(app);

    // Register all bindings except cancel (which is dynamic)
    for (id, default_binding) in default_bindings {
        if id == "cancel" {
            continue;
        }
        // Skip post-processing shortcut when the feature is disabled
        if id == "transcribe_with_post_process" && !user_settings.post_process_enabled {
            continue;
        }

        let binding = user_settings
            .bindings
            .get(&id)
            .cloned()
            .unwrap_or(default_binding);

        if let Err(e) = state.register(&binding) {
            error!(
                "Failed to register handy-keys shortcut {} during init: {}",
                id, e
            );
        }
    }

    app.manage(state);
    info!("handy-keys shortcuts initialized");
    Ok(())
}

/// Register a shortcut
pub fn register_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.register(&binding)
}

/// Unregister a shortcut
pub fn unregister_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;
    state.unregister(&binding)
}

/// Apply a change to the "modifier alone" setting, if handy-keys is running.
pub fn refresh_modifier_trigger(app: &AppHandle) -> Result<(), String> {
    match app.try_state::<HandyKeysState>() {
        Some(state) => state.refresh_modifier_trigger(),
        None => Ok(()),
    }
}

/// Start key recording mode
#[tauri::command]
#[specta::specta]
pub fn start_handy_keys_recording(app: AppHandle, binding_id: String) -> Result<(), String> {
    let settings = get_settings(&app);
    if settings.keyboard_implementation != settings::KeyboardImplementation::HandyKeys {
        return Err("handy-keys is not the active keyboard implementation".into());
    }

    // While Secure Input is active the tap receives no KeyDown/KeyUp, so the
    // recorder would silently capture just the modifier and overwrite the
    // binding with it (issue #1578). Refuse instead; the frontend maps this
    // marker to a localized explanation, and the noted impact makes the
    // warning banner appear with the full story.
    if crate::secure_input::is_enabled_now() {
        crate::secure_input::note_recorder_blocked(&app);
        return Err("secure-input-active".into());
    }

    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;

    // Suspend every registered shortcut so a combo that overlaps an existing
    // binding can't fire it (or have its keys swallowed) mid-capture.
    super::suspend_all_shortcuts(&app);

    let result = state.start_recording(&app, binding_id);
    if result.is_err() {
        super::resume_all_shortcuts(&app);
    }
    result
}

/// Stop key recording mode
#[tauri::command]
#[specta::specta]
pub fn stop_handy_keys_recording(app: AppHandle) -> Result<(), String> {
    let settings = get_settings(&app);
    if settings.keyboard_implementation != settings::KeyboardImplementation::HandyKeys {
        return Err("handy-keys is not the active keyboard implementation".into());
    }

    let state = app
        .try_state::<HandyKeysState>()
        .ok_or("HandyKeysState not initialized")?;

    // Restore shortcuts from settings regardless of how recording ended.
    // A commit has already registered the new binding via change_binding;
    // re-registering it here fails cleanly and is ignored.
    let result = state.stop_recording();
    super::resume_all_shortcuts(&app);
    result
}

#[cfg(test)]
mod tests {
    //! The manager thread and its forwarder driven through `manager_thread`
    //! and `forward_keys` with a fake listener. Real listeners need the
    //! Accessibility permission, so none is created here.

    use super::*;
    use handy_keys::{Key, Modifiers};
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    const THRESHOLD: Duration = Duration::from_millis(300);
    const OPT: Modifiers = Modifiers::OPT_LEFT;
    const WAIT: Duration = Duration::from_secs(2);

    /// What the thread asked of the app.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Did {
        Shortcut(&'static str, bool),
        Trigger(TriggerAction),
    }

    struct Recorder {
        did: Sender<(Did, Instant)>,
        wakes: Arc<AtomicUsize>,
        trigger: Option<Duration>,
    }

    fn binding(id: &str) -> &'static str {
        match id {
            "transcribe" => "transcribe",
            "cancel" => "cancel",
            _ => "other",
        }
    }

    impl Effects for Recorder {
        fn shortcut(&self, binding_id: &str, _: &str, pressed: bool) {
            let _ = self
                .did
                .send((Did::Shortcut(binding(binding_id), pressed), Instant::now()));
        }
        fn trigger(&self, _: &str, action: TriggerAction) {
            let _ = self.did.send((Did::Trigger(action), Instant::now()));
        }
        fn modifier_trigger(&self) -> Option<Duration> {
            self.trigger
        }
        fn can_start(&self) -> bool {
            true
        }
        fn woke(&self) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Harness {
        cmd: Option<Commands>,
        /// The fake listener: what the forwarder reads.
        keys: Option<Sender<KeyEvent>>,
        /// How many times the forwarder asked the listener for a key.
        pulls: Arc<AtomicUsize>,
        blocking: BlockingHotkeys,
        did: Receiver<(Did, Instant)>,
        wakes: Arc<AtomicUsize>,
        thread: Option<JoinHandle<()>>,
        forwarder: Option<JoinHandle<()>>,
    }

    impl Harness {
        fn start(trigger: Option<Duration>) -> Self {
            let (tx, rx) = mpsc::channel();
            let (key_tx, key_rx) = mpsc::channel::<KeyEvent>();
            let (did_tx, did_rx) = mpsc::channel();
            let wakes = Arc::new(AtomicUsize::new(0));
            let pulls = Arc::new(AtomicUsize::new(0));
            let blocking: BlockingHotkeys = Arc::new(Mutex::new(HashSet::new()));
            let effects = Recorder {
                did: did_tx,
                wakes: Arc::clone(&wakes),
                trigger,
            };
            let counter = Arc::clone(&pulls);
            let forward_tx = tx.clone();
            let forwarder = thread::spawn(move || {
                forward_keys(
                    || {
                        counter.fetch_add(1, Ordering::SeqCst);
                        key_rx.recv().ok()
                    },
                    forward_tx,
                )
            });
            let set = Arc::clone(&blocking);
            let thread = thread::spawn(move || HandyKeysState::manager_thread(rx, effects, set));
            Self {
                cmd: Some(Commands(tx)),
                keys: Some(key_tx),
                pulls,
                blocking,
                did: did_rx,
                wakes,
                thread: Some(thread),
                forwarder: Some(forwarder),
            }
        }

        fn command(&self, cmd: ManagerCommand) {
            assert!(self.cmd.as_ref().unwrap().send(cmd).is_ok());
        }

        fn try_register(&self, binding_id: &str, hotkey: &str) -> Result<(), String> {
            let (tx, rx) = mpsc::channel();
            self.command(ManagerCommand::Register {
                binding_id: binding_id.into(),
                hotkey_string: hotkey.into(),
                response: tx,
            });
            rx.recv_timeout(WAIT).unwrap()
        }

        fn register(&self, binding_id: &str, hotkey: &str) {
            self.try_register(binding_id, hotkey).unwrap();
        }

        fn unregister(&self, binding_id: &str) {
            let (tx, rx) = mpsc::channel();
            self.command(ManagerCommand::Unregister {
                binding_id: binding_id.into(),
                response: tx,
            });
            rx.recv_timeout(WAIT).unwrap().unwrap();
        }

        /// A key as the listener reports it.
        fn key(&self, event: KeyEvent) {
            self.keys.as_ref().unwrap().send(event).unwrap();
        }

        fn next(&self) -> (Did, Instant) {
            self.did.recv_timeout(WAIT).expect("nothing dispatched")
        }

        fn assert_quiet(&self) {
            if let Ok(did) = self.did.recv_timeout(Duration::from_millis(50)) {
                panic!("unexpected {did:?}");
            }
        }

        fn swallowed(&self) -> HashSet<Hotkey> {
            self.blocking.lock().unwrap().clone()
        }

        /// Whether the manager thread ends within `WAIT`.
        fn stops(&mut self) -> bool {
            ends(self.thread.take().unwrap())
        }
    }

    fn ends(thread: JoinHandle<()>) -> bool {
        let t0 = Instant::now();
        while !thread.is_finished() && t0.elapsed() < WAIT {
            thread::sleep(Duration::from_millis(1));
        }
        thread.is_finished()
    }

    fn mods(modifiers: Modifiers, down: bool) -> KeyEvent {
        KeyEvent {
            modifiers,
            key: None,
            is_key_down: down,
            changed_modifier: None,
        }
    }

    fn key(modifiers: Modifiers, key: Key, down: bool) -> KeyEvent {
        KeyEvent {
            modifiers,
            key: Some(key),
            is_key_down: down,
            changed_modifier: None,
        }
    }

    fn hotkey(s: &str) -> Hotkey {
        s.parse().unwrap()
    }

    fn percentile(sorted: &[Duration], p: usize) -> Duration {
        sorted[(sorted.len() - 1) * p / 100]
    }

    #[test]
    fn idle_thread_wakeups_in_one_second() {
        let h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        thread::sleep(Duration::from_millis(100));
        let before = h.wakes.load(Ordering::SeqCst);
        thread::sleep(Duration::from_secs(1));
        let wakes = h.wakes.load(Ordering::SeqCst) - before;
        println!("idle wakeups in 1 s: {wakes}");
        assert_eq!(wakes, 0);
        // The forwarder is still waiting for its first key.
        assert_eq!(h.pulls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn key_to_dispatch_latency() {
        let h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        let mut dispatch = Vec::new();
        let mut stamp = Vec::new();
        for i in 0..200u64 {
            // Land at a different point of any poll period each time.
            thread::sleep(Duration::from_micros(200 + (i * 7919) % 10_000));
            let t0 = Instant::now();
            h.key(mods(OPT, true));
            let (did, done) = h.next();
            let Did::Trigger(TriggerAction::Press { at }) = did else {
                panic!("{did:?}");
            };
            dispatch.push(done - t0);
            stamp.push(at - t0);
            h.key(mods(Modifiers::empty(), false));
            assert!(matches!(
                h.next().0,
                Did::Trigger(TriggerAction::Release { .. })
            ));
        }
        dispatch.sort();
        stamp.sort();
        println!(
            "key -> dispatch over 200: median {:?}, max {:?}; key -> press timestamp: median {:?}, max {:?}",
            percentile(&dispatch, 50),
            dispatch[199],
            percentile(&stamp, 50),
            stamp[199],
        );
        // Loose: the poll it replaces put the median near 5 ms.
        assert!(percentile(&dispatch, 50) < Duration::from_millis(2));
    }

    #[test]
    fn raw_keys_reach_the_trigger_before_the_chord() {
        let h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        // Option and Space, both queued before the thread looks: Space is
        // both a raw key and the chord's press.
        h.key(mods(OPT, true));
        h.key(key(OPT, Key::Space, true));
        assert!(matches!(
            h.next().0,
            Did::Trigger(TriggerAction::Press { .. })
        ));
        assert!(matches!(
            h.next().0,
            Did::Trigger(TriggerAction::Rebase { .. })
        ));
        // The chord's press, which the trigger already took over, is
        // absorbed; its release passes on.
        h.assert_quiet();
        h.key(key(OPT, Key::Space, false));
        assert_eq!(
            h.next().0,
            Did::Trigger(TriggerAction::Chord { pressed: false })
        );
        h.key(mods(Modifiers::empty(), false));
        h.assert_quiet();
    }

    #[test]
    fn chord_only_bindings_dispatch_without_a_trigger() {
        let h = Harness::start(None);
        h.register(TRANSCRIBE, "option+space");
        h.register("cancel", "escape");
        h.key(key(Modifiers::empty(), Key::Escape, true));
        assert_eq!(h.next().0, Did::Shortcut("cancel", true));
        // Autorepeat presses nothing again.
        h.key(key(Modifiers::empty(), Key::Escape, true));
        h.key(key(Modifiers::empty(), Key::Escape, false));
        assert_eq!(h.next().0, Did::Shortcut("cancel", false));
        h.key(mods(OPT, true));
        h.key(key(OPT, Key::Space, true));
        assert_eq!(h.next().0, Did::Shortcut("transcribe", true));
        h.assert_quiet();
    }

    #[test]
    fn commands_are_answered_while_keys_are_idle() {
        let h = Harness::start(Some(THRESHOLD));
        for n in 0..20 {
            let t0 = Instant::now();
            h.register(&format!("binding{n}"), &format!("ctrl+f{}", n + 1));
            assert!(t0.elapsed() < Duration::from_millis(500));
        }
        h.unregister("binding0");
    }

    #[test]
    fn registered_chords_are_swallowed_until_unregistered() {
        let h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        h.register("cancel", "escape");
        assert_eq!(
            h.swallowed(),
            HashSet::from([hotkey("option+space"), hotkey("escape")])
        );
        // The same chord twice is refused and changes nothing.
        let err = h.try_register("other", "option+space").unwrap_err();
        assert!(err.starts_with("Failed to register hotkey: Hotkey already registered"));
        h.unregister("cancel");
        assert_eq!(h.swallowed(), HashSet::from([hotkey("option+space")]));
        h.key(key(Modifiers::empty(), Key::Escape, true));
        h.assert_quiet();
    }

    #[test]
    fn stops_on_shutdown_and_lets_keys_through() {
        let mut h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        h.command(ManagerCommand::Shutdown);
        assert!(h.stops());
        assert!(h.swallowed().is_empty());
        // The forwarder waits without spinning, and ends at the next key.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(h.pulls.load(Ordering::SeqCst), 1);
        h.key(mods(OPT, true));
        assert!(ends(h.forwarder.take().unwrap()));
        // It took that key and asked for no other.
        assert_eq!(h.pulls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stops_when_the_command_sender_is_dropped() {
        let mut h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        // The forwarder still holds the channel open.
        h.cmd = None;
        assert!(h.stops());
        assert!(!h.forwarder.as_ref().unwrap().is_finished());
    }

    #[test]
    fn forwarder_ends_when_the_listener_stops() {
        let mut h = Harness::start(None);
        h.keys = None;
        assert!(ends(h.forwarder.take().unwrap()));
        // The manager thread still answers.
        h.register("cancel", "escape");
    }

    /// Chord matching, as handy-keys' `HotkeyManager` does it.
    mod chords {
        use super::*;

        fn chords(hotkeys: &[&str]) -> Chords {
            let mut chords = Chords::default();
            for h in hotkeys {
                chords.register(hotkey(h)).unwrap();
            }
            chords
        }

        #[test]
        fn press_then_release_by_key() {
            let mut c = chords(&["cmd+k"]);
            let down = key(Modifiers::CMD_LEFT, Key::K, true);
            assert_eq!(c.process(&down), vec![(0, true)]);
            // Autorepeat.
            assert_eq!(c.process(&down), vec![]);
            let up = key(Modifiers::CMD_LEFT, Key::K, false);
            assert_eq!(c.process(&up), vec![(0, false)]);
            assert_eq!(c.process(&up), vec![]);
        }

        #[test]
        fn modifier_release_releases_the_chord() {
            let mut c = chords(&["cmd+k"]);
            c.process(&key(Modifiers::CMD_LEFT, Key::K, true));
            assert_eq!(
                c.process(&mods(Modifiers::empty(), false)),
                vec![(0, false)]
            );
        }

        #[test]
        fn wrong_or_extra_modifiers_do_not_press() {
            let mut c = chords(&["cmd+k", "f1"]);
            assert_eq!(c.process(&key(Modifiers::SHIFT_LEFT, Key::K, true)), vec![]);
            assert_eq!(c.process(&key(Modifiers::CMD_LEFT, Key::F1, true)), vec![]);
            assert_eq!(
                c.process(&key(Modifiers::empty(), Key::F1, true)),
                vec![(1, true)]
            );
        }

        #[test]
        fn same_key_different_modifiers() {
            let mut c = chords(&["cmd+k", "ctrl+k"]);
            assert_eq!(
                c.process(&key(Modifiers::CTRL_LEFT, Key::K, true)),
                vec![(1, true)]
            );
        }

        #[test]
        fn modifier_only_chord_survives_an_unrelated_modifier() {
            let mut c = chords(&["cmd"]);
            assert_eq!(c.process(&mods(Modifiers::CMD_LEFT, true)), vec![(0, true)]);
            let both = Modifiers::CMD_LEFT | Modifiers::SHIFT_LEFT;
            assert_eq!(c.process(&mods(both, true)), vec![]);
            assert_eq!(c.process(&mods(Modifiers::CMD_LEFT, false)), vec![]);
            assert_eq!(
                c.process(&mods(Modifiers::empty(), false)),
                vec![(0, false)]
            );
        }

        #[test]
        fn unregistered_chords_stop_matching() {
            let mut c = chords(&["cmd+k"]);
            c.unregister(0).unwrap();
            assert!(c.unregister(0).is_err());
            assert_eq!(c.process(&key(Modifiers::CMD_LEFT, Key::K, true)), vec![]);
        }
    }
}
