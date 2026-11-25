//! Handy-keys based keyboard shortcut implementation
//!
//! This module provides an alternative to Tauri's global-shortcut plugin
//! using the handy-keys library for more control over keyboard events.
//!
//! ## Architecture
//!
//! The implementation uses a dedicated manager thread that owns the `HotkeyManager`:
//!
//! ```text
//! ┌─────────────────┐     commands      ┌──────────────────────┐
//! │   Main Thread   │ ───────────────▶ │   Manager Thread     │
//! │                 │   (via channel)   │                      │
//! │ - register()    │                   │ - owns HotkeyManager │
//! │ - unregister()  │                   │ - polls for events   │
//! └─────────────────┘                   │ - dispatches actions │
//!                                       └──────────────────────┘
//! ```
//!
//! This design ensures thread-safety since `HotkeyManager` is only accessed
//! from a single thread. Commands (register/unregister) are sent via an mpsc
//! channel and responses are synchronously awaited.
//!
//! ## Recording Mode
//!
//! For UI key capture, a separate `KeyboardListener` is created on-demand and
//! polled from a dedicated recording thread. Events are emitted to the frontend
//! via Tauri's event system.

use handy_keys::{Hotkey, HotkeyId, HotkeyManager, HotkeyState, KeyEvent, KeyboardListener};
use log::{debug, error, info};
use serde::Serialize;
use specta::Type;
use std::collections::HashMap;
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

/// The "modifier alone" trigger while it is on: a listener that sees every
/// key (it blocks nothing) and the state machine reading it.
struct ModifierTriggerRuntime {
    listener: Box<dyn RawSource>,
    machine: ModifierTrigger,
    /// The transcribe binding, e.g. "option+space".
    hotkey_string: String,
}

impl ModifierTriggerRuntime {
    /// The trigger for the transcribe binding `hotkey_string`, if the setting
    /// is on and the binding has modifiers and a key.
    fn build(effects: &impl Effects, keys: &impl KeySource, hotkey_string: &str) -> Option<Self> {
        let hold_threshold = effects.modifier_trigger()?;
        let chord: Hotkey = hotkey_string.parse().ok()?;
        let machine = ModifierTrigger::new(chord, hold_threshold)?;
        // Created after the manager's tap: both insert at the head of the
        // session taps, so this one sees the chord's key before the manager's
        // blocking tap swallows it, which the raw-stream takeover relies on.
        let listener = match keys.raw_listener() {
            Ok(listener) => listener,
            Err(e) => {
                error!("Failed to start the modifier-alone listener: {e}");
                return None;
            }
        };
        info!("Modifier-alone trigger on for '{hotkey_string}'");
        Some(Self {
            listener,
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
    /// of hold threshold alone keeps the running trigger and its listener.
    fn rebuild(
        current: &mut Option<Self>,
        effects: &impl Effects,
        keys: &impl KeySource,
        hotkey_string: Option<&str>,
    ) {
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
        *current = hotkey_string.and_then(|hotkey| Self::build(effects, keys, hotkey));
    }

    /// Feed the listener's queued events to the machine.
    fn drain(&mut self, effects: &impl Effects) {
        while let Some(event) = self.listener.try_recv() {
            // Only a modifier going down can start anything. Secure Input
            // hides the keys that would discard a press. With no model, every
            // Option keystroke would raise "no model selected"; leave that
            // notice to the shortcut itself.
            let can_start = event.key.is_some() || !event.is_key_down || effects.can_start();
            if let Some(action) =
                self.machine
                    .on_key_event(&event, std::time::Instant::now(), can_start)
            {
                effects.trigger(&self.hotkey_string, action);
            }
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

/// What the shortcut thread does to the rest of the app: the app itself in
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

/// The hotkey manager and the trigger's raw listener, as the thread uses them.
trait KeySource {
    type Id: Copy + Eq + std::hash::Hash;
    fn register(&self, hotkey: Hotkey) -> Result<Self::Id, String>;
    fn unregister(&self, id: Self::Id) -> Result<(), String>;
    /// A queued chord edge: the hotkey and whether it went down.
    fn try_recv(&self) -> Option<(Self::Id, bool)>;
    /// A listener that sees every key and blocks none.
    fn raw_listener(&self) -> Result<Box<dyn RawSource>, String>;
}

trait RawSource {
    fn try_recv(&self) -> Option<KeyEvent>;
}

impl RawSource for KeyboardListener {
    fn try_recv(&self) -> Option<KeyEvent> {
        KeyboardListener::try_recv(self)
    }
}

impl KeySource for HotkeyManager {
    type Id = HotkeyId;

    fn register(&self, hotkey: Hotkey) -> Result<HotkeyId, String> {
        HotkeyManager::register(self, hotkey).map_err(|e| e.to_string())
    }

    fn unregister(&self, id: HotkeyId) -> Result<(), String> {
        HotkeyManager::unregister(self, id).map_err(|e| e.to_string())
    }

    fn try_recv(&self) -> Option<(HotkeyId, bool)> {
        HotkeyManager::try_recv(self).map(|event| (event.id, event.state == HotkeyState::Pressed))
    }

    fn raw_listener(&self) -> Result<Box<dyn RawSource>, String> {
        Ok(Box::new(
            KeyboardListener::new().map_err(|e| e.to_string())?,
        ))
    }
}

/// State for the handy-keys shortcut manager
pub struct HandyKeysState {
    /// Channel to send commands to the manager thread (wrapped in Mutex for Sync)
    command_sender: Mutex<Sender<ManagerCommand>>,
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
        let (cmd_tx, cmd_rx) = mpsc::channel::<ManagerCommand>();

        // Start the manager thread
        let app_clone = app.clone();
        // Create the manager here rather than inside the thread so a startup
        // failure (e.g. missing Accessibility permission) reaches the caller,
        // which can then fall back to the Tauri implementation.
        let manager = HotkeyManager::new_with_blocking()
            .map_err(|e| format!("Failed to create HotkeyManager: {e}"))?;
        let thread_handle = thread::spawn(move || {
            Self::manager_thread(cmd_rx, app_clone, manager);
        });

        Ok(Self {
            command_sender: Mutex::new(cmd_tx),
            thread_handle: Mutex::new(Some(thread_handle)),
            recording_listener: Mutex::new(None),
            is_recording: AtomicBool::new(false),
            recording_binding_id: Mutex::new(None),
            recording_running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The main manager thread - owns the HotkeyManager and processes commands
    fn manager_thread<K: KeySource>(
        cmd_rx: Receiver<ManagerCommand>,
        effects: impl Effects,
        manager: K,
    ) {
        info!("handy-keys manager thread started");

        // Maps binding IDs to HotkeyIds and hotkey strings
        let mut binding_to_hotkey: HashMap<String, K::Id> = HashMap::new();
        let mut hotkey_to_binding: HashMap<K::Id, (String, String)> = HashMap::new(); // (binding_id, hotkey_string)
        let mut modifier_trigger: Option<ModifierTriggerRuntime> = None;

        loop {
            effects.woke();
            // Raw events first: a modifier went down before the chord it is
            // part of, so the trigger must see it before the chord's event.
            if let Some(trigger) = modifier_trigger.as_mut() {
                trigger.drain(&effects);
            }

            // Check for hotkey events (non-blocking)
            while let Some((id, is_pressed)) = manager.try_recv() {
                if let Some((binding_id, hotkey_string)) = hotkey_to_binding.get(&id) {
                    debug!(
                        "handy-keys event: binding={}, hotkey={}, pressed={}",
                        binding_id, hotkey_string, is_pressed
                    );
                    if let Some(trigger) = modifier_trigger
                        .as_mut()
                        .filter(|_| binding_id == TRANSCRIBE)
                    {
                        if let Some(action) = trigger
                            .machine
                            .on_chord(is_pressed, std::time::Instant::now())
                        {
                            effects.trigger(hotkey_string, action);
                        }
                        continue;
                    }
                    effects.shortcut(binding_id, hotkey_string, is_pressed);
                }
            }

            // Check for commands (non-blocking with timeout)
            match cmd_rx.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(cmd) => match cmd {
                    ManagerCommand::Register {
                        binding_id,
                        hotkey_string,
                        response,
                    } => {
                        let result = Self::do_register(
                            &manager,
                            &mut binding_to_hotkey,
                            &mut hotkey_to_binding,
                            &binding_id,
                            &hotkey_string,
                        );
                        if result.is_ok() && binding_id == TRANSCRIBE {
                            ModifierTriggerRuntime::rebuild(
                                &mut modifier_trigger,
                                &effects,
                                &manager,
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
                            &manager,
                            &mut binding_to_hotkey,
                            &mut hotkey_to_binding,
                            &binding_id,
                        );
                        if binding_id == TRANSCRIBE {
                            // Also while capturing a new shortcut: pressing
                            // Option then must not start dictation.
                            ModifierTriggerRuntime::rebuild(
                                &mut modifier_trigger,
                                &effects,
                                &manager,
                                None,
                            );
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
                            &manager,
                            transcribe.as_deref(),
                        );
                    }
                    ManagerCommand::Shutdown => {
                        info!("handy-keys manager thread shutting down");
                        break;
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No command, continue
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    info!("Command channel disconnected, shutting down");
                    break;
                }
            }
        }

        info!("handy-keys manager thread stopped");
    }

    /// Register a hotkey
    fn do_register<K: KeySource>(
        manager: &K,
        binding_to_hotkey: &mut HashMap<String, K::Id>,
        hotkey_to_binding: &mut HashMap<K::Id, (String, String)>,
        binding_id: &str,
        hotkey_string: &str,
    ) -> Result<(), String> {
        let hotkey: Hotkey = hotkey_string
            .parse()
            .map_err(|e| format!("Failed to parse hotkey '{}': {}", hotkey_string, e))?;

        let id = manager
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
    fn do_unregister<K: KeySource>(
        manager: &K,
        binding_to_hotkey: &mut HashMap<String, K::Id>,
        hotkey_to_binding: &mut HashMap<K::Id, (String, String)>,
        binding_id: &str,
    ) -> Result<(), String> {
        if let Some(id) = binding_to_hotkey.remove(binding_id) {
            manager
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
    //! The shortcut thread driven through `manager_thread` with fake key
    //! sources. Real
    //! listeners need the Accessibility permission, so none is created here.

    use super::*;
    use handy_keys::{Key, Modifiers};
    use std::sync::atomic::{AtomicU32, AtomicUsize};
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

    /// The hotkey manager and raw listener, fed by the test.
    struct FakeKeys {
        next_id: AtomicU32,
        chords: Receiver<(u32, bool)>,
        raw: Mutex<Option<Receiver<KeyEvent>>>,
    }

    struct FakeRaw(Receiver<KeyEvent>);

    impl RawSource for FakeRaw {
        fn try_recv(&self) -> Option<KeyEvent> {
            self.0.try_recv().ok()
        }
    }

    impl KeySource for FakeKeys {
        type Id = u32;
        fn register(&self, _: Hotkey) -> Result<u32, String> {
            Ok(self.next_id.fetch_add(1, Ordering::SeqCst))
        }
        fn unregister(&self, _: u32) -> Result<(), String> {
            Ok(())
        }
        fn try_recv(&self) -> Option<(u32, bool)> {
            self.chords.try_recv().ok()
        }
        fn raw_listener(&self) -> Result<Box<dyn RawSource>, String> {
            let raw = self.raw.lock().unwrap().take().ok_or("taken")?;
            Ok(Box::new(FakeRaw(raw)))
        }
    }

    struct Harness {
        cmd: Option<Sender<ManagerCommand>>,
        raw: Sender<KeyEvent>,
        chords: Sender<(u32, bool)>,
        did: Receiver<(Did, Instant)>,
        wakes: Arc<AtomicUsize>,
        thread: Option<JoinHandle<()>>,
    }

    impl Harness {
        fn start(trigger: Option<Duration>) -> Self {
            let (cmd_tx, cmd_rx) = mpsc::channel();
            let (raw_tx, raw_rx) = mpsc::channel();
            let (chord_tx, chord_rx) = mpsc::channel();
            let (did_tx, did_rx) = mpsc::channel();
            let wakes = Arc::new(AtomicUsize::new(0));
            let effects = Recorder {
                did: did_tx,
                wakes: Arc::clone(&wakes),
                trigger,
            };
            let keys = FakeKeys {
                next_id: AtomicU32::new(0),
                chords: chord_rx,
                raw: Mutex::new(Some(raw_rx)),
            };
            let thread =
                thread::spawn(move || HandyKeysState::manager_thread(cmd_rx, effects, keys));
            Self {
                cmd: Some(cmd_tx),
                raw: raw_tx,
                chords: chord_tx,
                did: did_rx,
                wakes,
                thread: Some(thread),
            }
        }

        fn command(&self, cmd: ManagerCommand) {
            self.cmd.as_ref().unwrap().send(cmd).unwrap();
        }

        fn register(&self, binding_id: &str, hotkey: &str) {
            let (tx, rx) = mpsc::channel();
            self.command(ManagerCommand::Register {
                binding_id: binding_id.into(),
                hotkey_string: hotkey.into(),
                response: tx,
            });
            rx.recv_timeout(WAIT).unwrap().unwrap();
        }

        /// A key as the listener reports it.
        fn key(&self, event: KeyEvent) {
            self.raw.send(event).unwrap();
        }

        /// The hotkey manager's report of the `n`th registered chord.
        fn chord(&self, n: u32, pressed: bool) {
            self.chords.send((n, pressed)).unwrap();
        }

        fn next(&self) -> (Did, Instant) {
            self.did.recv_timeout(WAIT).expect("nothing dispatched")
        }

        fn assert_quiet(&self) {
            if let Ok(did) = self.did.recv_timeout(Duration::from_millis(50)) {
                panic!("unexpected {did:?}");
            }
        }

        /// Whether the thread ends within `WAIT`.
        fn stops(&mut self) -> bool {
            let thread = self.thread.take().unwrap();
            let t0 = Instant::now();
            while !thread.is_finished() && t0.elapsed() < WAIT {
                thread::sleep(Duration::from_millis(1));
            }
            thread.is_finished()
        }
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
        // Polls every 10 ms.
        assert!(wakes > 10, "{wakes}");
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
        assert!(percentile(&dispatch, 50) < Duration::from_millis(100));
    }

    #[test]
    fn raw_keys_reach_the_trigger_before_the_chord() {
        let h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        // Option, Space and the manager's report of Option+Space, all queued
        // before the thread looks.
        h.key(mods(OPT, true));
        h.key(key(OPT, Key::Space, true));
        h.chord(0, true);
        assert!(matches!(
            h.next().0,
            Did::Trigger(TriggerAction::Press { .. })
        ));
        assert!(matches!(
            h.next().0,
            Did::Trigger(TriggerAction::Rebase { .. })
        ));
        // The manager's report of the Space the raw stream handed over is
        // absorbed; the chord's release passes on.
        h.assert_quiet();
        h.key(key(OPT, Key::Space, false));
        h.chord(0, false);
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
        h.chord(1, true);
        assert_eq!(h.next().0, Did::Shortcut("cancel", true));
        h.key(key(Modifiers::empty(), Key::Escape, false));
        h.chord(1, false);
        assert_eq!(h.next().0, Did::Shortcut("cancel", false));
        h.key(mods(OPT, true));
        h.key(key(OPT, Key::Space, true));
        h.chord(0, true);
        assert_eq!(h.next().0, Did::Shortcut("transcribe", true));
        h.assert_quiet();
    }

    #[test]
    fn commands_are_answered_while_keys_are_idle() {
        let h = Harness::start(Some(THRESHOLD));
        for n in 0..20 {
            let t0 = Instant::now();
            h.register(&format!("binding{n}"), &format!("ctrl+f{}", n % 12 + 1));
            assert!(t0.elapsed() < Duration::from_millis(500));
        }
        let (tx, rx) = mpsc::channel();
        h.command(ManagerCommand::Unregister {
            binding_id: "binding0".into(),
            response: tx,
        });
        assert_eq!(rx.recv_timeout(WAIT).unwrap(), Ok(()));
    }

    #[test]
    fn stops_on_shutdown() {
        let mut h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        h.command(ManagerCommand::Shutdown);
        assert!(h.stops());
    }

    #[test]
    fn stops_when_the_command_sender_is_dropped() {
        let mut h = Harness::start(Some(THRESHOLD));
        h.register(TRANSCRIBE, "option+space");
        h.cmd = None;
        assert!(h.stops());
    }
}
