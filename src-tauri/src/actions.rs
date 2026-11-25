#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::VadPolicy;
use crate::managers::audio::{recording_notice_code, AudioRecordingManager, RecordingReadiness};
use crate::managers::history::HistoryManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::policy::{live_decode, LiveDecode, ModelTraits};
use crate::managers::transcription::StreamWorkKind;
use crate::managers::transcription::{Transcription, TranscriptionManager};
use crate::notices::{self, Notice, NoticeCode, NoticeSource};
use crate::settings::{get_settings, AppSettings, OverlayStyle, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{set_tray_state, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_specta::Event;

/// How a dictation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptionOutcome {
    /// The text was typed.
    Typed,
    /// Nothing was heard, so nothing was typed.
    Empty,
    /// The microphone sent only digital silence (e.g. muted).
    Silent,
    /// Transcribing or pasting failed.
    Failed,
}

/// Emitted when a dictation finishes (not on cancel); onboarding uses it to measure speed.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct TranscriptionCompleteEvent {
    pub outcome: TranscriptionOutcome,
    pub audio_seconds: f64,
    pub wait_seconds: f64,
}

fn emit_transcription_complete(
    app: &AppHandle,
    outcome: TranscriptionOutcome,
    sample_count: usize,
    stop_time: Instant,
) {
    let event = TranscriptionCompleteEvent {
        outcome,
        audio_seconds: sample_count as f64
            / crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE as f64,
        wait_seconds: stop_time.elapsed().as_secs_f64(),
    };
    if let Err(e) = event.emit(app) {
        warn!("Failed to emit transcription complete: {e}");
    }
}

/// Nothing to type: `Silent` if the microphone sent only zeros, else `Empty`.
fn empty_outcome(input_silent: bool) -> TranscriptionOutcome {
    if input_silent {
        TranscriptionOutcome::Silent
    } else {
        TranscriptionOutcome::Empty
    }
}

/// Ends a dictation that has nothing to type. A microphone that sent only
/// digital silence (muted, or the wrong default device) gets a notice, so it
/// doesn't look like the model missed the speech (#1899).
fn finish_without_text(
    app: &AppHandle,
    input_silent: bool,
    sample_count: usize,
    stop_time: Instant,
) {
    if input_silent {
        warn!("The microphone sent only digital silence; nothing was transcribed");
        notices::raise(
            app,
            Notice::new(NoticeCode::MicrophoneSilent, NoticeSource::Dictation),
        );
    } else {
        utils::hide_recording_overlay(app);
    }
    set_tray_state(app, TrayIconState::Idle);
    emit_transcription_complete(app, empty_outcome(input_silent), sample_count, stop_time);
}

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Deadline for the whole post-processing step, retries included. The HTTP
/// client only bounds connecting, so an endpoint that accepts the request and
/// never answers would otherwise hold the dictation until a proxy gives up.
const POST_PROCESS_TIMEOUT: Duration = Duration::from_secs(60);

/// Bumped per recording session, so a late paste from an earlier session
/// doesn't reset the tray or hide the overlay of a newer one.
static RECORDING_SESSION: AtomicU64 = AtomicU64::new(0);

/// Drop guard that finishes the transcription pipeline, including immediate
/// model unloading on early exits.
struct FinishGuard(AppHandle, Arc<TranscriptionManager>);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.1.maybe_unload_immediately("transcription session");
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
        // The pipeline just freed its large transient buffers (captured PCM,
        // WAV copy, engine scratch); hand the cached pages back to the OS so
        // they don't sit in malloc arenas until they get swapped out (#1792).
        crate::memory::trim_freed_memory();
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Strip a leading `<think>...</think>` block. Some endpoints can't disable
/// reasoning, and some local servers put the reasoning text into `content`
/// instead of a separate field — without this the user would get the model's
/// chain of thought pasted along with the cleaned transcription.
fn strip_think_block(s: &str) -> &str {
    if let Some(rest) = s.trim_start().strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            return rest[end + "</think>".len()..].trim_start();
        }
    }
    s
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

/// Returns `true` when a transcription has no meaningful content to
/// post-process (empty or whitespace-only). Used to skip the post-processing
/// LLM call when nothing was actually transcribed, which would otherwise make
/// the model reply with an error message such as "you need to provide the
/// transcription".
fn is_blank_transcription(transcription: &str) -> bool {
    transcription.trim().is_empty()
}

fn non_blank_post_processed_text(text: String) -> Option<String> {
    if is_blank_transcription(&text) {
        warn!(
            "Post-processing returned an empty response. Falling back to original transcription."
        );
        None
    } else {
        Some(text)
    }
}

async fn complete_unless_cancelled<F, C>(operation: F, is_cancelled: C) -> Option<F::Output>
where
    F: Future,
    C: Fn() -> bool,
{
    tokio::pin!(operation);

    loop {
        if is_cancelled() {
            return None;
        }

        if let Ok(result) =
            tokio::time::timeout(CANCELLATION_POLL_INTERVAL, operation.as_mut()).await
        {
            return Some(result);
        }
    }
}

fn should_use_streaming_overlay(style: OverlayStyle, is_streaming: bool) -> bool {
    style == OverlayStyle::Live && is_streaming
}

/// What became of a transcription's post-processing.
#[derive(Debug, PartialEq, Eq)]
enum PostProcessOutcome {
    /// There was no speech to process.
    Skipped,
    Done(String),
    /// The user asked for it, but it couldn't run or produced nothing, so the
    /// transcript goes out as it is.
    Failed,
}

async fn post_process_transcription(
    settings: &AppSettings,
    transcription: &str,
) -> PostProcessOutcome {
    if is_blank_transcription(transcription) {
        debug!("Post-processing skipped because the transcription is empty");
        return PostProcessOutcome::Skipped;
    }

    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return PostProcessOutcome::Failed;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return PostProcessOutcome::Failed;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return PostProcessOutcome::Failed;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return PostProcessOutcome::Failed;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return PostProcessOutcome::Failed;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Ask these providers to skip reasoning/thinking — post-processing rarely
    // benefits from it and it adds seconds of latency. llm_client picks the
    // field the endpoint understands and retries without it if rejected.
    let disable_reasoning = matches!(provider.id.as_str(), "custom" | "openrouter");

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(&prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return PostProcessOutcome::Failed;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            PostProcessOutcome::Failed
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            PostProcessOutcome::Done(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        PostProcessOutcome::Failed
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return PostProcessOutcome::Failed;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            disable_reasoning,
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                let content = strip_think_block(&content);
                match serde_json::from_str::<serde_json::Value>(content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return PostProcessOutcome::Done(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return PostProcessOutcome::Done(strip_invisible_chars(content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return PostProcessOutcome::Done(strip_invisible_chars(content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return PostProcessOutcome::Failed;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        disable_reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(strip_think_block(&content));
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            PostProcessOutcome::Done(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            PostProcessOutcome::Failed
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            PostProcessOutcome::Failed
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    /// `final_text` is the bare transcript because post-processing failed.
    pub post_process_failed: bool,
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;
    let mut post_process_failed = false;

    if post_process {
        // A reply that's blank once cleaned is no better than none.
        let outcome = match tokio::time::timeout(
            POST_PROCESS_TIMEOUT,
            post_process_transcription(&settings, &final_text),
        )
        .await
        {
            Ok(PostProcessOutcome::Done(text)) => non_blank_post_processed_text(text)
                .map_or(PostProcessOutcome::Failed, PostProcessOutcome::Done),
            Ok(outcome) => outcome,
            Err(_) => {
                warn!(
                    "Post-processing timed out after {:?}; pasting the original transcription",
                    POST_PROCESS_TIMEOUT
                );
                PostProcessOutcome::Failed
            }
        };
        match outcome {
            PostProcessOutcome::Skipped => {}
            PostProcessOutcome::Failed => post_process_failed = true,
            PostProcessOutcome::Done(processed_text) => {
                post_processed_text = Some(processed_text.clone());
                final_text = processed_text;

                if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                    if let Some(prompt) = settings
                        .post_process_prompts
                        .iter()
                        .find(|prompt| &prompt.id == prompt_id)
                    {
                        post_process_prompt = Some(prompt.prompt.clone());
                    }
                }
            }
        }
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
        post_process_failed,
    }
}

/// What starting a dictation needs from the app, one method per step, so the
/// order of the steps can be tested without an `AppHandle`.
pub(crate) trait StartSteps: Clone + Send + 'static {
    /// Whether a session can start. Raises its own notice when it can't.
    fn can_start(&self) -> bool;
    /// Whether the microphone is already in a session, which refuses the
    /// start.
    fn is_recording(&self) -> bool;
    /// Tray, live-stream plan and overlay: everything shown before the mic
    /// opens. Returns how the session's audio is filtered.
    fn prepare_session(&self) -> VadPolicy;
    /// Starts capture. Blocks while an on-demand microphone opens.
    fn start_recording(&self, vad_policy: VadPolicy) -> anyhow::Result<RecordingReadiness>;
    /// Whether `generation` is still the live capture (not stopped or
    /// cancelled since).
    fn is_current(&self, generation: u64) -> bool;
    /// Tells the overlay the microphone is delivering audio.
    fn emit_ready(&self);
    /// Plays the start sound to its end; returns at once when feedback is off.
    fn play_start_sound(&self);
    /// Mutes system audio for the capture `generation`, unless it has been
    /// stopped or cancelled since.
    fn apply_mute(&self, generation: u64);
}

pub(crate) enum StartOutcome {
    /// Nothing started; the refusal already told the user why.
    Refused,
    /// The microphone couldn't start.
    Failed(anyhow::Error),
    /// Capture started; the worker waits for the first audio.
    Started(std::thread::JoinHandle<()>),
}

/// The start sequence of a dictation, in order.
pub(crate) fn run_start<S: StartSteps>(steps: S) -> StartOutcome {
    if !steps.can_start() {
        return StartOutcome::Refused;
    }
    // The start sound says the press was taken, so it plays now, while the
    // microphone may still be opening; the overlay turning to recording is
    // the mic-ready cue. Its own thread opens the feedback output if needed.
    // A microphone already in a session refuses the start below, so that
    // press gets no sound.
    let start_sound = (!steps.is_recording())
        .then(|| {
            let steps = steps.clone();
            std::thread::Builder::new()
                .name("start-sound".into())
                .spawn(move || steps.play_start_sound())
                .map_err(|e| warn!("Failed to spawn the start sound thread: {e}"))
                .ok()
        })
        .flatten();
    let vad_policy = steps.prepare_session();

    let recording_start_time = Instant::now();
    let readiness = match steps.start_recording(vad_policy) {
        Ok(readiness) => readiness,
        Err(e) => {
            debug!("Failed to start recording: {}", e);
            return StartOutcome::Failed(e);
        }
    };
    debug!(
        "Recording request accepted in {:?}; waiting for first microphone samples",
        recording_start_time.elapsed()
    );
    let generation = readiness.generation();
    StartOutcome::Started(std::thread::spawn(move || {
        if !readiness.wait() {
            debug!("Microphone readiness wait ended without receiving samples");
            return;
        }

        // Development-only preview hook for evaluating the brief
        // arming animation on hardware that normally starts too fast
        // to make it visible.
        #[cfg(debug_assertions)]
        if let Ok(delay_ms) = std::env::var("HANDY_DEBUG_MIC_READY_DELAY_MS")
            .unwrap_or_default()
            .parse::<u64>()
        {
            let delay_ms = delay_ms.min(10_000);
            if delay_ms > 0 {
                debug!("Delaying microphone-ready cue by {delay_ms}ms for UI preview");
                std::thread::sleep(Duration::from_millis(delay_ms));
            }
        }

        if !steps.is_current(generation) {
            debug!("Microphone became ready for an inactive recording");
            return;
        }

        debug!("Microphone is receiving samples; recording is ready");
        steps.emit_ready();

        // Mute no earlier than the end of the start sound (0.3-0.6 s, often
        // longer than the microphone takes to open), so its tail isn't cut
        // off. Ready above doesn't wait for it.
        if let Some(start_sound) = start_sound {
            let _ = start_sound.join();
        }
        steps.apply_mute(generation);
    }))
}

/// [`StartSteps`] on the running app.
#[derive(Clone)]
struct AppStartSteps {
    app: AppHandle,
    binding_id: String,
    selected_model: String,
}

impl AppStartSteps {
    /// Starts loading the model and the VAD in the background. Returns
    /// whether a model load started here, which reports its own failure.
    fn kick_off_loads(&self) -> bool {
        let started = Instant::now();
        // An empty selection has nothing to load; `can_start` says so.
        let load_started = !self.selected_model.is_empty()
            && self
                .app
                .state::<Arc<TranscriptionManager>>()
                .initiate_model_load(true);
        let rm = Arc::clone(&self.app.state::<Arc<AudioRecordingManager>>());
        std::thread::spawn(move || {
            if let Err(e) = rm.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });
        debug!("start-path model_kickoff={:?}", started.elapsed());
        load_started
    }
}

impl StartSteps for AppStartSteps {
    fn can_start(&self) -> bool {
        let tm = self.app.state::<Arc<TranscriptionManager>>();
        // Don't open the mic if nothing can transcribe the recording. A load
        // started by this start reports its own failure; one already running
        // doesn't. An accepted start kicks off its loads after the start
        // sound, in `prepare_session`.
        if tm.is_model_loaded() {
            return true;
        }
        if self.selected_model.is_empty() {
            self.kick_off_loads();
            warn!("Not starting recording: no model is selected");
            tm.raise_no_model_selected();
            return false;
        }
        if let Err(e) = self
            .app
            .state::<Arc<ModelManager>>()
            .get_model_path(&self.selected_model)
        {
            let load_started = self.kick_off_loads();
            warn!("Not starting recording: no model can transcribe it ({})", e);
            if !load_started {
                tm.raise_model_load_failed(&self.selected_model, &e);
            }
            return false;
        }
        true
    }

    fn is_recording(&self) -> bool {
        self.app
            .state::<Arc<AudioRecordingManager>>()
            .is_recording()
    }

    fn prepare_session(&self) -> VadPolicy {
        self.kick_off_loads();
        let app = &self.app;
        let tm = app.state::<Arc<TranscriptionManager>>();
        RECORDING_SESSION.fetch_add(1, Ordering::SeqCst);
        let tray_started = Instant::now();
        set_tray_state(app, TrayIconState::Recording);
        let tray_elapsed = tray_started.elapsed();

        let plan_started = Instant::now();
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;

        let selected_model_info = app
            .state::<Arc<ModelManager>>()
            .get_model_info(&settings.selected_model);

        // Use the app-facing model capability as the single pre-recording source
        // for live streaming decisions. Unknown support is represented as false
        // until the model registry is updated by discovery or runtime load.
        let model_supports_streaming = selected_model_info
            .as_ref()
            .map(|m| m.supports_streaming)
            .unwrap_or(false);
        let vad_policy = if !settings.vad_enabled {
            VadPolicy::Disabled
        } else if model_supports_streaming {
            VadPolicy::Streaming
        } else {
            VadPolicy::Offline
        };
        // Before the model loads, whether it keeps its accuracy on long audio
        // is unknown, so assume not: the worker decides again with the loaded
        // model, and a wrong guess costs only an idle worker.
        let expected_model = ModelTraits {
            streaming: model_supports_streaming,
            long_form: false,
        };
        if live_decode(expected_model, settings.long_form_chunking) != LiveDecode::None {
            tm.start_stream();
        }
        let plan_elapsed = plan_started.elapsed();

        // Sizing the overlay follows the same advertised capability. A model that
        // doesn't stream (or whose capability is not known yet) gets the compact
        // pill instead of an oversized transparent live window.
        let overlay_started = Instant::now();
        match settings.overlay_style {
            OverlayStyle::Live if model_supports_streaming => utils::show_streaming_overlay(app),
            OverlayStyle::Live | OverlayStyle::Minimal => show_recording_overlay(app),
            OverlayStyle::None => {} // show_overlay_state no-ops on None anyway
        }
        // Everything above runs before capture can begin, so each span here is
        // added keypress->capture latency.
        debug!(
            "start-path pre-recording steps: tray={:?} settings+stream_plan={:?} overlay={:?}",
            tray_elapsed,
            plan_elapsed,
            overlay_started.elapsed()
        );
        debug!("Microphone mode - always_on: {}", is_always_on);
        vad_policy
    }

    fn start_recording(&self, vad_policy: VadPolicy) -> anyhow::Result<RecordingReadiness> {
        self.app
            .state::<Arc<AudioRecordingManager>>()
            .try_start_recording(&self.binding_id, vad_policy)
    }

    fn is_current(&self, generation: u64) -> bool {
        self.app
            .state::<Arc<AudioRecordingManager>>()
            .is_recording_readiness_current(generation)
    }

    fn emit_ready(&self) {
        utils::emit_recording_ready(&self.app);
    }

    fn play_start_sound(&self) {
        play_feedback_sound_blocking(&self.app, SoundType::Start);
    }

    fn apply_mute(&self, generation: u64) {
        self.app
            .state::<Arc<AudioRecordingManager>>()
            .apply_mute(generation);
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        let tm = app.state::<Arc<TranscriptionManager>>();
        let steps = AppStartSteps {
            app: app.clone(),
            binding_id: binding_id.to_string(),
            selected_model: get_settings(app).selected_model,
        };
        match run_start(steps) {
            StartOutcome::Refused => return,
            StartOutcome::Started(_worker) => {
                // Dynamically register the cancel shortcut in a separate task to avoid deadlock
                shortcut::register_cancel_shortcut(app);
            }
            StartOutcome::Failed(err) => {
                // Starting failed (for example due to blocked microphone permissions).
                // Revert UI state so we don't stay stuck in the recording overlay.
                tm.cancel_stream();
                set_tray_state(app, TrayIconState::Idle);
                notices::raise(
                    app,
                    Notice::new(recording_notice_code(&err), NoticeSource::Dictation)
                        .with_detail(format!("{err:#}")),
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Prevent a slow microphone from emitting a ready event or muting
        // after the user has already requested stop.
        app.state::<Arc<AudioRecordingManager>>()
            .invalidate_recording_readiness();

        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        set_tray_state(app, TrayIconState::Transcribing);
        // Stop should give immediate visual feedback. Live streaming can keep
        // the larger panel, but it still switches from listening to a working
        // spinner while the stream finalizes. Non-streaming paths use the
        // compact transcribing pill (None no-ops in show_*).
        let style = get_settings(app).overlay_style;
        // Capture this before finalizing the stream so every later working state
        // targets the same overlay that was shown for this transcription.
        let use_streaming_overlay = should_use_streaming_overlay(style, tm.is_streaming());
        if use_streaming_overlay {
            tm.emit_stream_working(StreamWorkKind::Transcribing);
        } else {
            show_transcribing_overlay(app);
        }

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;
        let cancel_generation = rm.cancel_generation();

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone(), Arc::clone(&tm));
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) {
                // Read now: by the time transcribing ends, another take may
                // have replaced it.
                let input_silent = rm.last_input_silent();
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if rm.was_cancelled_since(cancel_generation) {
                    debug!("Transcription operation cancelled after recording stop");
                    tm.cancel_stream();
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                    return;
                }

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    // Tear down any streaming worker so its channel doesn't leak
                    // and block the next start_stream.
                    tm.cancel_stream();
                    finish_without_text(&ah, input_silent, 0, stop_time);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save: the live decode's
                    // result when it has one (all audio was already fed to it),
                    // otherwise a batch run of the samples. A cancel is surfaced
                    // as an error, so it never starts a batch run.
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe_recording(samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    if rm.was_cancelled_since(cancel_generation) {
                        debug!("Transcription operation cancelled before output handling");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    }

                    match transcription_result {
                        Ok(Transcription {
                            text: transcription,
                            language,
                        }) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                utils::redact_text(&transcription)
                            );

                            if post_process {
                                if use_streaming_overlay {
                                    tm.emit_stream_working(StreamWorkKind::Polishing);
                                } else {
                                    show_processing_overlay(&ah);
                                }
                            }
                            let Some(processed) = complete_unless_cancelled(
                                process_transcription_output(&ah, &transcription, post_process),
                                || rm.was_cancelled_since(cancel_generation),
                            )
                            .await
                            else {
                                debug!("Transcription operation cancelled during output handling");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            };

                            if rm.was_cancelled_since(cancel_generation) {
                                debug!("Transcription operation cancelled before paste");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                    language,
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            // The user's hook only changes what gets pasted;
                            // history above keeps the text as produced.
                            let Some(final_text) = complete_unless_cancelled(
                                crate::transcription_hook::apply(&ah, processed.final_text),
                                || rm.was_cancelled_since(cancel_generation),
                            )
                            .await
                            else {
                                debug!("Transcription operation cancelled during hook");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            };

                            if final_text.is_empty() {
                                finish_without_text(&ah, input_silent, sample_count, stop_time);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let post_process_failed = processed.post_process_failed;
                                let rm_for_paste = Arc::clone(&rm);
                                // Taken before the paste thread exists, and so
                                // before the next session can start.
                                let session = RECORDING_SESSION.load(Ordering::SeqCst);
                                // Off the main thread: paste sleeps between steps, and a
                                // paste into Handy's own windows needs the main thread free.
                                std::thread::Builder::new()
                                    .name("paste".into())
                                    .spawn(move || {
                                        if rm_for_paste.was_cancelled_since(cancel_generation) {
                                            debug!(
                                                "Transcription operation cancelled before paste"
                                            );
                                            utils::hide_recording_overlay(&ah_clone);
                                            set_tray_state(&ah_clone, TrayIconState::Idle);
                                            return;
                                        }

                                        // A new session may have taken over the
                                        // UI while this paste ran.
                                        let owns_ui =
                                            || RECORDING_SESSION.load(Ordering::SeqCst) == session;
                                        let (outcome, notice) =
                                            match utils::paste(final_text, ah_clone.clone()) {
                                                Ok(utils::PasteOutcome::Pasted) => {
                                                    debug!(
                                                        "Text pasted successfully in {:?}",
                                                        paste_time.elapsed()
                                                    );
                                                    // Leave a newer session's notice alone.
                                                    if owns_ui() {
                                                        notices::clear(
                                                            &ah_clone,
                                                            NoticeSource::Dictation,
                                                        );
                                                    }
                                                    let notice = post_process_failed.then(|| {
                                                        Notice::new(
                                                            NoticeCode::PostProcessFailed,
                                                            NoticeSource::Dictation,
                                                        )
                                                    });
                                                    (TranscriptionOutcome::Typed, notice)
                                                }
                                                Ok(utils::PasteOutcome::Blocked) => (
                                                    TranscriptionOutcome::Failed,
                                                    Some(Notice::new(
                                                        NoticeCode::PasteBlockedElevated,
                                                        NoticeSource::Dictation,
                                                    )),
                                                ),
                                                Err(e) => {
                                                    error!("Failed to paste transcription: {}", e);
                                                    (
                                                        TranscriptionOutcome::Failed,
                                                        Some(Notice::new(
                                                            NoticeCode::PasteFailed,
                                                            NoticeSource::Dictation,
                                                        )),
                                                    )
                                                }
                                            };
                                        match (owns_ui(), notice) {
                                            (true, Some(notice)) => {
                                                notices::raise(&ah_clone, notice)
                                            }
                                            (true, None) => {
                                                utils::hide_recording_overlay(&ah_clone)
                                            }
                                            (false, Some(notice)) => {
                                                notices::raise_without_overlay(&ah_clone, notice)
                                            }
                                            (false, None) => {}
                                        }
                                        if owns_ui() {
                                            set_tray_state(&ah_clone, TrayIconState::Idle);
                                        }
                                        emit_transcription_complete(
                                            &ah_clone,
                                            outcome,
                                            sample_count,
                                            stop_time,
                                        );
                                    })
                                    .map(|_| ())
                                    .unwrap_or_else(|e| {
                                        error!("Failed to start the paste thread: {:?}", e);
                                        utils::hide_recording_overlay(&ah);
                                        set_tray_state(&ah, TrayIconState::Idle);
                                        emit_transcription_complete(
                                            &ah,
                                            TranscriptionOutcome::Failed,
                                            sample_count,
                                            stop_time,
                                        );
                                    });
                            }
                        }
                        Err(err) => {
                            if rm.was_cancelled_since(cancel_generation) {
                                debug!(
                                    "Transcription operation cancelled after transcription error"
                                );
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            error!("Transcription failed: {}", err);
                            notices::raise(
                                &ah,
                                Notice::new(
                                    NoticeCode::TranscriptionFailed,
                                    NoticeSource::Dictation,
                                )
                                .with_detail(err.to_string()),
                            );
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            set_tray_state(&ah, TrayIconState::Idle);
                            emit_transcription_complete(
                                &ah,
                                TranscriptionOutcome::Failed,
                                sample_count,
                                stop_time,
                            );
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                // Tear down any streaming worker so its channel doesn't leak.
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                if !rm.was_cancelled_since(cancel_generation) {
                    emit_transcription_complete(&ah, TranscriptionOutcome::Empty, 0, stop_time);
                }
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::{
        complete_unless_cancelled, is_blank_transcription, non_blank_post_processed_text,
        post_process_transcription, should_use_streaming_overlay, strip_invisible_chars,
        strip_think_block, PostProcessOutcome,
    };
    use crate::settings::{get_default_settings, AppSettings, OverlayStyle};
    use std::future;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// Settings that run unstructured post-processing against `base_url`.
    fn post_process_settings(base_url: &str) -> AppSettings {
        let mut settings = get_default_settings();
        let provider = settings
            .post_process_providers
            .iter_mut()
            .find(|provider| provider.id == settings.post_process_provider_id)
            .expect("the default provider exists");
        provider.base_url = base_url.to_string();
        provider.supports_structured_output = false;
        let provider_id = provider.id.clone();
        settings
            .post_process_models
            .insert(provider_id, "model".to_string());
        settings.post_process_selected_prompt_id =
            settings.post_process_prompts.first().map(|p| p.id.clone());
        settings
    }

    /// Answers one chat completion request with `body`, and returns its URL.
    fn serve_once(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 8192];
            let _ = stream.read(&mut request);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        format!("http://{address}")
    }

    fn run_post_process(settings: &AppSettings, transcription: &str) -> PostProcessOutcome {
        tauri::async_runtime::block_on(post_process_transcription(settings, transcription))
    }

    #[test]
    fn blank_speech_is_skipped_but_missing_setup_failed() {
        let settings = post_process_settings("http://127.0.0.1:1");
        assert_eq!(
            run_post_process(&settings, "  "),
            PostProcessOutcome::Skipped
        );

        let mut settings = post_process_settings("http://127.0.0.1:1");
        settings.post_process_models.clear();
        assert_eq!(
            run_post_process(&settings, "hello"),
            PostProcessOutcome::Failed
        );
    }

    #[test]
    fn a_post_processing_reply_is_done() {
        let url = serve_once(r#"{"choices":[{"message":{"content":"polished"}}]}"#);
        assert_eq!(
            run_post_process(&post_process_settings(&url), "hello"),
            PostProcessOutcome::Done("polished".to_string())
        );
    }

    #[test]
    fn a_post_processing_reply_with_no_content_failed() {
        let url = serve_once(r#"{"choices":[]}"#);
        assert_eq!(
            run_post_process(&post_process_settings(&url), "hello"),
            PostProcessOutcome::Failed
        );
    }

    #[test]
    fn blank_transcription_is_detected() {
        assert!(is_blank_transcription(""));
        assert!(is_blank_transcription("   "));
        assert!(is_blank_transcription("\t\n  \r\n"));
    }

    #[test]
    fn non_blank_transcription_is_kept() {
        assert!(!is_blank_transcription("hello"));
        assert!(!is_blank_transcription("  hello  "));
    }

    #[test]
    fn blank_post_processed_text_is_rejected() {
        for text in ["", "   ", "\t\n\r\n", "\u{00A0}\u{2003}"] {
            assert_eq!(non_blank_post_processed_text(text.to_string()), None);
        }
    }

    #[test]
    fn cleaned_empty_post_processed_text_is_rejected() {
        for text in [
            "\u{200B}\u{200C}\u{200D}\u{FEFF}",
            "<think>reasoning only</think>",
            "<think>reasoning only</think>\n\u{200B} ",
        ] {
            let text = strip_invisible_chars(strip_think_block(text));
            assert_eq!(non_blank_post_processed_text(text), None);
        }
    }

    #[test]
    fn non_blank_post_processed_text_is_preserved() {
        for text in ["Cleaned text.", "  정리된 문장입니다.\n", "0"] {
            assert_eq!(
                non_blank_post_processed_text(text.to_string()),
                Some(text.to_string())
            );
        }
    }

    #[test]
    fn completed_operation_returns_its_output() {
        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::ready("done"),
            || false,
        ));

        assert_eq!(result, Some("done"));
    }

    #[test]
    fn pending_operation_stops_after_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_for_thread = Arc::clone(&cancelled);
        let cancel_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            cancelled_for_thread.store(true, Ordering::Release);
        });

        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::pending::<()>(),
            || cancelled.load(Ordering::Acquire),
        ));

        cancel_thread.join().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn leading_think_block_is_stripped() {
        assert_eq!(
            strip_think_block("<think>pondering...</think>Cleaned text."),
            "Cleaned text."
        );
        assert_eq!(
            strip_think_block("  \n<think>multi\nline</think>\n  Cleaned text."),
            "Cleaned text."
        );
    }

    #[test]
    fn content_without_think_block_is_unchanged() {
        assert_eq!(strip_think_block("Cleaned text."), "Cleaned text.");
        assert_eq!(
            strip_think_block("Mentions <think> mid-sentence."),
            "Mentions <think> mid-sentence."
        );
        // Unclosed block: leave untouched rather than guess
        assert_eq!(
            strip_think_block("<think>never closed"),
            "<think>never closed"
        );
    }

    #[test]
    fn live_overlay_uses_streaming_states_only_for_streaming_models() {
        assert!(should_use_streaming_overlay(OverlayStyle::Live, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::Live, false));
        assert!(!should_use_streaming_overlay(OverlayStyle::Minimal, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::None, true));
    }
}

#[cfg(test)]
mod start_order_tests {
    use super::{run_start, StartOutcome, StartSteps};
    use crate::audio_toolkit::VadPolicy;
    use crate::managers::audio::RecordingReadiness;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// How long a step that should happen is waited for.
    const WAIT: Duration = Duration::from_secs(2);
    /// How long a step that shouldn't happen yet is watched for.
    const WATCH: Duration = Duration::from_millis(100);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Step {
        MicOpened,
        Ready,
        ToneRequested,
        ToneEnded,
        Mute,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mic {
        Works,
        /// Already in a session, so the start is refused.
        Busy,
        /// Fails to open.
        Fails,
    }

    /// Stands in for the app: a microphone whose open waits on a gate, and
    /// whose first audio comes when the test sends it.
    struct Fake {
        can_start: bool,
        mic: Mic,
        /// Set once the session is stopped or cancelled.
        stopped: AtomicBool,
        mic_gate: Mutex<Option<Receiver<()>>>,
        first_audio: Mutex<Option<Receiver<()>>>,
        tone_gate: Mutex<Option<Receiver<()>>>,
        log: Mutex<Vec<Step>>,
        events: Mutex<Sender<Step>>,
    }

    #[derive(Clone)]
    struct FakeSteps(Arc<Fake>);

    /// The test's side of a [`Fake`]. Dropping a sender releases its gate.
    struct Controls {
        mic_gate: Sender<()>,
        first_audio: Sender<()>,
        tone_gate: Sender<()>,
        events: Receiver<Step>,
    }

    impl Controls {
        fn wait_for(&self, step: Step, within: Duration) -> bool {
            let deadline = Instant::now() + within;
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                match self.events.recv_timeout(left) {
                    Ok(s) if s == step => return true,
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                        return false
                    }
                }
            }
        }
    }

    fn fake(can_start: bool, mic: Mic) -> (FakeSteps, Controls) {
        let (mic_tx, mic_rx) = mpsc::channel();
        let (audio_tx, audio_rx) = mpsc::channel();
        let (tone_tx, tone_rx) = mpsc::channel();
        let (events_tx, events_rx) = mpsc::channel();
        let steps = FakeSteps(Arc::new(Fake {
            can_start,
            mic,
            stopped: AtomicBool::new(false),
            mic_gate: Mutex::new(Some(mic_rx)),
            first_audio: Mutex::new(Some(audio_rx)),
            tone_gate: Mutex::new(Some(tone_rx)),
            log: Mutex::new(Vec::new()),
            events: Mutex::new(events_tx),
        }));
        let controls = Controls {
            mic_gate: mic_tx,
            first_audio: audio_tx,
            tone_gate: tone_tx,
            events: events_rx,
        };
        (steps, controls)
    }

    impl FakeSteps {
        fn record(&self, step: Step) {
            self.0.log.lock().unwrap().push(step);
            let _ = self.0.events.lock().unwrap().send(step);
        }

        fn steps(&self) -> Vec<Step> {
            self.0.log.lock().unwrap().clone()
        }

        fn count(&self, step: Step) -> usize {
            self.steps().iter().filter(|s| **s == step).count()
        }

        /// Where `step` first comes in the order things happened.
        fn at(&self, step: Step) -> usize {
            self.steps()
                .iter()
                .position(|s| *s == step)
                .unwrap_or_else(|| panic!("{step:?} never happened"))
        }
    }

    /// Waits on the gate until the test sends on it or drops it.
    fn pass(gate: &Mutex<Option<Receiver<()>>>) {
        if let Some(gate) = gate.lock().unwrap().take() {
            let _ = gate.recv();
        }
    }

    impl StartSteps for FakeSteps {
        fn can_start(&self) -> bool {
            self.0.can_start
        }

        fn is_recording(&self) -> bool {
            self.0.mic == Mic::Busy
        }

        fn prepare_session(&self) -> VadPolicy {
            // Gated too, so tray and overlay can't come before the sound.
            pass(&self.0.mic_gate);
            VadPolicy::Disabled
        }

        fn start_recording(&self, _: VadPolicy) -> anyhow::Result<RecordingReadiness> {
            pass(&self.0.mic_gate);
            match self.0.mic {
                Mic::Busy => anyhow::bail!("Already recording"),
                Mic::Fails => anyhow::bail!("Failed to open microphone"),
                Mic::Works => {}
            }
            self.record(Step::MicOpened);
            let first_audio = self.0.first_audio.lock().unwrap().take();
            Ok(RecordingReadiness::for_test(
                first_audio.expect("one start per fake"),
                1,
            ))
        }

        fn is_current(&self, generation: u64) -> bool {
            generation == 1 && !self.0.stopped.load(Ordering::SeqCst)
        }

        fn emit_ready(&self) {
            self.record(Step::Ready);
        }

        fn play_start_sound(&self) {
            self.record(Step::ToneRequested);
            pass(&self.0.tone_gate);
            self.record(Step::ToneEnded);
        }

        fn apply_mute(&self, generation: u64) {
            if self.is_current(generation) {
                self.record(Step::Mute);
            }
        }
    }

    fn started(outcome: StartOutcome) -> thread::JoinHandle<()> {
        match outcome {
            StartOutcome::Started(worker) => worker,
            StartOutcome::Refused => panic!("start was refused"),
            StartOutcome::Failed(e) => panic!("start failed: {e}"),
        }
    }

    /// The start sound is a key-down cue: it's requested while the
    /// microphone is still opening, but the overlay's ready event still waits
    /// for the microphone's first audio. The sound plays once.
    #[test]
    fn start_sound_plays_before_the_microphone_opens() {
        let (steps, controls) = fake(true, Mic::Works);
        let starter = {
            let steps = steps.clone();
            thread::spawn(move || run_start(steps))
        };

        assert!(
            controls.wait_for(Step::ToneRequested, WAIT),
            "the start sound waited for the microphone to open"
        );
        assert_eq!(steps.count(Step::MicOpened), 0);

        controls.mic_gate.send(()).unwrap();
        let worker = started(starter.join().unwrap());
        assert!(
            !controls.wait_for(Step::Ready, WATCH),
            "ready came before the microphone delivered audio"
        );

        controls.first_audio.send(()).unwrap();
        controls.tone_gate.send(()).unwrap();
        worker.join().unwrap();
        assert!(steps.at(Step::MicOpened) <= steps.at(Step::Ready));
        assert_eq!(steps.count(Step::Ready), 1);
        assert_eq!(steps.count(Step::ToneRequested), 1);
    }

    /// Mute waits for the start sound to end so its tail isn't cut, but the
    /// ready event doesn't wait for it.
    #[test]
    fn mute_follows_the_start_sound_and_ready_does_not_wait_for_it() {
        let (steps, controls) = fake(true, Mic::Works);
        controls.mic_gate.send(()).unwrap();
        controls.first_audio.send(()).unwrap();
        let worker = started(run_start(steps.clone()));

        assert!(controls.wait_for(Step::Ready, WAIT));
        assert!(
            !controls.wait_for(Step::Mute, WATCH),
            "muted while the start sound was playing"
        );
        controls.tone_gate.send(()).unwrap();
        worker.join().unwrap();
        assert!(steps.at(Step::ToneEnded) <= steps.at(Step::Mute));
        assert_eq!(steps.count(Step::ToneRequested), 1);
        assert_eq!(steps.count(Step::Mute), 1);
    }

    /// A stop while the start sound is still playing: the worker must not
    /// mute afterwards, or system audio stays muted past the recording.
    #[test]
    fn a_session_stopped_during_the_start_sound_is_not_muted() {
        let (steps, controls) = fake(true, Mic::Works);
        controls.mic_gate.send(()).unwrap();
        controls.first_audio.send(()).unwrap();
        let worker = started(run_start(steps.clone()));

        assert!(controls.wait_for(Step::Ready, WAIT));
        steps.0.stopped.store(true, Ordering::SeqCst);
        controls.tone_gate.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(steps.count(Step::Mute), 0);
    }

    #[test]
    fn a_refused_start_plays_nothing_and_opens_nothing() {
        let (steps, _controls) = fake(false, Mic::Works);
        assert!(matches!(run_start(steps.clone()), StartOutcome::Refused));
        assert!(steps.steps().is_empty());
    }

    #[test]
    fn a_start_into_a_busy_microphone_plays_nothing() {
        let (steps, controls) = fake(true, Mic::Busy);
        drop(controls);
        assert!(matches!(run_start(steps.clone()), StartOutcome::Failed(_)));
        assert!(steps.steps().is_empty());
    }

    /// A microphone that fails to open after the press was taken: the start
    /// sound has played once, and there's no ready event or mute.
    #[test]
    fn a_failed_microphone_open_leaves_only_the_start_sound() {
        let (steps, controls) = fake(true, Mic::Fails);
        drop(controls);
        assert!(matches!(run_start(steps.clone()), StartOutcome::Failed(_)));
        let deadline = Instant::now() + WAIT;
        while steps.count(Step::ToneEnded) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(steps.steps(), [Step::ToneRequested, Step::ToneEnded]);
    }
}
