//! Keeps the feedback-sound output open for a short while after each sound.
//!
//! Opening an output stream costs ~15-60 ms on macOS, which used to come
//! before every start sound. The stream now stays open until
//! [`IDLE_TIMEOUT`] passes with nothing playing, so back-to-back dictations
//! reuse it. It's reopened when the output device changes or the stream
//! reports a fatal error, and a reaper thread closes it once idle so Handy
//! doesn't hold the device open.

use crate::audio_toolkit::audio::output_device_name;
use log::{debug, warn};
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use rodio::{cpal, DeviceSinkBuilder, MixerDeviceSink};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// How long the output stays open after the last sound ends.
const IDLE_TIMEOUT: Duration = Duration::from_secs(10);

/// How often the reaper checks back while a sound is playing, so the idle
/// clock is noticed soon after it starts.
const PLAYING_POLL: Duration = Duration::from_secs(1);

/// An open output and what's needed to decide whether it can be reused.
struct Entry<S> {
    /// Identifies the device the stream was opened on.
    device: String,
    sink: Arc<S>,
    /// Set by the stream's error callback once the stream is unusable.
    failed: Arc<AtomicBool>,
    /// Sounds currently playing through `sink`.
    playing: usize,
    /// When the last sound ended; `None` while one is playing.
    idle_since: Option<Instant>,
}

/// What the reaper should do next.
#[derive(Debug, PartialEq)]
enum Reap {
    /// Nothing is open; the reaper can stop.
    Empty,
    /// The stream went idle and was closed; the reaper can stop.
    Closed,
    /// Check again after this long.
    Wait(Duration),
}

/// The reuse rules, kept apart from rodio so they can be tested without a
/// device. Every method takes `now` so tests control the clock.
struct WarmOutput<S> {
    idle_timeout: Duration,
    entry: Option<Entry<S>>,
    reaper_running: bool,
}

impl<S> WarmOutput<S> {
    const fn new(idle_timeout: Duration) -> Self {
        Self {
            idle_timeout,
            entry: None,
            reaper_running: false,
        }
    }

    /// Returns the open stream for `device`, opening one with `open` if
    /// there's none, it's for another device, it failed, or it's idle past the
    /// timeout. `open` gets the flag its error callback should set. A replaced
    /// stream with nothing playing is closed before the open, since some
    /// outputs allow only one stream; one still playing is returned for the
    /// caller to drop outside the lock.
    #[allow(clippy::type_complexity)]
    fn acquire<E>(
        &mut self,
        device: &str,
        now: Instant,
        open: impl FnOnce(Arc<AtomicBool>) -> Result<S, E>,
    ) -> Result<(Arc<S>, Option<Arc<S>>), E> {
        let reusable = self.entry.as_ref().is_some_and(|e| {
            e.device == device && !e.failed.load(Ordering::SeqCst) && !self.expired(e, now)
        });
        let mut stale = None;
        if !reusable {
            if let Some(old) = self.entry.take() {
                if old.playing > 0 {
                    stale = Some(old.sink);
                }
            }
            let failed = Arc::new(AtomicBool::new(false));
            let sink = Arc::new(open(failed.clone())?);
            self.entry = Some(Entry {
                device: device.to_string(),
                sink,
                failed,
                playing: 0,
                idle_since: None,
            });
        }
        let entry = self.entry.as_mut().expect("entry was just ensured");
        entry.playing += 1;
        entry.idle_since = None;
        Ok((entry.sink.clone(), stale))
    }

    /// Marks a sound on `sink` as finished, starting the idle clock when it
    /// was the last one. A stream that was already replaced is ignored.
    fn release(&mut self, sink: &Arc<S>, now: Instant) {
        if let Some(entry) = self.entry.as_mut() {
            if Arc::ptr_eq(&entry.sink, sink) {
                entry.playing = entry.playing.saturating_sub(1);
                if entry.playing == 0 {
                    entry.idle_since = Some(now);
                }
            }
        }
    }

    /// Takes the stream if it's been idle for the timeout, for the caller to
    /// drop.
    fn reap(&mut self, now: Instant) -> (Reap, Option<Arc<S>>) {
        let Some(entry) = self.entry.as_ref() else {
            return (Reap::Empty, None);
        };
        if self.expired(entry, now) {
            return (Reap::Closed, self.entry.take().map(|e| e.sink));
        }
        let wait = match entry.idle_since {
            Some(since) => self.idle_timeout.saturating_sub(now - since),
            None => PLAYING_POLL.min(self.idle_timeout),
        };
        (Reap::Wait(wait), None)
    }

    fn expired(&self, entry: &Entry<S>, now: Instant) -> bool {
        entry.playing == 0
            && entry
                .idle_since
                .is_some_and(|since| now.saturating_duration_since(since) >= self.idle_timeout)
    }
}

static OUTPUT: LazyLock<Mutex<WarmOutput<MixerDeviceSink>>> =
    LazyLock::new(|| Mutex::new(WarmOutput::new(IDLE_TIMEOUT)));

fn lock() -> std::sync::MutexGuard<'static, WarmOutput<MixerDeviceSink>> {
    // A panic mid-play can't leave the cache inconsistent in a way that
    // matters: at worst a stream stays counted as playing until replaced.
    OUTPUT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Picks the device the way Handy always has: the selected output if it's
/// present, else the system default. Also says whether it's the default,
/// whose stream follows the system default on macOS.
fn resolve_device(
    selected: Option<&str>,
) -> Result<(cpal::Device, bool), Box<dyn std::error::Error>> {
    if let Some(name) = selected.filter(|n| *n != "Default") {
        let host = crate::audio_toolkit::get_cpal_host();
        let found = host
            .output_devices()?
            .find(|device| output_device_name(device).as_deref() == Some(name));
        match found {
            Some(device) => return Ok((device, false)),
            None => warn!("Device '{}' not found, using default device", name),
        }
    } else {
        debug!("Using default device");
    }
    // The same lookup rodio's `from_default_device` does.
    cpal::default_host()
        .default_output_device()
        .map(|device| (device, true))
        .ok_or_else(|| "no default output device".into())
}

fn open_sink(
    device: cpal::Device,
    failed: Arc<AtomicBool>,
) -> Result<MixerDeviceSink, Box<dyn std::error::Error>> {
    let mut sink = DeviceSinkBuilder::from_device(device)?
        // rodio defaults to a fixed 2048-frame buffer, which some Bluetooth
        // outputs reject (#2524).
        .with_buffer_size(cpal::BufferSize::Default)
        .with_error_callback(move |err: cpal::Error| {
            use cpal::ErrorKind;
            match err.kind() {
                // The stream keeps working through these.
                ErrorKind::Xrun | ErrorKind::RealtimeDenied | ErrorKind::DeviceChanged => {
                    debug!("Feedback output: {err}");
                }
                _ => {
                    warn!("Feedback output failed, reopening on next sound: {err}");
                    failed.store(true, Ordering::SeqCst);
                }
            }
        })
        .open_stream()?;
    // Closing it is expected, so skip rodio's stderr note on drop.
    sink.log_on_drop(false);
    Ok(sink)
}

/// Runs `play` on an open output for the selected device, opening one only
/// if needed, and keeps it open for [`IDLE_TIMEOUT`] afterwards.
pub fn with_output<T>(
    selected_device: Option<&str>,
    play: impl FnOnce(&MixerDeviceSink) -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    let (device, follows_default) = resolve_device(selected_device)?;
    // Names can repeat, so tell devices apart by ID. A stream on the default
    // and one on the same device picked by name behave differently once the
    // system default changes, so they don't share a key.
    let Ok(id) = device.id() else {
        // Without an ID there's no telling devices apart, so don't reuse.
        let sink = open_sink(device, Arc::new(AtomicBool::new(false)))?;
        return play(&sink);
    };
    let key = format!("{}{id}", if follows_default { "default:" } else { "" });

    let (sink, stale, start_reaper) = {
        let mut output = lock();
        let (sink, stale) = output.acquire(&key, Instant::now(), |failed| {
            let started = Instant::now();
            let sink = open_sink(device, failed);
            debug!(
                "Opened feedback output '{}' in {:?}",
                key,
                started.elapsed()
            );
            sink
        })?;
        let start_reaper = !output.reaper_running;
        output.reaper_running = true;
        (sink, stale, start_reaper)
    };
    drop(stale);
    if start_reaper {
        spawn_reaper();
    }

    // Released on drop so a panicking `play` can't keep the stream open. The
    // handle goes under the lock too, so a sound that replaces this stream
    // can't open the device while this one still holds it.
    struct Release(Option<Arc<MixerDeviceSink>>);
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(sink) = self.0.take() {
                let mut output = lock();
                output.release(&sink, Instant::now());
                drop(sink);
            }
        }
    }
    let lease = Release(Some(sink));
    play(lease.0.as_ref().expect("held until drop"))
}

fn spawn_reaper() {
    let spawned = thread::Builder::new()
        .name("feedback-output-reaper".into())
        .spawn(|| loop {
            let next = {
                let mut output = lock();
                let (next, closed) = output.reap(Instant::now());
                if !matches!(next, Reap::Wait(_)) {
                    output.reaper_running = false;
                }
                // Close under the lock, so a sound arriving now waits for the
                // device to be free before reopening it (some outputs allow
                // only one stream).
                drop(closed);
                next
            };
            match next {
                Reap::Wait(wait) => thread::sleep(wait.max(Duration::from_millis(50))),
                Reap::Closed => {
                    debug!("Closed idle feedback output");
                    return;
                }
                Reap::Empty => return,
            }
        });
    if let Err(e) = spawned {
        // The stream then closes on the next sound after the timeout instead.
        warn!("Failed to start feedback output reaper: {e}");
        lock().reaper_running = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDLE: Duration = Duration::from_secs(10);

    /// Opens a numbered fake stream and counts opens.
    fn opener(opens: &mut u32) -> impl FnOnce(Arc<AtomicBool>) -> Result<u32, ()> + '_ {
        move |_| {
            *opens += 1;
            Ok(*opens)
        }
    }

    #[test]
    fn reuses_stream_within_idle_timeout() {
        let mut out = WarmOutput::new(IDLE);
        let mut opens = 0;
        let t0 = Instant::now();

        let (a, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        out.release(&a, t0);
        let later = t0 + IDLE - Duration::from_millis(1);
        let (b, _) = out.acquire("Speakers", later, opener(&mut opens)).unwrap();

        assert_eq!(opens, 1);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn reopens_after_idle_timeout() {
        let mut out = WarmOutput::new(IDLE);
        let mut opens = 0;
        let t0 = Instant::now();

        let (a, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        out.release(&a, t0);
        let (b, stale) = out
            .acquire("Speakers", t0 + IDLE, opener(&mut opens))
            .unwrap();

        assert_eq!(opens, 2);
        assert!(!Arc::ptr_eq(&a, &b));
        // Nothing was playing on it, so it closed before the reopen.
        assert!(stale.is_none());
        assert_eq!(Arc::strong_count(&a), 1);
    }

    #[test]
    fn reopens_when_device_changes() {
        let mut out = WarmOutput::new(IDLE);
        let mut opens = 0;
        let t0 = Instant::now();

        let (a, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        out.release(&a, t0);
        out.acquire("Headphones", t0, opener(&mut opens)).unwrap();

        assert_eq!(opens, 2);
    }

    #[test]
    fn reopens_after_stream_error() {
        let mut out = WarmOutput::new(IDLE);
        let mut flag = None;
        let t0 = Instant::now();

        let (a, _) = out
            .acquire("Speakers", t0, |failed| {
                flag = Some(failed);
                Ok::<_, ()>(1)
            })
            .unwrap();
        out.release(&a, t0);
        flag.unwrap().store(true, Ordering::SeqCst);
        let mut opens = 0;
        out.acquire("Speakers", t0, opener(&mut opens)).unwrap();

        assert_eq!(opens, 1);
    }

    #[test]
    fn failed_open_leaves_nothing_cached() {
        let mut out = WarmOutput::<u32>::new(IDLE);
        let t0 = Instant::now();

        assert!(out.acquire("Speakers", t0, |_| Err(())).is_err());
        assert_eq!(out.reap(t0).0, Reap::Empty);
    }

    #[test]
    fn idle_clock_starts_when_the_last_sound_ends() {
        let mut out = WarmOutput::new(IDLE);
        let mut opens = 0;
        let t0 = Instant::now();

        // Two overlapping sounds; the stream must outlive both.
        let (a, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        let (b, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        out.release(&a, t0);
        assert_eq!(out.reap(t0 + IDLE * 2).0, Reap::Wait(PLAYING_POLL));

        let end = t0 + IDLE * 2;
        out.release(&b, end);
        assert_eq!(
            out.reap(end + Duration::from_secs(4)).0,
            Reap::Wait(Duration::from_secs(6))
        );
        let (reap, closed) = out.reap(end + IDLE);
        assert_eq!(reap, Reap::Closed);
        assert!(closed.is_some_and(|c| Arc::ptr_eq(&c, &a)));
        assert_eq!(out.reap(end + IDLE).0, Reap::Empty);
        assert_eq!(opens, 1);
    }

    #[test]
    fn release_of_replaced_stream_is_ignored() {
        let mut out = WarmOutput::new(IDLE);
        let mut opens = 0;
        let t0 = Instant::now();

        // A sound still playing on the old device when the device changes.
        let (old, _) = out.acquire("Speakers", t0, opener(&mut opens)).unwrap();
        let (new, stale) = out.acquire("Headphones", t0, opener(&mut opens)).unwrap();
        // The caller still holds `old`, so dropping `stale` doesn't stop it.
        assert!(stale.is_some_and(|s| Arc::ptr_eq(&s, &old)));
        out.release(&old, t0);

        // The new stream is still counted as playing, so it isn't reaped.
        assert_eq!(out.reap(t0 + IDLE * 2).0, Reap::Wait(PLAYING_POLL));
        out.release(&new, t0);
        assert_eq!(out.reap(t0 + IDLE).0, Reap::Closed);
    }

    /// Opens the real default output twice. Needs an output device, which
    /// CI runners lack, so run it by hand:
    /// `cargo test feedback_output -- --ignored --nocapture`
    #[test]
    #[ignore = "needs an audio output device"]
    fn second_sound_skips_the_open() {
        let time_open = || {
            let started = Instant::now();
            with_output(None, |_| Ok(started.elapsed())).unwrap()
        };
        let cold = time_open();
        let warm = time_open();
        println!("cold {cold:?}, warm {warm:?}");
        assert!(warm < cold);
    }
}
