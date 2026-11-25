//! "Modifier alone" trigger: the transcribe shortcut's modifiers, pressed on
//! their own (Option for `option+space`), also start dictation.
//!
//! A pure state machine fed with raw keyboard events and the binding's own
//! chord edges, so every decision is unit-tested without a keyboard. The
//! idea of starting on the bare modifier's key-down and discarding the
//! session when it turns out to be a chord (Option+E, Option+click) comes from
//! Hex (MIT, https://github.com/anomalyco/hex, `src/suppression/hotkey.rs`).
//!
//! Recording starts on the modifier's key-down; whether it was a tap or a hold
//! is decided later by the coordinator, as for any other press. The modifier
//! is never swallowed: other apps still see it, so Option+letter typing works.
//!
//! * Another key or a mouse button while the modifier is held: the press was
//!   typing, not dictation. Its session is discarded.
//! * An extra modifier within the hold threshold (Option then Shift): a chord
//!   in the making, discarded too.
//! * The binding's own chord (Space with Option held): the chord takes the
//!   press over. Tap vs hold is timed from the chord's press, so a slow
//!   Option followed by a Space tap still reads as a tap.

use handy_keys::{Hotkey, KeyEvent, Modifiers};
use std::time::{Duration, Instant};

/// What the trigger asks of the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerAction {
    /// The modifiers went down alone: press the transcribe binding.
    Press { at: Instant },
    /// The modifiers came up: release the transcribe binding.
    Release { at: Instant },
    /// The press at `pressed_at` was typing: drop the recording it started.
    Discard { pressed_at: Instant },
    /// The binding's chord took over the press at `pressed_at`: measure the
    /// hold from `at` instead.
    Rebase { pressed_at: Instant, at: Instant },
    /// A chord edge to pass on unchanged.
    Chord { pressed: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// The modifiers aren't held, or were pressed without a press sent.
    Idle,
    /// The modifiers are held alone and a press was sent at `at`.
    Pressed { at: Instant },
    /// The binding's chord took this hold over: its release is the chord's.
    Chorded,
    /// The press was discarded or never counted. Inert until the modifiers
    /// come up.
    Dirty,
}

/// Every modifier group, so a trigger's groups can be picked out.
const GROUPS: [Modifiers; 5] = [
    Modifiers::CMD,
    Modifiers::SHIFT,
    Modifiers::CTRL,
    Modifiers::OPT,
    Modifiers::FN,
];

pub struct ModifierTrigger {
    /// The binding, such as `option+space`.
    chord: Hotkey,
    /// The binding's modifiers group by group, for "held, maybe with others".
    groups: Modifiers,
    hold_threshold: Duration,
    state: State,
    /// Whether the trigger modifiers were held at the last modifier event.
    held: bool,
}

impl ModifierTrigger {
    /// A trigger for `chord`'s modifiers, or `None` when the binding has no
    /// modifiers or no key (it would be the trigger itself).
    pub fn new(chord: Hotkey, hold_threshold: Duration) -> Option<Self> {
        if chord.modifiers.is_empty() || chord.key.is_none() {
            return None;
        }
        let groups = GROUPS
            .into_iter()
            .filter(|group| chord.modifiers.intersects(*group))
            .fold(Modifiers::empty(), |acc, group| acc | group);
        Some(Self {
            chord,
            groups,
            hold_threshold,
            state: State::Idle,
            held: false,
        })
    }

    /// Exactly the trigger modifiers are held.
    fn alone(&self, modifiers: Modifiers) -> bool {
        self.chord.modifiers.matches(modifiers)
    }

    /// The trigger modifiers are held, perhaps with others.
    fn covered(&self, modifiers: Modifiers) -> bool {
        self.chord.modifiers.matches(modifiers & self.groups)
    }

    /// A raw keyboard or mouse event. `can_start` is false when a press must
    /// not start anything (macOS Secure Input hides the keys that would
    /// discard it).
    pub fn on_key_event(
        &mut self,
        event: &KeyEvent,
        now: Instant,
        can_start: bool,
    ) -> Option<TriggerAction> {
        let Some(key) = event.key else {
            return self.on_modifiers(event.modifiers, now, can_start);
        };
        // A key the user presses while holding the modifiers carries them.
        // One without them was posted by a program, such as Handy pasting
        // the last transcription, and says nothing about this press.
        if !event.is_key_down || !self.covered(event.modifiers) {
            return None;
        }
        // The chord's own key is the chord, reported by the hotkey manager
        // (before or after this event), not typing.
        let is_chord = Some(key) == self.chord.key && self.alone(event.modifiers);
        match self.state {
            State::Pressed { at } if !is_chord => {
                self.state = State::Dirty;
                Some(TriggerAction::Discard { pressed_at: at })
            }
            _ => None,
        }
    }

    fn on_modifiers(
        &mut self,
        modifiers: Modifiers,
        now: Instant,
        can_start: bool,
    ) -> Option<TriggerAction> {
        let held = self.covered(modifiers);
        let went_down = held && !self.held;
        self.held = held;
        match self.state {
            State::Idle if went_down => {
                // Only the trigger's own down edge with nothing else held
                // counts: Shift then Option is a chord, and so is releasing
                // Shift from Shift+Option.
                if can_start && self.alone(modifiers) {
                    self.state = State::Pressed { at: now };
                    Some(TriggerAction::Press { at: now })
                } else {
                    self.state = State::Dirty;
                    None
                }
            }
            State::Pressed { .. } if !held => {
                self.state = State::Idle;
                Some(TriggerAction::Release { at: now })
            }
            State::Pressed { at }
                if !self.alone(modifiers)
                    && now.saturating_duration_since(at) < self.hold_threshold =>
            {
                self.state = State::Dirty;
                Some(TriggerAction::Discard { pressed_at: at })
            }
            State::Chorded | State::Dirty if !held => {
                self.state = State::Idle;
                None
            }
            _ => None,
        }
    }

    /// The binding's chord (from the hotkey manager) went down or up.
    pub fn on_chord(&mut self, pressed: bool, now: Instant) -> Option<TriggerAction> {
        match self.state {
            State::Pressed { at } if pressed => {
                self.state = State::Chorded;
                Some(TriggerAction::Rebase {
                    pressed_at: at,
                    at: now,
                })
            }
            _ => {
                // A chord whose modifiers we never saw go down: the modifiers'
                // release belongs to it, so it must not press anything later.
                if pressed && self.state == State::Idle {
                    self.state = State::Chorded;
                }
                Some(TriggerAction::Chord { pressed })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use handy_keys::Key;

    const THRESHOLD: Duration = Duration::from_millis(300);
    const OPT: Modifiers = Modifiers::OPT_LEFT;

    fn trigger() -> ModifierTrigger {
        ModifierTrigger::new("option+space".parse().unwrap(), THRESHOLD).unwrap()
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

    fn ms(t0: Instant, n: u64) -> Instant {
        t0 + Duration::from_millis(n)
    }

    #[test]
    fn needs_modifiers_and_a_key() {
        assert!(ModifierTrigger::new("option".parse().unwrap(), THRESHOLD).is_none());
        assert!(ModifierTrigger::new("f13".parse().unwrap(), THRESHOLD).is_none());
        assert!(ModifierTrigger::new("ctrl+shift+space".parse().unwrap(), THRESHOLD).is_some());
    }

    #[test]
    fn bare_option_presses_on_down_and_releases_on_up() {
        let mut t = trigger();
        let t0 = Instant::now();
        assert_eq!(
            t.on_key_event(&mods(OPT, true), t0, true),
            Some(TriggerAction::Press { at: t0 })
        );
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 120), true),
            Some(TriggerAction::Release { at: ms(t0, 120) })
        );
        // And again: the trigger is ready for the next press.
        assert_eq!(
            t.on_key_event(&mods(Modifiers::OPT_RIGHT, true), ms(t0, 500), true),
            Some(TriggerAction::Press { at: ms(t0, 500) })
        );
    }

    #[test]
    fn option_letter_discards_and_stays_inert_until_option_is_up() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&key(OPT, Key::E, true), ms(t0, 60), true),
            Some(TriggerAction::Discard { pressed_at: t0 })
        );
        // More typing and the release send nothing.
        assert_eq!(
            t.on_key_event(&key(OPT, Key::E, false), ms(t0, 90), true),
            None
        );
        assert_eq!(
            t.on_key_event(&key(OPT, Key::U, true), ms(t0, 900), true),
            None
        );
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 1000), true),
            None
        );
        // The next bare press counts again.
        assert_eq!(
            t.on_key_event(&mods(OPT, true), ms(t0, 1200), true),
            Some(TriggerAction::Press { at: ms(t0, 1200) })
        );
    }

    #[test]
    fn a_key_long_into_the_hold_still_discards() {
        // Option held past the threshold, then an arrow key: still typing
        // (word navigation), never dictation to paste.
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&key(OPT, Key::LeftArrow, true), ms(t0, 2000), true),
            Some(TriggerAction::Discard { pressed_at: t0 })
        );
    }

    #[test]
    fn option_click_discards() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&key(OPT, Key::MouseLeft, true), ms(t0, 400), true),
            Some(TriggerAction::Discard { pressed_at: t0 })
        );
    }

    #[test]
    fn extra_modifier_discards_only_within_the_threshold() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&mods(OPT | Modifiers::SHIFT_LEFT, true), ms(t0, 100), true),
            Some(TriggerAction::Discard { pressed_at: t0 })
        );

        let mut t = trigger();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&mods(OPT | Modifiers::SHIFT_LEFT, true), ms(t0, 400), true),
            None
        );
        // Shift comes back up: still the same hold, released with Option.
        assert_eq!(t.on_key_event(&mods(OPT, false), ms(t0, 500), true), None);
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 700), true),
            Some(TriggerAction::Release { at: ms(t0, 700) })
        );
    }

    #[test]
    fn both_options_count_as_one_hold() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        let both = OPT | Modifiers::OPT_RIGHT;
        assert_eq!(t.on_key_event(&mods(both, true), ms(t0, 50), true), None);
        assert_eq!(
            t.on_key_event(&mods(Modifiers::OPT_RIGHT, false), ms(t0, 80), true),
            None
        );
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 90), true),
            Some(TriggerAction::Release { at: ms(t0, 90) })
        );
    }

    #[test]
    fn option_pressed_while_another_modifier_is_held_does_nothing() {
        let mut t = trigger();
        let t0 = Instant::now();
        let shift = Modifiers::SHIFT_LEFT;
        assert_eq!(t.on_key_event(&mods(shift, true), t0, true), None);
        assert_eq!(
            t.on_key_event(&mods(shift | OPT, true), ms(t0, 20), true),
            None
        );
        // Releasing Shift leaves Option alone, but that's no Option press.
        assert_eq!(t.on_key_event(&mods(OPT, false), ms(t0, 40), true), None);
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 60), true),
            None
        );
    }

    #[test]
    fn secure_input_starts_nothing_for_that_hold() {
        let mut t = trigger();
        let t0 = Instant::now();
        assert_eq!(t.on_key_event(&mods(OPT, true), t0, false), None);
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 100), false),
            None
        );
    }

    #[test]
    fn chord_takes_over_and_absorbs_the_option_release() {
        // Option down, then Space: one session, timed from Space. The raw
        // Space key-down arriving first or not at all changes nothing.
        for raw_space in [true, false] {
            let mut t = trigger();
            let t0 = Instant::now();
            t.on_key_event(&mods(OPT, true), t0, true);
            if raw_space {
                assert_eq!(
                    t.on_key_event(&key(OPT, Key::Space, true), ms(t0, 400), true),
                    None
                );
            }
            assert_eq!(
                t.on_chord(true, ms(t0, 401)),
                Some(TriggerAction::Rebase {
                    pressed_at: t0,
                    at: ms(t0, 401)
                })
            );
            assert_eq!(
                t.on_chord(false, ms(t0, 480)),
                Some(TriggerAction::Chord { pressed: false })
            );
            assert_eq!(
                t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 600), true),
                None
            );
        }
    }

    #[test]
    fn raw_space_after_the_chord_is_not_typing() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        t.on_chord(true, ms(t0, 100));
        assert_eq!(
            t.on_key_event(&key(OPT, Key::Space, true), ms(t0, 101), true),
            None
        );
    }

    #[test]
    fn second_chord_in_the_same_hold_passes_through() {
        // Option held: Space taps after the first are ordinary chord presses
        // (the second one stops a locked session).
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        t.on_chord(true, ms(t0, 100));
        t.on_chord(false, ms(t0, 150));
        assert_eq!(
            t.on_chord(true, ms(t0, 900)),
            Some(TriggerAction::Chord { pressed: true })
        );
    }

    #[test]
    fn chord_without_a_seen_option_press_never_presses_later() {
        // The raw listener missed Option going down: the chord passes through,
        // and a late modifier event can't add a second press.
        let mut t = trigger();
        let t0 = Instant::now();
        assert_eq!(
            t.on_chord(true, t0),
            Some(TriggerAction::Chord { pressed: true })
        );
        assert_eq!(t.on_key_event(&mods(OPT, true), ms(t0, 1), true), None);
        assert_eq!(
            t.on_chord(false, ms(t0, 80)),
            Some(TriggerAction::Chord { pressed: false })
        );
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 100), true),
            None
        );
        assert_eq!(
            t.on_key_event(&mods(OPT, true), ms(t0, 500), true),
            Some(TriggerAction::Press { at: ms(t0, 500) })
        );
    }

    #[test]
    fn chord_after_a_discard_passes_through() {
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        t.on_key_event(&key(OPT, Key::E, true), ms(t0, 50), true);
        assert_eq!(
            t.on_chord(true, ms(t0, 300)),
            Some(TriggerAction::Chord { pressed: true })
        );
    }

    #[test]
    fn a_posted_key_without_option_does_not_discard() {
        // Handy pastes with Cmd+V while the user holds Option for the next
        // dictation: the posted keys carry only their own flags.
        let mut t = trigger();
        let t0 = Instant::now();
        t.on_key_event(&mods(OPT, true), t0, true);
        assert_eq!(
            t.on_key_event(&key(Modifiers::CMD_LEFT, Key::V, true), ms(t0, 50), true),
            None
        );
        assert_eq!(
            t.on_key_event(&mods(Modifiers::empty(), false), ms(t0, 900), true),
            Some(TriggerAction::Release { at: ms(t0, 900) })
        );
    }

    #[test]
    fn key_ups_and_idle_keys_are_ignored() {
        let mut t = trigger();
        let t0 = Instant::now();
        assert_eq!(
            t.on_key_event(&key(Modifiers::empty(), Key::A, true), t0, true),
            None
        );
        assert_eq!(
            t.on_key_event(&key(Modifiers::empty(), Key::A, false), t0, true),
            None
        );
        t.on_key_event(&mods(OPT, true), ms(t0, 10), true);
        assert_eq!(
            t.on_key_event(&key(OPT, Key::A, false), ms(t0, 20), true),
            None
        );
    }
}
