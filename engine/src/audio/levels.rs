//! Player-facing volume levels (master / music / effects).
//!
//! Every source the [`super::AudioContext`] plays is wrapped in [`Levels`],
//! which re-reads its bus gain from these process-wide atomics while it plays,
//! so moving a volume slider reaches sounds that are already playing - no
//! per-sink bookkeeping of "the gain before the player's volume".

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use rodio::{Sample, Source};

/// Which player volume a source answers to (besides master).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bus {
    Music,
    Effects,
    /// Master only - streamed cutscene audio.
    Master,
}

const ONE: u32 = 0x3f80_0000; // 1.0f32.to_bits()
static MASTER: AtomicU32 = AtomicU32::new(ONE);
static MUSIC: AtomicU32 = AtomicU32::new(ONE);
static EFFECTS: AtomicU32 = AtomicU32::new(ONE);

/// Set the player's volume levels, each clamped to `0..=1`.
pub fn set_volume_levels(master: f32, music: f32, effects: f32) {
    let store = |slot: &AtomicU32, value: f32| {
        slot.store(value.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed)
    };
    store(&MASTER, master);
    store(&MUSIC, music);
    store(&EFFECTS, effects);
}

/// The effective gain for `bus`: master times the bus's own level.
pub fn bus_gain(bus: Bus) -> f32 {
    let read = |slot: &AtomicU32| f32::from_bits(slot.load(Ordering::Relaxed));
    let master = read(&MASTER);
    match bus {
        Bus::Music => master * read(&MUSIC),
        Bus::Effects => master * read(&EFFECTS),
        Bus::Master => master,
    }
}

/// Samples between gain refreshes - ~5 ms at 48 kHz stereo, so a slider
/// change is heard at once without an atomic load per sample.
const REFRESH_SAMPLES: u32 = 512;

/// A source scaled by its [`Bus`]'s live gain.
#[derive(Clone)]
pub struct Levels<S> {
    inner: S,
    bus: Bus,
    gain: f32,
    until_refresh: u32,
}

impl<S> Levels<S> {
    pub fn new(inner: S, bus: Bus) -> Self {
        Self {
            inner,
            bus,
            gain: bus_gain(bus),
            until_refresh: REFRESH_SAMPLES,
        }
    }
}

impl<S> Iterator for Levels<S>
where
    S: Source,
    S::Item: Sample,
{
    type Item = S::Item;

    fn next(&mut self) -> Option<S::Item> {
        if self.until_refresh == 0 {
            self.gain = bus_gain(self.bus);
            self.until_refresh = REFRESH_SAMPLES;
        }
        self.until_refresh -= 1;
        let sample = self.inner.next()?;
        Some(if self.gain == 1.0 {
            sample
        } else {
            sample.amplify(self.gain)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S> Source for Levels<S>
where
    S: Source,
    S::Item: Sample,
{
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::buffer::SamplesBuffer;

    #[test]
    fn levels_scale_by_master_times_bus() {
        set_volume_levels(0.5, 0.5, 1.0);
        assert_eq!(bus_gain(Bus::Music), 0.25);
        assert_eq!(bus_gain(Bus::Effects), 0.5);
        assert_eq!(bus_gain(Bus::Master), 0.5);
        let out: Vec<f32> = Levels::new(
            SamplesBuffer::new(1, 44_100, vec![1.0f32, -1.0]),
            Bus::Effects,
        )
        .collect();
        assert_eq!(out, vec![0.5, -0.5]);
        set_volume_levels(1.0, 1.0, 1.0);
    }
}
