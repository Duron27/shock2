//! Where the mixed sound goes, and the sinks that feed it.
//!
//! On the desktop that is cpal's default device through rodio's
//! `OutputStream`. The Xbox 360 has no cpal backend: there the engine runs
//! rodio's mixer itself and a thread of its own hands the mix to xsdk's
//! audio driver a frame at a time (xbox360/README.md).
//!
//! rodio only builds a `SpatialSink` from an `OutputStreamHandle`, which only
//! cpal can make, so [`SpatialSink`] here is rodio's own, built on a plain
//! [`Sink`] instead.
#![cfg_attr(
    not(target_os = "xbox360"),
    allow(
        dead_code,
        reason = "the mixer target is the console's; host builds only test it"
    )
)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rodio::cpal::FromSample;
use rodio::dynamic_mixer::DynamicMixerController;
use rodio::source::Spatial;
#[cfg(not(target_os = "xbox360"))]
use rodio::{OutputStream, OutputStreamHandle};
use rodio::{Sample, Sink, Source};

/// Channels and rate of the mix handed to xsdk; it upmixes to its six.
pub const MIX_CHANNELS: u16 = 2;
pub const MIX_RATE_HZ: u32 = 48_000;

pub struct Output {
    target: Target,
}

enum Target {
    /// cpal's device; the stream plays while it is held.
    #[cfg(not(target_os = "xbox360"))]
    Device {
        _stream: OutputStream,
        handle: OutputStreamHandle,
    },
    /// A mixer something else drains: xsdk's feeder on the console.
    Mixer(Arc<DynamicMixerController<f32>>),
}

impl Output {
    pub fn open() -> Output {
        #[cfg(target_os = "xbox360")]
        {
            Output::mixer(xsdk::open().expect("xsdk audio did not open"))
        }
        #[cfg(not(target_os = "xbox360"))]
        {
            let (stream, handle) = OutputStream::try_default().expect("no audio output device");
            Output {
                target: Target::Device {
                    _stream: stream,
                    handle,
                },
            }
        }
    }

    /// Sinks play into `mixer`; whoever holds its other end plays the mix.
    pub fn mixer(mixer: Arc<DynamicMixerController<f32>>) -> Output {
        Output {
            target: Target::Mixer(mixer),
        }
    }

    pub fn sink(&self) -> Sink {
        match &self.target {
            #[cfg(not(target_os = "xbox360"))]
            Target::Device { handle, .. } => Sink::try_new(handle).expect("audio device gone"),
            Target::Mixer(mixer) => {
                // What `Sink::try_new` does with a handle's mixer.
                let (sink, queue) = Sink::new_idle();
                mixer.add(queue);
                sink
            }
        }
    }

    pub fn spatial_sink(
        &self,
        emitter_position: [f32; 3],
        left_ear: [f32; 3],
        right_ear: [f32; 3],
    ) -> SpatialSink {
        SpatialSink {
            sink: self.sink(),
            positions: Arc::new(Mutex::new(SoundPositions {
                emitter_position,
                left_ear,
                right_ear,
            })),
        }
    }
}

/// rodio's `SpatialSink`: a sink whose sounds pan between two ears.
pub struct SpatialSink {
    sink: Sink,
    positions: Arc<Mutex<SoundPositions>>,
}

struct SoundPositions {
    emitter_position: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
}

impl SpatialSink {
    pub fn set_emitter_position(&self, pos: [f32; 3]) {
        self.positions.lock().unwrap().emitter_position = pos;
    }

    pub fn set_left_ear_position(&self, pos: [f32; 3]) {
        self.positions.lock().unwrap().left_ear = pos;
    }

    pub fn set_right_ear_position(&self, pos: [f32; 3]) {
        self.positions.lock().unwrap().right_ear = pos;
    }

    pub fn append<S>(&self, source: S)
    where
        S: Source + Send + 'static,
        f32: FromSample<S::Item>,
        S::Item: Sample + Send,
    {
        let positions = self.positions.clone();
        let now = self.positions.lock().unwrap();
        let source = Spatial::new(source, now.emitter_position, now.left_ear, now.right_ear)
            .periodic_access(Duration::from_millis(10), move |spatial| {
                let pos = positions.lock().unwrap();
                spatial.set_positions(pos.emitter_position, pos.left_ear, pos.right_ear);
            });
        self.sink.append(source);
    }

    pub fn volume(&self) -> f32 {
        self.sink.volume()
    }

    pub fn set_volume(&self, value: f32) {
        self.sink.set_volume(value);
    }

    pub fn speed(&self) -> f32 {
        self.sink.speed()
    }

    pub fn set_speed(&self, value: f32) {
        self.sink.set_speed(value);
    }

    pub fn play(&self) {
        self.sink.play();
    }

    pub fn pause(&self) {
        self.sink.pause();
    }

    pub fn is_paused(&self) -> bool {
        self.sink.is_paused()
    }

    pub fn stop(&self) {
        self.sink.stop();
    }

    pub fn empty(&self) -> bool {
        self.sink.empty()
    }

    pub fn len(&self) -> usize {
        self.sink.len()
    }
}

/// xsdk's frame: `samples` of each of `channels` channels, a channel after
/// another (xsdk `audio.h`).
pub const XSDK_CHANNELS: usize = 6;
pub const XSDK_FRAME_SAMPLES: usize = 256;

/// Fills one xsdk frame from the interleaved stereo mix: left and right on
/// the front pair, silence on the rest. A mixer with nothing playing runs
/// dry; that is silence too.
pub fn fill_frame(frame: &mut [f32], stereo: &mut impl Iterator<Item = f32>) {
    let (front, rest) = frame.split_at_mut(2 * XSDK_FRAME_SAMPLES);
    let (left, right) = front.split_at_mut(XSDK_FRAME_SAMPLES);
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        *l = stereo.next().unwrap_or(0.0);
        *r = stereo.next().unwrap_or(0.0);
    }
    rest.fill(0.0);
}

#[cfg(target_os = "xbox360")]
mod xsdk {
    use std::sync::Arc;

    use rodio::dynamic_mixer::{self, DynamicMixerController};
    use tracing::{info, warn};

    use super::{MIX_CHANNELS, MIX_RATE_HZ, XSDK_CHANNELS, XSDK_FRAME_SAMPLES, fill_frame};

    unsafe extern "C" {
        fn xsdk_audio_open(frames_ahead: u32) -> i32;
        fn xsdk_audio_next_frame(timeout_ms: u32, frame: *mut *mut f32) -> i32;
        fn xsdk_audio_queue_frame();
    }

    const STATUS_IO_TIMEOUT: i32 = 0xC000_00B5_u32 as i32;
    /// A wait for a free slot gives up after this, to try again.
    const WAIT_MS: u32 = 100;

    /// Opens the driver and starts the thread that feeds it the mix.
    pub fn open() -> Option<Arc<DynamicMixerController<f32>>> {
        let status = unsafe { xsdk_audio_open(0) };
        if status < 0 {
            warn!("xsdk_audio_open failed: {:#010x}", status as u32);
            return None;
        }
        let (controller, mut mixer) = dynamic_mixer::mixer::<f32>(MIX_CHANNELS, MIX_RATE_HZ);
        std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || {
                loop {
                    let mut frame = std::ptr::null_mut();
                    match unsafe { xsdk_audio_next_frame(WAIT_MS, &mut frame) } {
                        STATUS_IO_TIMEOUT => continue,
                        status if status < 0 => {
                            info!("audio feeder done: {:#010x}", status as u32);
                            return;
                        }
                        _ => {}
                    }
                    let frame = unsafe {
                        std::slice::from_raw_parts_mut(frame, XSDK_CHANNELS * XSDK_FRAME_SAMPLES)
                    };
                    fill_frame(frame, &mut mixer);
                    unsafe { xsdk_audio_queue_frame() };
                }
            })
            .ok()?;
        info!("audio: xsdk, {MIX_CHANNELS} channels at {MIX_RATE_HZ} Hz");
        Some(controller)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::buffer::SamplesBuffer;
    use rodio::dynamic_mixer;

    #[test]
    fn a_frame_takes_left_and_right_onto_the_front_pair() {
        let mut stereo = (0..).map(|i| i as f32);
        let mut frame = vec![9.0; XSDK_CHANNELS * XSDK_FRAME_SAMPLES];
        fill_frame(&mut frame, &mut stereo);
        assert_eq!(&frame[..3], &[0.0, 2.0, 4.0]);
        assert_eq!(
            &frame[XSDK_FRAME_SAMPLES..XSDK_FRAME_SAMPLES + 3],
            &[1.0, 3.0, 5.0]
        );
        assert!(frame[2 * XSDK_FRAME_SAMPLES..].iter().all(|&s| s == 0.0));
        assert_eq!(stereo.next(), Some(512.0));
    }

    #[test]
    fn a_dry_mix_is_silence() {
        let mut frame = vec![9.0; XSDK_CHANNELS * XSDK_FRAME_SAMPLES];
        fill_frame(&mut frame, &mut std::iter::empty());
        assert!(frame.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn mixer_sinks_play_into_the_mix_and_finish() {
        let (controller, mut mix) = dynamic_mixer::mixer::<f32>(MIX_CHANNELS, MIX_RATE_HZ);
        let output = Output::mixer(controller);
        let sink = output.sink();
        sink.append(SamplesBuffer::new(2, MIX_RATE_HZ, vec![0.5_f32; 480]));
        let spatial = output.spatial_sink([0.0, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]);
        spatial.append(SamplesBuffer::new(1, MIX_RATE_HZ, vec![0.25_f32; 240]));

        let heard: Vec<f32> = (&mut mix).take(480).collect();
        assert!(
            heard.iter().any(|&s| s > 0.5),
            "both sinks mixed: {heard:?}"
        );
        // Drained past the end: both sinks are done, as a device's would be.
        for _ in 0..4800 {
            mix.next();
        }
        assert!(sink.empty() && spatial.empty());
    }
}
