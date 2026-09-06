//! In-process audio capture.
//!
//! Capture runs on a dedicated thread that owns the `cpal` stream (CoreAudio
//! on macOS, the PipeWire ALSA shim on Linux desktop targets), converts the
//! input to mono `f32`, and appends to a shared buffer. WAV slices are cut
//! from that continuous buffer, so segmented dictation loses no audio at
//! chunk boundaries.

use std::{
    path::Path,
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
    time::Duration,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use thiserror::Error;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("no audio input device is available")]
    NoInputDevice,
    #[error("failed to query the input device: {0}")]
    Device(String),
    #[error("failed to build the capture stream: {0}")]
    BuildStream(String),
    #[error("failed to start the capture stream: {0}")]
    Play(String),
    #[error("capture thread failed: {0}")]
    CaptureThread(String),
    #[error("capture thread panicked")]
    ThreadPanicked,
    #[error("failed to write WAV file: {0}")]
    Wav(#[from] hound::Error),
    #[error("failed to create the audio file: {0}")]
    Io(#[from] std::io::Error),
}

/// A running capture. Cloning is deliberately unsupported: one dictation
/// session owns one capture.
pub struct AudioCapture {
    /// Mono f32 samples at the device's native rate.
    buffer: Arc<Mutex<Vec<f32>>>,
    device_rate: u32,
    stop_tx: mpsc::Sender<()>,
    capture_thread: Option<JoinHandle<Result<(), AudioError>>>,
    consumer_thread: Option<JoinHandle<()>>,
}

impl AudioCapture {
    /// Build a capture that synthesizes `duration` of silence without
    /// touching an audio device. Used by daemon HTTP tests.
    pub(crate) fn synthetic_silence(duration: Duration, rate: u32) -> Self {
        let (stop_tx, _stop_rx) = mpsc::channel();
        let samples = vec![0.0f32; (duration.as_secs_f64() * f64::from(rate)) as usize];
        Self {
            buffer: Arc::new(Mutex::new(samples)),
            device_rate: rate,
            stop_tx,
            capture_thread: None,
            consumer_thread: None,
        }
    }
}

impl AudioCapture {
    pub fn start() -> Result<Self, AudioError> {
        let (sample_tx, sample_rx) = mpsc::sync_channel::<Vec<f32>>(64);
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32, AudioError>>();
        let buffer = Arc::new(Mutex::new(Vec::new()));

        let capture_thread =
            std::thread::spawn(move || run_capture_stream(sample_tx, stop_rx, ready_tx));

        let device_rate = ready_rx.recv().map_err(|_| AudioError::ThreadPanicked)??;

        let consumer_buffer = Arc::clone(&buffer);
        let consumer_thread = std::thread::spawn(move || {
            while let Ok(block) = sample_rx.recv() {
                let mut guard = consumer_buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.extend_from_slice(&block);
            }
        });

        Ok(Self {
            buffer,
            device_rate,
            stop_tx,
            capture_thread: Some(capture_thread),
            consumer_thread: Some(consumer_thread),
        })
    }

    /// Seconds of audio captured so far.
    pub fn elapsed_secs(&self) -> f64 {
        let len = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        len as f64 / f64::from(self.device_rate)
    }

    /// Write `[start_secs, end_secs]` of the capture to `path` as a 16 kHz
    /// mono s16 WAV. `end_secs` is clamped to what has been captured.
    pub fn cut_wav(&self, path: &Path, start_secs: f64, end_secs: f64) -> Result<(), AudioError> {
        let rate = self.device_rate;
        let samples = {
            let guard = self
                .buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let start = ((start_secs.max(0.0) * f64::from(rate)) as usize).min(guard.len());
            let end = ((end_secs.max(0.0) * f64::from(rate)) as usize).min(guard.len());
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            guard[lo..hi].to_vec()
        };
        let resampled = resample_linear(&samples, rate, TARGET_SAMPLE_RATE);
        write_wav(path, &resampled)
    }

    /// Stop capture and return all audio as 16 kHz mono f32.
    pub fn stop(mut self) -> Result<Vec<f32>, AudioError> {
        let _ = self.stop_tx.send(());
        if let Some(thread) = self.capture_thread.take() {
            thread.join().map_err(|_| AudioError::ThreadPanicked)??;
        }
        if let Some(thread) = self.consumer_thread.take() {
            thread.join().map_err(|_| AudioError::ThreadPanicked)?;
        }
        let buffer = std::mem::take(
            &mut *self
                .buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        Ok(resample_linear(
            &buffer,
            self.device_rate,
            TARGET_SAMPLE_RATE,
        ))
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(thread) = self.capture_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.consumer_thread.take() {
            let _ = thread.join();
        }
    }
}

fn run_capture_stream(
    sample_tx: mpsc::SyncSender<Vec<f32>>,
    stop_rx: mpsc::Receiver<()>,
    ready_tx: mpsc::Sender<Result<u32, AudioError>>,
) -> Result<(), AudioError> {
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(device) => device,
        None => {
            let _ = ready_tx.send(Err(AudioError::NoInputDevice));
            return Err(AudioError::NoInputDevice);
        }
    };
    let supported = match device.default_input_config() {
        Ok(config) => config,
        Err(error) => {
            let _ = ready_tx.send(Err(AudioError::Device(error.to_string())));
            return Err(AudioError::Device(error.to_string()));
        }
    };

    let device_rate = supported.sample_rate();
    let channels = usize::from(supported.channels());
    let config = supported.config();
    let error_callback = |error| {
        tracing::warn!(%error, "audio capture stream reported an error");
    };

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| forward_mono(data, channels, 1.0, &sample_tx, |v| v),
            error_callback,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                forward_mono(data, channels, 1.0 / 32768.0, &sample_tx, f32::from);
            },
            error_callback,
            None,
        ),
        format => {
            let _ = ready_tx.send(Err(AudioError::BuildStream(format!(
                "unsupported input sample format {format:?}"
            ))));
            return Err(AudioError::BuildStream(format!(
                "unsupported input sample format {format:?}"
            )));
        }
    }
    .map_err(|error| AudioError::BuildStream(error.to_string()))?;

    stream
        .play()
        .map_err(|error| AudioError::Play(error.to_string()))?;
    if ready_tx.send(Ok(device_rate)).is_err() {
        return Err(AudioError::ThreadPanicked);
    }

    // Park until stopped; dropping the stream ends capture.
    let _ = stop_rx.recv();
    drop(stream);
    Ok(())
}

fn forward_mono<T: Copy>(
    data: &[T],
    channels: usize,
    scale: f32,
    sample_tx: &mpsc::SyncSender<Vec<f32>>,
    convert: impl Fn(T) -> f32,
) {
    let channels = channels.max(1);
    let mono: Vec<f32> = data
        .chunks(channels)
        .map(|frame| frame.iter().map(|&v| convert(v)).sum::<f32>() / frame.len() as f32 * scale)
        .collect();
    // If the consumer is gone the capture is being torn down; dropping the
    // block is correct.
    let _ = sample_tx.try_send(mono);
}

/// Linear-interpolation resampler. Adequate for ASR dictation audio and
/// dependency-free; rates are integer ratios in practice (48k/44.1k -> 16k).
pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let out_len = (input.len() as u64 * u64::from(to_rate) / u64::from(from_rate)) as usize;
    (0..out_len)
        .map(|i| {
            let position = i as f64 * f64::from(from_rate) / f64::from(to_rate);
            let index = position.floor() as usize;
            let fraction = (position - index as f64) as f32;
            let a = input.get(index).copied().unwrap_or(0.0);
            let b = input.get(index + 1).copied().unwrap_or(a);
            a + (b - a) * fraction
        })
        .collect()
}

/// Write 16 kHz mono f32 samples as a 16-bit PCM WAV.
pub fn write_wav(path: &Path, samples: &[f32]) -> Result<(), AudioError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_is_identity_when_rates_match() {
        let input = vec![0.1, -0.2, 0.3];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input);
    }

    #[test]
    fn resample_halves_length_when_downsampling_two_to_one() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let output = resample_linear(&input, 32_000, 16_000);
        assert_eq!(output.len(), 50);
        assert!((output[0] - 0.0).abs() < f32::EPSILON);
        assert!((output[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn write_wav_produces_readable_16k_mono_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.wav");
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        write_wav(&path, &samples).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        let read: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(read.len(), samples.len());
        assert_eq!(read[1], (0.5_f32 * 32767.0) as i16);
    }
}
