// SPDX-License-Identifier: MIT

//! The file/siggen-backed offline test harness (nextgen spec §2.3).
//!
//! `phosphene-analyze` develops against recorded IQ and the M0 signal
//! generator — whose ground truth is known — not against live v1; the taps
//! are only needed for live operation. [`OfflineTap`] is that development
//! path: feed it `cf32` samples from any source (a `phosphene-sources`
//! `SigGen` fill, a `.cfile` loaded with [`load_cf32`]), and it runs them
//! through
//! `phosphene-core`'s calibrated analyzer and serves the results back
//! through **both** tap traits — [`SpectrumTap`] (dB power frames) and
//! [`AnalysisTap`] (the raw IQ history) — exactly as live v1 will.
//!
//! Every later lane in this crate tests through this harness, so detector,
//! tracker, measurement and classifier code is identical between offline
//! tests and live integration (NFR-A4 determinism: same input, same config,
//! same output).

use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::io::{self, Read as _, Write as _};
use std::path::Path;

use phosphene_core::{Complex, CoreError, SpectrumAnalyzer, WindowKind};

use crate::tap::{
    AnalysisTap, BandMeta, CaptureMeta, FrameView, IqBuf, SpectrumTap, TapCaps, TapError, TimeSpan,
};

/// Configuration for an [`OfflineTap`].
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineTapConfig {
    /// Sample rate of the fed IQ, Hz.
    pub sample_rate_hz: f64,
    /// Absolute center frequency of the fed IQ, if known.
    pub center_freq_hz: Option<f64>,
    /// FFT size / bins per frame (a `phosphene-core` supported size).
    pub fft_size: usize,
    /// Analysis window.
    pub window: WindowKind,
    /// How many recent spectrum frames the tap retains.
    pub spectrum_history_frames: usize,
    /// How much raw IQ history the tap retains, seconds (NFR-A2: the live
    /// ring targets ~250 ms–1 s at max rate).
    pub iq_history_s: f64,
}

impl OfflineTapConfig {
    /// Defaults matching the live targets: 1024-bin Hann frames, 64 frames
    /// of spectrum history, 250 ms of IQ history.
    pub fn new(sample_rate_hz: f64) -> Self {
        OfflineTapConfig {
            sample_rate_hz,
            center_freq_hz: None,
            fft_size: phosphene_core::DEFAULT_FFT_SIZE,
            window: WindowKind::Hann,
            spectrum_history_frames: 64,
            iq_history_s: 0.25,
        }
    }
}

/// Errors from building an [`OfflineTap`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessError {
    /// The configuration was rejected; the message names the field and why.
    InvalidConfig(String),
    /// The FFT size was rejected by `phosphene-core`.
    Core(CoreError),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HarnessError::InvalidConfig(msg) => {
                write!(f, "invalid harness configuration: {msg}")
            }
            HarnessError::Core(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HarnessError {}

impl From<CoreError> for HarnessError {
    fn from(e: CoreError) -> Self {
        HarnessError::Core(e)
    }
}

/// Offline implementation of both taps over fed IQ. See the [module
/// docs](self).
pub struct OfflineTap {
    analyzer: SpectrumAnalyzer,
    sample_rate_hz: f64,
    center_freq_hz: Option<f64>,
    /// Samples awaiting a full frame.
    pending: Vec<Complex<f32>>,
    /// Most recent dB frames, oldest first.
    frames: VecDeque<Vec<f32>>,
    max_frames: usize,
    /// Sequence number the next produced frame will get.
    next_seq: u64,
    /// Raw IQ history, oldest first, bounded to `max_iq` samples.
    iq: VecDeque<Complex<f32>>,
    max_iq: usize,
    /// Configured history depth, seconds (reported via [`TapCaps`]).
    iq_history_s: f64,
    /// Scratch for one dB frame.
    scratch: Vec<f32>,
}

impl fmt::Debug for OfflineTap {
    // Manual: `SpectrumAnalyzer` (a dyn-FFT holder) has no Debug.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OfflineTap")
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("center_freq_hz", &self.center_freq_hz)
            .field("fft_size", &self.analyzer.size())
            .field("frames_held", &self.frames.len())
            .field("iq_held", &self.iq.len())
            .field("next_seq", &self.next_seq)
            .finish_non_exhaustive()
    }
}

impl OfflineTap {
    /// Build a harness tap; validates the configuration up front.
    pub fn new(config: OfflineTapConfig) -> Result<Self, HarnessError> {
        let rate = config.sample_rate_hz;
        if !rate.is_finite() || rate <= 0.0 {
            return Err(HarnessError::InvalidConfig(format!(
                "sample rate must be finite and positive, got {rate} Hz"
            )));
        }
        if config.spectrum_history_frames == 0 {
            return Err(HarnessError::InvalidConfig(
                "spectrum history must be at least one frame".to_string(),
            ));
        }
        if !config.iq_history_s.is_finite() || config.iq_history_s <= 0.0 {
            return Err(HarnessError::InvalidConfig(format!(
                "IQ history must be finite and positive, got {} s",
                config.iq_history_s
            )));
        }
        let analyzer = SpectrumAnalyzer::new(config.fft_size, config.window)?;
        let max_iq = (config.iq_history_s * rate).round() as usize;
        if max_iq == 0 {
            // A zero-capacity ring must be refused, never silently degraded:
            // the AnalysisTap contract is a BOUNDED ring (spec §2.2, NFR-A2),
            // and the eviction logic below is only bounded for max_iq ≥ 1.
            return Err(HarnessError::InvalidConfig(format!(
                "IQ history of {} s rounds to zero retained samples at {rate} Hz \
                 — increase the duration to at least one sample period",
                config.iq_history_s
            )));
        }
        Ok(OfflineTap {
            scratch: vec![0.0; analyzer.size()],
            analyzer,
            sample_rate_hz: rate,
            center_freq_hz: config.center_freq_hz,
            pending: Vec::new(),
            frames: VecDeque::with_capacity(config.spectrum_history_frames),
            max_frames: config.spectrum_history_frames,
            next_seq: 0,
            iq: VecDeque::with_capacity(max_iq),
            max_iq,
            iq_history_s: config.iq_history_s,
        })
    }

    /// Feed a block of `cf32` samples, producing one dB frame per complete
    /// `fft_size` samples (non-overlapping, 100% coverage — v1 §7.1) and
    /// appending to the bounded raw-IQ history. Chunking-invariant: any split
    /// of the same stream produces the same frames.
    pub fn feed(&mut self, samples: &[Complex<f32>]) {
        for &s in samples {
            if self.iq.len() == self.max_iq {
                self.iq.pop_front();
            }
            self.iq.push_back(s);

            self.pending.push(s);
            if self.pending.len() == self.analyzer.size() {
                self.analyzer.process(&self.pending, &mut self.scratch);
                self.pending.clear();
                if self.frames.len() == self.max_frames {
                    self.frames.pop_front();
                }
                self.frames.push_back(self.scratch.clone());
                self.next_seq += 1;
            }
        }
    }

    /// Total frames produced so far (the next frame's sequence number).
    pub fn frames_produced(&self) -> u64 {
        self.next_seq
    }
}

impl SpectrumTap for OfflineTap {
    fn latest_frames(&self, out: &mut FrameView) {
        out.clear();
        let take = out.max_frames().min(self.frames.len());
        let newest = self.next_seq; // frames held end at next_seq - 1
        let start = self.frames.len() - take;
        for (i, frame) in self.frames.iter().skip(start).enumerate() {
            let seq = newest - take as u64 + i as u64;
            out.push(frame, seq);
        }
    }

    fn meta(&self) -> BandMeta {
        let bins = self.analyzer.size();
        BandMeta {
            center_freq_hz: self.center_freq_hz,
            span_hz: self.sample_rate_hz,
            bins,
            bin_hz: self.sample_rate_hz / bins as f64,
            frame_dt_s: bins as f64 / self.sample_rate_hz,
            cal_offset_db: 0.0,
        }
    }
}

impl AnalysisTap for OfflineTap {
    fn snapshot(&self, span: TimeSpan, out: &mut IqBuf) -> Result<CaptureMeta, TapError> {
        if !span.duration_s.is_finite() || span.duration_s <= 0.0 {
            return Err(TapError::InvalidSpan {
                requested_s: span.duration_s,
            });
        }
        let n = (span.duration_s * self.sample_rate_hz).round() as usize;
        if n > self.iq.len() {
            return Err(TapError::InsufficientHistory {
                requested_s: span.duration_s,
                available_s: self.iq.len() as f64 / self.sample_rate_hz,
            });
        }
        out.samples.clear();
        out.samples.extend(self.iq.iter().skip(self.iq.len() - n));
        Ok(CaptureMeta {
            sample_rate_hz: self.sample_rate_hz,
            center_freq_hz: self.center_freq_hz,
            duration_s: n as f64 / self.sample_rate_hz,
        })
    }

    fn caps(&self) -> TapCaps {
        TapCaps {
            sample_rate_hz: self.sample_rate_hz,
            history_s: self.iq_history_s,
            center_freq_hz: self.center_freq_hz,
        }
    }
}

/// Load a raw interleaved little-endian `cf32` IQ file (the `.cfile`
/// convention: `I0 Q0 I1 Q1 …`, 4-byte floats, no header).
///
/// Fails with [`io::ErrorKind::InvalidData`] if the file length is not a
/// multiple of 8 bytes.
pub fn load_cf32(path: &Path) -> io::Result<Vec<Complex<f32>>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: length {} is not a whole number of cf32 samples (8 bytes each)",
                path.display(),
                bytes.len()
            ),
        ));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            Complex::new(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            )
        })
        .collect())
}

/// Write samples as a raw interleaved little-endian `cf32` IQ file — the
/// inverse of [`load_cf32`]; used to author test fixtures from the
/// generator.
pub fn write_cf32(path: &Path, samples: &[Complex<f32>]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(samples.len() * 8);
    for s in samples {
        bytes.extend_from_slice(&s.re.to_le_bytes());
        bytes.extend_from_slice(&s.im.to_le_bytes());
    }
    fs::File::create(path)?.write_all(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_configs_with_named_fields() {
        let mut c = OfflineTapConfig::new(0.0);
        let HarnessError::InvalidConfig(msg) = OfflineTap::new(c.clone()).unwrap_err() else {
            panic!("wrong error kind");
        };
        assert!(msg.contains("sample rate"), "unhelpful: {msg}");

        c.sample_rate_hz = 1_000_000.0;
        c.spectrum_history_frames = 0;
        assert!(matches!(
            OfflineTap::new(c.clone()),
            Err(HarnessError::InvalidConfig(_))
        ));

        c.spectrum_history_frames = 4;
        c.iq_history_s = -1.0;
        assert!(matches!(
            OfflineTap::new(c.clone()),
            Err(HarnessError::InvalidConfig(_))
        ));

        c.iq_history_s = 0.25;
        c.fft_size = 1000;
        assert!(matches!(
            OfflineTap::new(c),
            Err(HarnessError::Core(CoreError::InvalidFftSize(1000)))
        ));
    }

    #[test]
    fn refuses_iq_history_that_rounds_to_zero_samples() {
        // 0.1 ms at 1 kHz is 0.1 samples — a positive duration that rounds
        // to a zero-capacity ring must be refused, not silently unbounded
        // (spec §2.2 / NFR-A2: AnalysisTap is a bounded ring).
        let mut c = OfflineTapConfig::new(1_000.0);
        c.iq_history_s = 1e-4;
        let HarnessError::InvalidConfig(msg) = OfflineTap::new(c).unwrap_err() else {
            panic!("wrong error kind");
        };
        assert!(msg.contains("0.0001"), "must name the duration: {msg}");
        assert!(msg.contains("1000"), "must name the rate: {msg}");
    }

    #[test]
    fn iq_ring_is_bounded_and_evicts_oldest_first() {
        const RATE: f64 = 1_024_000.0;
        const CAP: usize = 16;
        let mut c = OfflineTapConfig::new(RATE);
        c.iq_history_s = CAP as f64 / RATE; // exactly 16 samples of history
        let mut tap = OfflineTap::new(c).unwrap();

        // Sustained feed of 100 distinguishable samples, in uneven chunks.
        let fed: Vec<Complex<f32>> = (0..100).map(|i| Complex::new(i as f32, 0.0)).collect();
        for chunk in fed.chunks(7) {
            tap.feed(chunk);
        }

        // The full available history is exactly CAP samples — the newest
        // CAP, oldest first; everything earlier was evicted.
        let mut buf = IqBuf::default();
        tap.snapshot(TimeSpan::last(CAP as f64 / RATE), &mut buf)
            .unwrap();
        assert_eq!(buf.samples, fed[100 - CAP..].to_vec());
        assert!(matches!(
            tap.snapshot(TimeSpan::last((CAP + 1) as f64 / RATE), &mut buf),
            Err(TapError::InsufficientHistory { .. })
        ));

        // Feeding more keeps the bound and keeps rolling forward.
        tap.feed(&[Complex::new(100.0, 0.0)]);
        tap.snapshot(TimeSpan::last(CAP as f64 / RATE), &mut buf)
            .unwrap();
        assert_eq!(buf.samples[0], Complex::new(85.0, 0.0));
        assert_eq!(buf.samples[CAP - 1], Complex::new(100.0, 0.0));
    }

    #[test]
    fn snapshot_rejects_invalid_and_oversized_spans() {
        let mut c = OfflineTapConfig::new(1_024_000.0);
        c.iq_history_s = 0.01;
        let mut tap = OfflineTap::new(c).unwrap();
        tap.feed(&vec![Complex::new(0.0, 0.0); 1024]);

        let mut buf = IqBuf::default();
        assert!(matches!(
            tap.snapshot(TimeSpan::last(-1.0), &mut buf),
            Err(TapError::InvalidSpan { .. })
        ));
        assert!(matches!(
            tap.snapshot(TimeSpan::last(0.5), &mut buf),
            Err(TapError::InsufficientHistory { .. })
        ));
        let meta = tap.snapshot(TimeSpan::last(0.0005), &mut buf).unwrap();
        assert_eq!(buf.samples.len(), 512);
        assert!((meta.duration_s - 0.0005).abs() < 1e-9);
    }
}
