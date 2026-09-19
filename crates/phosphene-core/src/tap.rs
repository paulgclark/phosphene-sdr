// SPDX-License-Identifier: MIT

//! The two read-only taps into v1 (nextgen spec §2.2) and their supporting
//! types — the canonical, frozen home (D-017).
//!
//! ## ⚠ Frozen interfaces (D-017)
//!
//! These signatures are the single canonical definition of the tap contract:
//! the D-017 swap has landed, and `phosphene-analyze`'s `tap.rs` is now a
//! `pub use` re-export of this module (the temporary local copy, and the
//! parity test that held the two identical, are gone). The freeze still
//! binds: the next-gen track compiles against these exact signatures, so do
//! not "improve" anything here.
//!
//! Both taps are lock-free, read-only, single-producer/single-consumer views
//! over data v1's compute path already holds: `SpectrumTap` exposes the most
//! recent dB power frames the display computes (feeds detection and every
//! FR-AM measurement), and `AnalysisTap` exposes the raw pre-FFT IQ ring
//! (feeds classification only, phase A1+; ring depth per NFR-A2).
//!
//! No implementation lives here yet: v1's IQ ring (NFR-P3) is wired app-side
//! and the display keeps no frame history this tap could read for free, so
//! per D-017 this lane freezes the signatures and defers the wiring to the
//! live-integration lane.

use std::fmt;

use crate::Complex;

/// Band-level metadata for the frames a [`SpectrumTap`] serves
/// (spec §2.2: "center, span, bin_hz, frame_dt, cal ref").
#[derive(Debug, Clone, PartialEq)]
pub struct BandMeta {
    /// Absolute center (tuned) frequency in Hz, if known. `None` means the
    /// source has no tune metadata and all frequencies are offsets from an
    /// unknown center (v1 clarification C5: absent, never invented).
    pub center_freq_hz: Option<f64>,
    /// Displayed span in Hz — `bins × bin_hz`, i.e. the sample rate for the
    /// non-overlapping full-span frames of v1 §7.1.
    pub span_hz: f64,
    /// Number of frequency bins per frame (the FFT size).
    pub bins: usize,
    /// Width of one bin in Hz.
    pub bin_hz: f64,
    /// Seconds between successive frames.
    pub frame_dt_s: f64,
    /// Calibration reference: dB to add to a frame's raw values to place them
    /// on v1's calibrated scale (§7.2's `C_cal`). `0.0` means frames are
    /// already dBFS/bin per the D-006 convention.
    pub cal_offset_db: f32,
}

impl BandMeta {
    /// Frequency offset (Hz, relative to band center) of a bin position on
    /// the DC-centered axis. Fractional `bin` supported for centroids; DC is
    /// at `bins / 2` (the `phosphene-core` fftshift convention).
    pub fn bin_offset_hz(&self, bin: f64) -> f64 {
        (bin - (self.bins / 2) as f64) * self.bin_hz
    }

    /// Absolute frequency (Hz) of a bin position, when the band center is
    /// known.
    pub fn bin_absolute_hz(&self, bin: f64) -> Option<f64> {
        self.center_freq_hz.map(|c| c + self.bin_offset_hz(bin))
    }
}

/// Caller-allocated destination for [`SpectrumTap::latest_frames`]: the most
/// recent up-to-`max_frames` dB power frames, oldest first.
///
/// The tap fills it with [`clear`](Self::clear) +
/// [`push`](Self::push); readers use [`frame`](Self::frame) /
/// [`seq`](Self::seq). Sequence numbers are the producer's monotonic frame
/// counter, so a polling consumer can tell which frames it has already seen.
#[derive(Debug, Clone)]
pub struct FrameView {
    bins: usize,
    max_frames: usize,
    /// `len × bins`, row-major, oldest frame first.
    data: Vec<f32>,
    len: usize,
    /// Sequence number of the newest held frame; meaningless while empty.
    newest_seq: u64,
}

impl FrameView {
    /// Allocate a view holding up to `max_frames` frames of `bins` bins each.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    pub fn new(bins: usize, max_frames: usize) -> Self {
        assert!(bins > 0, "FrameView needs at least one bin");
        assert!(
            max_frames > 0,
            "FrameView needs room for at least one frame"
        );
        FrameView {
            bins,
            max_frames,
            data: vec![0.0; bins * max_frames],
            len: 0,
            newest_seq: 0,
        }
    }

    /// Bins per frame.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Capacity in frames.
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Number of valid frames currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no frames are held.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Frame `i`, where `0` is the oldest held frame and `len() - 1` the
    /// newest.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub fn frame(&self, i: usize) -> &[f32] {
        assert!(
            i < self.len,
            "frame index {i} out of range (len {})",
            self.len
        );
        &self.data[i * self.bins..(i + 1) * self.bins]
    }

    /// Producer sequence number of frame `i` (same indexing as
    /// [`frame`](Self::frame)). Held frames are consecutive, so
    /// `seq(i) = seq(len-1) - (len-1-i)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub fn seq(&self, i: usize) -> u64 {
        assert!(
            i < self.len,
            "frame index {i} out of range (len {})",
            self.len
        );
        self.newest_seq - (self.len - 1 - i) as u64
    }

    /// Drop all held frames (tap side: start of a fill).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append one frame with its producer sequence number (tap side: fill
    /// oldest first). Frames must be pushed in consecutive `seq` order.
    ///
    /// # Panics
    ///
    /// Panics if the view is full, `frame.len() != bins()`, or `seq` is not
    /// exactly one past the previously pushed frame's.
    pub fn push(&mut self, frame: &[f32], seq: u64) {
        assert!(
            self.len < self.max_frames,
            "FrameView is full — clear() between fills"
        );
        assert_eq!(frame.len(), self.bins, "frame length must equal bins()");
        if self.len > 0 {
            assert_eq!(
                seq,
                self.newest_seq + 1,
                "frames must be pushed in consecutive sequence order"
            );
        }
        let start = self.len * self.bins;
        self.data[start..start + self.bins].copy_from_slice(frame);
        self.newest_seq = seq;
        self.len += 1;
    }
}

/// A span of recent history requested from an [`AnalysisTap`]: the most
/// recent `duration_s` seconds of raw IQ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSpan {
    /// Length of the requested span in seconds, ending now (the newest
    /// sample the tap holds).
    pub duration_s: f64,
}

impl TimeSpan {
    /// The most recent `duration_s` seconds.
    pub fn last(duration_s: f64) -> Self {
        TimeSpan { duration_s }
    }
}

/// Caller-allocated destination for [`AnalysisTap::snapshot`]: raw complex
/// baseband samples, oldest first.
#[derive(Debug, Clone, Default)]
pub struct IqBuf {
    /// The snapshot samples (`cf32`, full-scale-normalised per v1 §8.2).
    pub samples: Vec<Complex<f32>>,
}

/// What an [`AnalysisTap::snapshot`] actually delivered.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureMeta {
    /// Sample rate of the delivered samples, Hz.
    pub sample_rate_hz: f64,
    /// Absolute center frequency the samples were captured at, if known.
    pub center_freq_hz: Option<f64>,
    /// Duration actually delivered, seconds (`samples / rate`).
    pub duration_s: f64,
}

/// Static capabilities of an [`AnalysisTap`]
/// (spec §2.2: "rate, available history depth, center").
#[derive(Debug, Clone, PartialEq)]
pub struct TapCaps {
    /// Sample rate of the underlying IQ ring, Hz.
    pub sample_rate_hz: f64,
    /// Maximum history depth the ring can hold, seconds (NFR-A2: default
    /// target ~250 ms–1 s at max rate). A fresh ring may hold less *yet*;
    /// [`AnalysisTap::snapshot`] reports actual availability.
    pub history_s: f64,
    /// Absolute center frequency of the ring's samples, if known.
    pub center_freq_hz: Option<f64>,
}

/// Errors from tap reads.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TapError {
    /// The requested [`TimeSpan`] exceeds what the ring currently holds.
    InsufficientHistory {
        /// Seconds requested.
        requested_s: f64,
        /// Seconds actually available.
        available_s: f64,
    },
    /// The requested span is not representable (non-finite or non-positive).
    InvalidSpan {
        /// The offending duration, seconds.
        requested_s: f64,
    },
}

impl fmt::Display for TapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TapError::InsufficientHistory {
                requested_s,
                available_s,
            } => write!(
                f,
                "requested {requested_s} s of IQ history but only {available_s} s is available"
            ),
            TapError::InvalidSpan { requested_s } => {
                write!(f, "invalid IQ history span: {requested_s} s")
            }
        }
    }
}

impl std::error::Error for TapError {}

/// Read-only view of the most recent dB power frames the display computes —
/// feeds detection and all inspector measurements (spec §2.2; no IQ, cheap).
pub trait SpectrumTap {
    /// Fill `out` with up to `out.max_frames()` of the most recent frames,
    /// oldest first. Fewer (or none) are delivered if the producer has not
    /// yet made that many.
    fn latest_frames(&self, out: &mut FrameView);

    /// Metadata for the frames: center, span, bin width, frame cadence,
    /// calibration reference.
    fn meta(&self) -> BandMeta;
}

/// Read-only view of the raw pre-FFT IQ ring — feeds classification only
/// (spec §2.2; phase A1+).
pub trait AnalysisTap {
    /// Copy the requested span of recent raw `cf32` into `out` (oldest
    /// first), returning what was delivered.
    fn snapshot(&self, span: TimeSpan, out: &mut IqBuf) -> Result<CaptureMeta, TapError>;

    /// Rate, maximum history depth, and center of the ring.
    fn caps(&self) -> TapCaps;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_meta_maps_bins_to_offsets_dc_centered() {
        let meta = BandMeta {
            center_freq_hz: Some(100_000_000.0),
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        };
        assert_eq!(meta.bin_offset_hz(512.0), 0.0); // DC at bins/2
        assert_eq!(meta.bin_offset_hz(812.0), 300_000.0);
        assert_eq!(meta.bin_offset_hz(0.0), -512_000.0);
        assert_eq!(meta.bin_absolute_hz(812.0), Some(100_300_000.0));
    }

    #[test]
    fn band_meta_without_center_never_invents_absolute_frequencies() {
        // v1 clarification C5: absent, never invented. Offsets still work.
        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        };
        assert_eq!(meta.bin_offset_hz(512.0), 0.0);
        assert_eq!(meta.bin_offset_hz(212.0), -300_000.0);
        assert_eq!(meta.bin_absolute_hz(212.0), None);
    }

    #[test]
    fn time_span_last_is_the_requested_duration() {
        assert_eq!(TimeSpan::last(0.25), TimeSpan { duration_s: 0.25 });
    }

    #[test]
    fn frame_view_holds_consecutive_frames_oldest_first() {
        let mut view = FrameView::new(4, 3);
        assert!(view.is_empty());
        view.push(&[0.0; 4], 10);
        view.push(&[1.0; 4], 11);
        assert_eq!(view.len(), 2);
        assert_eq!(view.frame(0), &[0.0; 4]);
        assert_eq!(view.frame(1), &[1.0; 4]);
        assert_eq!(view.seq(0), 10);
        assert_eq!(view.seq(1), 11);
        view.clear();
        assert!(view.is_empty());
        view.push(&[2.0; 4], 42);
        assert_eq!(view.seq(0), 42);
    }

    #[test]
    fn seq_lets_a_polling_consumer_skip_frames_already_seen() {
        // Two overlapping fills, as a polling consumer would experience them:
        // the producer made frames 7..=9, then 8..=11 by the next poll.
        let mut view = FrameView::new(2, 4);
        for seq in 7..=9 {
            view.push(&[seq as f32; 2], seq);
        }
        let mut last_seen = view.seq(view.len() - 1); // consumer caught up: 9

        view.clear();
        for seq in 8..=11 {
            view.push(&[seq as f32; 2], seq);
        }
        let fresh: Vec<u64> = (0..view.len())
            .map(|i| view.seq(i))
            .filter(|&s| s > last_seen)
            .collect();
        assert_eq!(fresh, vec![10, 11], "only unseen frames are fresh");
        last_seen = view.seq(view.len() - 1);
        assert_eq!(last_seen, 11);
    }

    #[test]
    #[should_panic(expected = "consecutive")]
    fn frame_view_rejects_sequence_gaps() {
        let mut view = FrameView::new(2, 2);
        view.push(&[0.0; 2], 5);
        view.push(&[0.0; 2], 7);
    }

    #[test]
    fn tap_error_messages_carry_the_numbers() {
        let msg = TapError::InsufficientHistory {
            requested_s: 1.0,
            available_s: 0.25,
        }
        .to_string();
        assert!(msg.contains('1'), "unhelpful: {msg}");
        assert!(msg.contains("0.25"), "unhelpful: {msg}");
    }

    /// Minimal implementations proving both traits are implementable (and
    /// object-safe) exactly as frozen.
    struct StubTaps;

    impl SpectrumTap for StubTaps {
        fn latest_frames(&self, out: &mut FrameView) {
            out.clear();
            out.push(&vec![-100.0; out.bins()], 1);
        }

        fn meta(&self) -> BandMeta {
            BandMeta {
                center_freq_hz: None,
                span_hz: 8.0,
                bins: 8,
                bin_hz: 1.0,
                frame_dt_s: 1.0,
                cal_offset_db: 0.0,
            }
        }
    }

    impl AnalysisTap for StubTaps {
        fn snapshot(&self, span: TimeSpan, out: &mut IqBuf) -> Result<CaptureMeta, TapError> {
            out.samples.clear();
            Err(TapError::InsufficientHistory {
                requested_s: span.duration_s,
                available_s: 0.0,
            })
        }

        fn caps(&self) -> TapCaps {
            TapCaps {
                sample_rate_hz: 8.0,
                history_s: 0.25,
                center_freq_hz: None,
            }
        }
    }

    #[test]
    fn traits_are_object_safe_and_implementable_as_frozen() {
        let spectrum: &dyn SpectrumTap = &StubTaps;
        let mut view = FrameView::new(8, 2);
        spectrum.latest_frames(&mut view);
        assert_eq!(view.len(), 1);
        assert_eq!(spectrum.meta().bins, 8);

        let analysis: &dyn AnalysisTap = &StubTaps;
        let mut buf = IqBuf::default();
        let err = analysis
            .snapshot(TimeSpan::last(1.0), &mut buf)
            .unwrap_err();
        assert_eq!(
            err,
            TapError::InsufficientHistory {
                requested_s: 1.0,
                available_s: 0.0,
            }
        );
        assert_eq!(analysis.caps().history_s, 0.25);
    }
}
