// SPDX-License-Identifier: MIT

//! The assembled detection pipeline: noise floor → CFAR → blobs (nextgen
//! spec §7.0's first stage; FR-AD1–AD3).
//!
//! Per D-023, the FR-AD1 robust floor **feeds the detector**: each frame is
//! absorbed into the noise estimator first, and the CFAR stage thresholds
//! against that floor (not against instantaneous reference powers), which
//! is what keeps a strong occupant from masking a weak neighbor sitting in
//! its reference window.
//!
//! [`Detector`] consumes the dB power frames a [`crate::tap::SpectrumTap`]
//! serves — push each new frame with its producer sequence number — and
//! yields [`Detection`] regions as they complete, exposing the per-bin
//! noise floor (`N_k`, `σ_k`) and the current frame's occupancy mask along
//! the way. It is pure streaming state over its inputs: no I/O, no clocks,
//! no randomness, so a fixed frame sequence and configuration reproduce the
//! same detections exactly (NFR-A4).
//!
//! SNR figures in the emitted detections rest on the §7.1 floor, which
//! needs a window of history: detections whose lifetime falls inside the
//! first noise window (`noise.window_frames` frames) carry coarse SNR — in
//! the worst case a signal present from frame 0 *is* its bin's entire
//! window and reads near 0 dB until noise-only frames arrive. Downstream
//! confidence handling (FR-AM, NFR-A3) is expected to weigh early figures
//! accordingly.
//!
//! Frames must arrive in increasing sequence order. A **gap** in the
//! sequence (a polling consumer that missed frames) is a time
//! discontinuity: open components are flushed at the gap rather than
//! bridged across it, and their detections are returned by the same
//! [`push_frame`](Detector::push_frame) call.
//!
//! ## DC bin exclusion (D-126/AN-4)
//!
//! A real radio's LO leakage lands in the DC bin (`bins/2`) every frame; CFAR
//! sees a strong, continuous, bin-centered occupant there and detects it like
//! any signal, and blob-building would split it into several. [`Detector`]
//! forces the DC band unoccupied in the mask it hands to blob formation —
//! see [`push_frame`](Detector::push_frame)'s doc for the mechanism and
//! [`DetectorConfig::new`]'s doc for how the band's width was measured, not
//! guessed. This is a detector-only exclusion: the dB frame a display reads
//! is never touched, so the spike is still drawn honestly.

use crate::detect::blob::BlobBuilder;
use crate::detect::cfar::{Cfar, CfarConfig};
use crate::detect::noise::{NoiseFloor, NoiseFloorConfig};
use crate::detect::{DetectError, Detection};

/// Configuration for [`Detector`].
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorConfig {
    /// Noise-floor estimation (§7.1, FR-AD1).
    pub noise: NoiseFloorConfig,
    /// CFAR thresholding (§7.2, FR-AD2).
    pub cfar: CfarConfig,
    /// Minimum occupied cells for a blob to be reported (§7.3; `1` keeps
    /// everything). At Pfa `p` a lone noise cell forms a spurious
    /// single-cell blob about `p × bins` times per frame; a larger minimum
    /// suppresses that clutter at negligible cost to real signals, which
    /// occupy many cells. AN-1/AN-3 (D-126) lean on this harder than before:
    /// with [`crate::track::TrackerConfig::birth_m`] defaulting to `1`, a
    /// blob that clears this size is published on its very first detection,
    /// so this is now the primary false-alarm gate, not the tracker's M-of-N
    /// window.
    pub min_blob_cells: usize,
    /// Bins excluded from detection on each side of the DC bin (`bins/2`,
    /// the [`crate::tap::BandMeta`] DC-centered convention), in addition to
    /// the DC bin itself (D-126/AN-4: a real radio's LO leakage sits in the
    /// centre bin every frame and, undetected, was blob-split into several
    /// spurious signals — see [`Detector`]'s module docs). `0` excludes
    /// only the DC bin.
    pub dc_guard_bins: usize,
}

impl DetectorConfig {
    /// Component defaults.
    ///
    /// `min_blob_cells` defaults to `5` (AN-1/AN-3, D-126): with the
    /// tracker's own `birth_m` defaulting to `1` (a blob confirms on its
    /// first detection), this is the gate standing between a real short
    /// burst and the noise-only single- and few-cell blobs Pfa produces —
    /// see [`CfarConfig::new`]'s doc for the paired `pfa` retune and the
    /// measured evidence both moved together.
    ///
    /// `dc_guard_bins` defaults to `1`, measured (not guessed) against the
    /// built-in generator's DC tone (`ToneConfig { offset_hz: 0.0, .. }`)
    /// from full scale down to −40 dBFS: a periodic-Hann-windowed
    /// bin-centered tone's DFT has exactly three nonzero coefficients — the
    /// DC bin and its immediate neighbours, the ±1 bins at exactly −6.02 dB
    /// relative power — and nothing beyond that ever cleared the noise
    /// floor at any tested level (`seal_tests.rs`'s DC-exclusion evidence).
    pub fn new() -> Self {
        DetectorConfig {
            noise: NoiseFloorConfig::new(),
            cfar: CfarConfig::new(),
            min_blob_cells: 5,
            dc_guard_bins: 1,
        }
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        DetectorConfig::new()
    }
}

/// Streaming detection pipeline. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct Detector {
    noise: NoiseFloor,
    cfar: Cfar,
    blobs: BlobBuilder,
    mask: Vec<bool>,
    /// [`DetectorConfig::dc_guard_bins`], applied to `mask` every frame.
    dc_guard_bins: usize,
    /// Sequence number the next frame must be ≥; `None` before any frame.
    next_seq: Option<u64>,
}

impl Detector {
    /// Build a detector for frames of `bins` bins; validates the whole
    /// configuration up front.
    pub fn new(bins: usize, config: DetectorConfig) -> Result<Self, DetectError> {
        // Checked, not `2 * dc_guard_bins + 1 >= bins`: that multiply
        // overflows (and panics, in a debug build) for a large enough
        // `dc_guard_bins` before it ever gets compared. An overflow here
        // means the requested guard is far larger than any real frame, so
        // it is exactly the "would swallow the whole frame" case.
        let swallows_frame = config
            .dc_guard_bins
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_add(1))
            .is_none_or(|span| span >= bins);
        if swallows_frame {
            return Err(DetectError::InvalidConfig(format!(
                "DC guard of {} bins each side would exclude the whole {bins}-bin frame \
                 from detection",
                config.dc_guard_bins
            )));
        }
        Ok(Detector {
            noise: NoiseFloor::new(bins, config.noise)?,
            cfar: Cfar::new(bins, config.cfar)?,
            blobs: BlobBuilder::new(bins, config.min_blob_cells)?,
            mask: vec![false; bins],
            dc_guard_bins: config.dc_guard_bins,
            next_seq: None,
        })
    }

    /// Process one dB power frame, returning every detection region that
    /// completed at this frame (including regions closed by a sequence
    /// gap, which precede regions that ended normally).
    ///
    /// The DC band (`bins/2` plus [`DetectorConfig::dc_guard_bins`] on each
    /// side, D-126/AN-4) is forced unoccupied in the mask before blob
    /// formation sees it, whatever the CFAR threshold says: LO leakage
    /// there is never detected or blob-split. This is a detection-only
    /// exclusion — `frame` itself (and so the spectrum display) is
    /// untouched, and the noise floor still estimates every bin, DC
    /// included. **A real signal sitting inside the guard band is,
    /// symmetrically, never detected either** — the band is kept to the
    /// smallest width the evidence supports (`DetectorConfig::new`'s doc).
    ///
    /// # Panics
    ///
    /// Panics if `frame.len()` differs from the constructed bin count, or
    /// if `seq` runs backwards (frames must be pushed in increasing
    /// sequence order — the [`crate::tap::FrameView`] contract).
    pub fn push_frame(&mut self, frame: &[f32], seq: u64) -> Vec<Detection> {
        assert_eq!(
            frame.len(),
            self.bins(),
            "frame length must equal the detector's bin count"
        );
        let mut out = Vec::new();
        if let Some(expected) = self.next_seq {
            assert!(
                seq >= expected,
                "frame sequence ran backwards: got {seq} after {}",
                expected - 1
            );
            if seq > expected {
                out.extend(self.blobs.flush());
            }
        }
        self.noise.push_frame(frame);
        self.cfar
            .detect(frame, self.noise.floor_dbfs(), &mut self.mask);
        exclude_dc_band(&mut self.mask, self.dc_guard_bins);
        out.extend(
            self.blobs
                .push_frame(frame, &self.mask, self.noise.floor_dbfs(), seq),
        );
        self.next_seq = Some(seq + 1);
        out
    }

    /// Flush every still-open detection region (end of input). The detector
    /// remains usable; the noise-floor window is kept.
    pub fn finish(&mut self) -> Vec<Detection> {
        self.blobs.flush()
    }

    /// Per-bin noise floor `N_k` in dBFS/bin (FR-AD1). Meaningless before
    /// the first frame.
    pub fn noise_floor(&self) -> &[f32] {
        self.noise.floor_dbfs()
    }

    /// Per-bin noise-floor spread `σ_k` in dB (FR-AD1).
    pub fn noise_spread(&self) -> &[f32] {
        self.noise.spread_db()
    }

    /// The most recent frame's occupancy mask (FR-AD2), with the DC band
    /// (D-126/AN-4, `push_frame`'s doc) forced unoccupied. All-false before
    /// the first frame.
    pub fn last_mask(&self) -> &[bool] {
        &self.mask
    }

    /// Bins per frame this detector was built for.
    pub fn bins(&self) -> usize {
        self.cfar.bins()
    }
}

/// Force `mask` unoccupied across the DC band: the centre bin (`mask.len()
/// / 2`, the [`crate::tap::BandMeta`] DC-centered convention every other
/// bin index in this crate already shares) plus `guard` bins on each side.
/// Applied after CFAR thresholding and before [`BlobBuilder`] sees the
/// mask, so LO leakage there can be neither detected nor blob-split
/// (D-126/AN-4). `Detector::new`'s span check guarantees `2 * guard + 1 <
/// mask.len()`, so the band never swallows the whole frame.
fn exclude_dc_band(mask: &mut [bool], guard: usize) {
    let center = mask.len() / 2;
    let lo = center.saturating_sub(guard);
    let hi = (center + guard).min(mask.len() - 1);
    mask[lo..=hi].fill(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_errors_from_any_component_surface_at_construction() {
        let mut c = DetectorConfig::new();
        c.noise.percentile = 2.0;
        assert!(matches!(
            Detector::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
        let mut c = DetectorConfig::new();
        c.cfar.pfa = 0.0;
        assert!(matches!(
            Detector::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
        let mut c = DetectorConfig::new();
        c.min_blob_cells = 0;
        assert!(matches!(
            Detector::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
    }

    #[test]
    fn dc_guard_that_would_swallow_the_whole_frame_is_rejected() {
        let mut c = DetectorConfig::new();
        c.dc_guard_bins = 32; // 2*32+1 = 65 > 64 bins
        let DetectError::InvalidConfig(msg) = Detector::new(64, c).unwrap_err();
        assert!(msg.contains("DC guard"), "unhelpful: {msg}");
    }

    #[test]
    fn dc_guard_that_overflows_the_span_check_is_rejected_not_panicked() {
        // `2 * dc_guard_bins + 1` overflows `usize` right here — this must
        // return `InvalidConfig`, never panic on the overflow.
        let mut c = DetectorConfig::new();
        c.dc_guard_bins = usize::MAX / 2 + 1;
        let DetectError::InvalidConfig(msg) = Detector::new(64, c).unwrap_err();
        assert!(msg.contains("DC guard"), "unhelpful: {msg}");
    }

    #[test]
    fn dc_band_is_never_reported_occupied_however_strong() {
        // A continuous, full-scale-strong occupant across the whole default
        // guard band (center ± 1 bin, D-126/AN-4) — a worst case well past
        // any real LO leakage — must still never surface in the mask.
        let mut det = Detector::new(64, DetectorConfig::new()).unwrap();
        let center = 32; // 64 / 2
        let mut frame = [-90.0f32; 64];
        frame[center - 1..=center + 1].fill(-10.0);
        det.push_frame(&frame, 0);
        for k in center - 1..=center + 1 {
            assert!(!det.last_mask()[k], "DC bin {k} reported occupied");
        }
    }

    #[test]
    fn a_signal_just_outside_the_default_guard_band_is_still_detected() {
        let mut det = Detector::new(64, DetectorConfig::new()).unwrap();
        let center = 32;
        let mut frame = [-90.0f32; 64];
        frame[center + 2] = -10.0; // one bin outside the default guard
        det.push_frame(&frame, 0);
        assert!(
            det.last_mask()[center + 2],
            "a real signal just outside the DC guard band went undetected"
        );
    }

    #[test]
    fn a_sequence_gap_closes_open_regions_instead_of_bridging() {
        // This test exercises the gap-flushing mechanism, not AN-1/AN-3's
        // blob-size gate (D-126): a single-bin blob is exactly what the
        // shipped `min_blob_cells: 5` default exists to suppress, so use an
        // explicit `1` here to isolate the behavior under test.
        let mut cfg = DetectorConfig::new();
        cfg.min_blob_cells = 1;
        let mut det = Detector::new(64, cfg).unwrap();
        let mut quiet = [-90.0f32; 64];
        // Bin 20: off the default DC guard band around bin 32 (64/2), which
        // this test has nothing to do with — see the DC-exclusion tests
        // below for that behavior.
        quiet[20] = -40.0;
        assert!(det.push_frame(&quiet, 0).is_empty());
        assert!(det.push_frame(&quiet, 1).is_empty());
        // Frames 2..4 were missed; the same strong bin reappears at seq 5.
        let closed = det.push_frame(&quiet, 5);
        assert_eq!(closed.len(), 1, "gap must flush the open region");
        assert_eq!((closed[0].frame_lo, closed[0].frame_hi), (0, 1));
        let reopened = det.finish();
        assert_eq!(reopened.len(), 1);
        assert_eq!((reopened[0].frame_lo, reopened[0].frame_hi), (5, 5));
    }

    #[test]
    #[should_panic(expected = "backwards")]
    fn rejects_backwards_sequence_numbers() {
        let mut det = Detector::new(64, DetectorConfig::new()).unwrap();
        det.push_frame(&[-90.0; 64], 7);
        det.push_frame(&[-90.0; 64], 6);
    }
}
