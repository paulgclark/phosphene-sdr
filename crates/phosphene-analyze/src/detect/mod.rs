// SPDX-License-Identifier: MIT

//! Detection: noise floor, CFAR, blob extraction (nextgen spec §7.1–§7.3).
//!
//! This module owns [`Detection`] — the per-blob output the tracker
//! consumes — and the three algorithm stages that produce it, one submodule
//! per spec section:
//!
//! * [`noise`] — per-bin rank-order noise floor over a sliding window
//!   (§7.1, FR-AD1);
//! * [`cfar`] — cell-averaging CFAR along frequency to a configurable Pfa,
//!   producing the per-frame occupancy mask (§7.2, FR-AD2);
//! * [`blob`] — streaming connected components over the (freq, time)
//!   occupancy mask (§7.3, FR-AD3);
//! * [`detector`] — the three assembled into the streaming
//!   [`Detector`] pipeline consuming [`crate::tap::SpectrumTap`] frames.
//!
//! Everything is deterministic for a fixed input and configuration
//! (NFR-A4), and calibrated honesty is tested rather than assumed: the seal
//! tests drive the pipeline with M0 generator scenes of known ground truth
//! and verify the measured false-alarm rate against the configured Pfa, the
//! detection-probability curve against SNR, and the floor estimate against
//! the generator's configured noise level.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::tap::BandMeta;

pub mod blob;
pub mod cfar;
pub mod detector;
pub mod noise;

#[cfg(test)]
mod identity_tests;
#[cfg(test)]
mod seal_tests;

pub use blob::BlobBuilder;
pub use cfar::{Cfar, CfarConfig};
pub use detector::{Detector, DetectorConfig};
pub use noise::{NoiseFloor, NoiseFloorConfig};

/// Errors from configuring the detection stages.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum DetectError {
    /// The configuration was rejected; the message names the field and why.
    InvalidConfig(String),
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectError::InvalidConfig(msg) => {
                write!(f, "invalid detector configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for DetectError {}

/// One detection region: a connected component of the (freq, time) occupancy
/// mask (§7.3), in the grid coordinates of the [`crate::tap::SpectrumTap`]
/// frames it was found in. Grid indices are the single source of truth;
/// convert to Hz through [`BandMeta`] via the helper methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Lowest occupied bin (inclusive, DC-centered axis).
    pub bin_lo: usize,
    /// Highest occupied bin (inclusive).
    pub bin_hi: usize,
    /// First frame of the component (producer sequence number, inclusive).
    pub frame_lo: u64,
    /// Last frame of the component (inclusive).
    pub frame_hi: u64,
    /// Power-weighted centroid bin (fractional; §7.5 center convention).
    pub centroid_bin: f64,
    /// Peak per-bin power inside the component, dBFS/bin.
    pub peak_dbfs: f32,
    /// Integrated in-component power, dBFS.
    pub power_dbfs: f32,
    /// Peak SNR versus the §7.1 noise floor, dB.
    pub snr_db: f32,
}

impl Detection {
    /// Frequency extent as offsets from band center: `(low, high)` Hz edges
    /// of the occupied bins.
    pub fn offset_span_hz(&self, meta: &BandMeta) -> (f64, f64) {
        (
            meta.bin_offset_hz(self.bin_lo as f64 - 0.5),
            meta.bin_offset_hz(self.bin_hi as f64 + 0.5),
        )
    }

    /// Power-weighted centroid as an offset from band center, Hz.
    pub fn centroid_offset_hz(&self, meta: &BandMeta) -> f64 {
        meta.bin_offset_hz(self.centroid_bin)
    }

    /// Time extent in seconds (frame count × frame period).
    pub fn duration_s(&self, meta: &BandMeta) -> f64 {
        (self.frame_hi - self.frame_lo + 1) as f64 * meta.frame_dt_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> BandMeta {
        BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        }
    }

    #[test]
    fn detection_converts_grid_extents_to_hz() {
        let d = Detection {
            bin_lo: 810,
            bin_hi: 814,
            frame_lo: 100,
            frame_hi: 109,
            centroid_bin: 812.0,
            peak_dbfs: -20.0,
            power_dbfs: -19.0,
            snr_db: 40.0,
        };
        let m = meta();
        assert_eq!(d.centroid_offset_hz(&m), 300_000.0);
        let (lo, hi) = d.offset_span_hz(&m);
        assert_eq!(lo, 297_500.0);
        assert_eq!(hi, 302_500.0);
        assert!((d.duration_s(&m) - 0.010).abs() < 1e-12);
    }
}
