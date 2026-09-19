// SPDX-License-Identifier: MIT

//! The measure-stage engine: [`Measurer`] turns a frequency region of a
//! [`FrameView`] window into a complete [`MeasurementSet`] (FR-AM1–AM5),
//! against a caller-supplied §7.1 noise floor.
//!
//! The engine consumes the A0-0 shared types only: regions arrive as a
//! [`BinSpan`] (buildable from a [`Detection`] blob or a [`Track`]'s
//! kinematic state), frames through the [`crate::tap::SpectrumTap`] contract,
//! and the per-bin noise floor as a plain calibrated dBFS/bin slice — the
//! detect stage produces it, this stage only reads it.
//!
//! Everything is deterministic (NFR-A4): the same inputs and configuration
//! yield a bit-identical `MeasurementSet`.

use std::fmt;

use super::behavior::{self, BehaviorInputs};
use super::config::{CenterMethod, MeasureConfig};
use super::psd::{count_factor, db_to_lin, lin_to_db, snr_factor, RegionStats};
use super::temporal;
use super::{obw, CenterFrequency, Measured, MeasurementSet, ObwConvention, OccupiedBandwidth};
use crate::detect::Detection;
use crate::tap::{BandMeta, FrameView};
use crate::track::Track;

/// An inclusive range of frequency bins on the DC-centered axis — the
/// region a measurement runs over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinSpan {
    /// Lowest bin (inclusive).
    pub lo: usize,
    /// Highest bin (inclusive).
    pub hi: usize,
}

impl BinSpan {
    /// The frequency extent of a detection blob.
    pub fn from_detection(detection: &Detection) -> Self {
        BinSpan {
            lo: detection.bin_lo,
            hi: detection.bin_hi,
        }
    }

    /// The extent of a track's smoothed kinematic state
    /// (`center_offset_hz ± bandwidth_hz / 2`), padded by two bins per side
    /// so window skirts are not truncated, clamped to the band.
    pub fn from_track(track: &Track, meta: &BandMeta) -> Self {
        let center_bin = (meta.bins / 2) as f64 + track.center_offset_hz / meta.bin_hz;
        let half_bins = (track.bandwidth_hz / meta.bin_hz) / 2.0;
        let lo = (center_bin - half_bins - 2.0).floor().max(0.0) as usize;
        let hi = ((center_bin + half_bins + 2.0).ceil() as usize).min(meta.bins - 1);
        BinSpan { lo, hi: hi.max(lo) }
    }

    /// Number of bins covered.
    pub fn width(&self) -> usize {
        self.hi - self.lo + 1
    }
}

/// Errors from [`Measurer`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureError {
    /// The configuration was rejected; the message names the field and why.
    InvalidConfig(String),
    /// The inputs were inconsistent; the message names what mismatched.
    InvalidInput(String),
    /// No frame in the window had in-band power above the noise floor by
    /// the presence margin — there is nothing to measure, and reporting
    /// numbers anyway would fabricate a signal.
    NoSignal {
        /// Frames that were examined.
        frames_seen: usize,
    },
}

impl fmt::Display for MeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasureError::InvalidConfig(msg) => write!(f, "invalid measure configuration: {msg}"),
            MeasureError::InvalidInput(msg) => write!(f, "invalid measure input: {msg}"),
            MeasureError::NoSignal { frames_seen } => write!(
                f,
                "no signal present above the noise floor in any of {frames_seen} frames"
            ),
        }
    }
}

impl std::error::Error for MeasureError {}

/// The measure stage (spec §7.5). Construct once with a [`MeasureConfig`],
/// then call [`measure`](Self::measure) per track region and window.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurer {
    config: MeasureConfig,
}

impl Measurer {
    /// Build a measurer, validating the configuration up front.
    pub fn new(config: MeasureConfig) -> Result<Self, MeasureError> {
        if !config.enbw_bins.is_finite() || config.enbw_bins < 1.0 {
            return Err(MeasureError::InvalidConfig(format!(
                "window ENBW must be ≥ 1 bin (rectangular), got {}",
                config.enbw_bins
            )));
        }
        if !config.presence_margin_db.is_finite() || config.presence_margin_db < 0.0 {
            return Err(MeasureError::InvalidConfig(format!(
                "presence margin must be finite and non-negative, got {} dB",
                config.presence_margin_db
            )));
        }
        if !(0.0..=1.0).contains(&config.continuous_duty) {
            return Err(MeasureError::InvalidConfig(format!(
                "continuous-duty threshold must be within 0..=1, got {}",
                config.continuous_duty
            )));
        }
        for convention in std::iter::once(&config.headline_obw).chain(config.alt_obw.iter()) {
            match *convention {
                ObwConvention::PowerFraction { fraction } => {
                    if !(0.0..1.0).contains(&fraction) {
                        return Err(MeasureError::InvalidConfig(format!(
                            "OBW power fraction must be within 0..1, got {fraction}"
                        )));
                    }
                }
                ObwConvention::DbDown { db } => {
                    if !db.is_finite() || db <= 0.0 {
                        return Err(MeasureError::InvalidConfig(format!(
                            "OBW dB-down depth must be positive, got {db} dB"
                        )));
                    }
                }
            }
        }
        Ok(Measurer { config })
    }

    /// The validated configuration.
    pub fn config(&self) -> &MeasureConfig {
        &self.config
    }

    /// Measure the region `span` over every frame in `frames`.
    ///
    /// `noise_floor_dbfs` is the §7.1 per-bin noise floor (`N_k`), in
    /// calibrated dBFS/bin, one value per bin of the band — the detect
    /// stage's output. Frame values have `meta.cal_offset_db` applied before
    /// use; the floor must already be calibrated.
    pub fn measure(
        &self,
        span: BinSpan,
        frames: &FrameView,
        meta: &BandMeta,
        noise_floor_dbfs: &[f32],
    ) -> Result<MeasurementSet, MeasureError> {
        self.validate_inputs(span, frames, meta, noise_floor_dbfs)?;

        let floor_lin: Vec<f64> = noise_floor_dbfs
            .iter()
            .map(|&db| db_to_lin(f64::from(db)))
            .collect();
        let stats = RegionStats::compute(
            frames,
            (span.lo, span.hi),
            &floor_lin,
            meta.cal_offset_db,
            self.config.presence_margin_db,
        );
        if stats.n_on == 0 {
            return Err(MeasureError::NoSignal {
                frames_seen: stats.n_frames,
            });
        }

        // FR-AM3 — power and SNR. Linear sums over the span; the ENBW
        // division puts integrated power on the D-006 scale (see
        // `MeasureConfig::enbw_bins`), and cancels out of the SNR ratio.
        let sig_sum: f64 = stats.avg_stripped.iter().sum();
        let snr_db = lin_to_db(sig_sum / stats.noise_sum.max(f64::MIN_POSITIVE));
        let base_conf = snr_factor(snr_db) * count_factor(stats.n_on, 4.0);
        let power_dbfs = Measured::new(
            lin_to_db(sig_sum / self.config.enbw_bins) as f32,
            base_conf as f32,
        );
        let snr = Measured::new(snr_db as f32, base_conf as f32);

        // FR-AM2 — occupied bandwidth under every configured convention,
        // each carrying the convention that produced it.
        let peak_snr_db = stats.peak_snr_db();
        let headline = self.one_obw(
            &stats,
            self.config.headline_obw,
            peak_snr_db,
            base_conf,
            meta,
        );
        let obw_alt: Vec<OccupiedBandwidth> = self
            .config
            .alt_obw
            .iter()
            .map(|&c| self.one_obw(&stats, c, peak_snr_db, base_conf, meta))
            .collect();
        let headline_result = obw::compute(&stats.avg_stripped, self.config.headline_obw);

        // FR-AM1 — center and offset, carrying the method that actually
        // produced them (FR-AM2's principle, extended: a center without its
        // method is not a measurement). If the midpoint method was asked for
        // but no headline width exists, the fallback to the centroid is
        // *recorded* as the centroid — the value never claims a method that
        // did not produce it.
        let centroid_bin = {
            let total: f64 = stats.avg_stripped.iter().sum();
            let weighted: f64 = stats
                .avg_stripped
                .iter()
                .enumerate()
                .map(|(j, p)| p * (span.lo + j) as f64)
                .sum();
            weighted / total
        };
        let (center_bin, center_method) = match (self.config.center_method, &headline_result) {
            (CenterMethod::ObwMidpoint, Some(r)) => (
                span.lo as f64 + (r.lo_bins + r.hi_bins) / 2.0,
                CenterMethod::ObwMidpoint,
            ),
            _ => (centroid_bin, CenterMethod::PowerCentroid),
        };
        let carrier_offset_hz = CenterFrequency {
            hz: Measured::new(meta.bin_offset_hz(center_bin), base_conf as f32),
            method: center_method,
        };
        let center_hz = meta.bin_absolute_hz(center_bin).map(|hz| CenterFrequency {
            hz: Measured::new(hz, base_conf as f32),
            method: center_method,
        });

        // FR-AM4 — temporal descriptors from the on/off series.
        let temporal_analysis = temporal::analyze(
            &stats.on,
            meta.frame_dt_s,
            self.config.continuous_duty,
            self.config.min_pri_correlation,
        );

        // FR-AM5 — behavior flags from the per-frame centroids.
        let behavior = behavior::analyze(
            &BehaviorInputs {
                centroids: &stats.centroids,
                temporal: &temporal_analysis,
                hop_jump_bins: self.config.hop_jump_bins,
                drift_min_bins: self.config.drift_min_bins,
                frame_dt_s: meta.frame_dt_s,
            },
            meta,
        );

        Ok(MeasurementSet {
            center_hz,
            carrier_offset_hz,
            obw: headline,
            obw_alt,
            power_dbfs,
            snr_db: snr,
            temporal: temporal_analysis.temporal,
            behavior,
        })
    }

    /// Measure the frequency extent of a detection blob.
    pub fn measure_detection(
        &self,
        detection: &Detection,
        frames: &FrameView,
        meta: &BandMeta,
        noise_floor_dbfs: &[f32],
    ) -> Result<MeasurementSet, MeasureError> {
        self.measure(
            BinSpan::from_detection(detection),
            frames,
            meta,
            noise_floor_dbfs,
        )
    }

    /// Measure the extent of a track's smoothed kinematic state.
    pub fn measure_track(
        &self,
        track: &Track,
        frames: &FrameView,
        meta: &BandMeta,
        noise_floor_dbfs: &[f32],
    ) -> Result<MeasurementSet, MeasureError> {
        self.measure(
            BinSpan::from_track(track, meta),
            frames,
            meta,
            noise_floor_dbfs,
        )
    }

    /// One OBW figure under one convention, with a confidence honest about
    /// the dynamic range the convention needs: an x-dB-down width measured
    /// with a peak SNR barely above x is noise-limited, and one clipped by
    /// the span edge may extend beyond what was measured.
    fn one_obw(
        &self,
        stats: &RegionStats,
        convention: ObwConvention,
        peak_snr_db: f64,
        base_conf: f64,
        meta: &BandMeta,
    ) -> OccupiedBandwidth {
        match obw::compute(&stats.avg_stripped, convention) {
            Some(r) => {
                let depth_margin = ((peak_snr_db - r.needed_db) / 12.0).clamp(0.0, 1.0);
                let edge_penalty = if r.edge_limited { 0.3 } else { 1.0 };
                OccupiedBandwidth {
                    hz: Measured::new(
                        r.width_bins() * meta.bin_hz,
                        (base_conf * depth_margin * edge_penalty) as f32,
                    ),
                    convention,
                }
            }
            // Unreachable after the NoSignal gate, but degrade honestly:
            // zero width at zero confidence rather than a panic.
            None => OccupiedBandwidth {
                hz: Measured::new(0.0, 0.0),
                convention,
            },
        }
    }

    fn validate_inputs(
        &self,
        span: BinSpan,
        frames: &FrameView,
        meta: &BandMeta,
        noise_floor_dbfs: &[f32],
    ) -> Result<(), MeasureError> {
        if frames.is_empty() {
            return Err(MeasureError::InvalidInput(
                "the frame window holds no frames".to_string(),
            ));
        }
        if frames.bins() != meta.bins {
            return Err(MeasureError::InvalidInput(format!(
                "frame width {} does not match band metadata bins {}",
                frames.bins(),
                meta.bins
            )));
        }
        if noise_floor_dbfs.len() != meta.bins {
            return Err(MeasureError::InvalidInput(format!(
                "noise floor has {} bins, the band has {}",
                noise_floor_dbfs.len(),
                meta.bins
            )));
        }
        if span.lo > span.hi || span.hi >= meta.bins {
            return Err(MeasureError::InvalidInput(format!(
                "bin span {}..={} does not fit a {}-bin band",
                span.lo, span.hi, meta.bins
            )));
        }
        if !(meta.bin_hz.is_finite() && meta.bin_hz > 0.0)
            || !(meta.frame_dt_s.is_finite() && meta.frame_dt_s > 0.0)
        {
            return Err(MeasureError::InvalidInput(format!(
                "band metadata must have positive bin width and frame period, \
                 got {} Hz and {} s",
                meta.bin_hz, meta.frame_dt_s
            )));
        }
        Ok(())
    }
}
