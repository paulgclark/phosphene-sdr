// SPDX-License-Identifier: MIT

//! The FR-AM signal-inspector measurements (nextgen spec §5.2, §7.5).
//!
//! This module owns the [`MeasurementSet`] contract — the bundle the measure
//! stage computes per track from spectrum frames alone — the [`Measured`]
//! wrapper that carries the NFR-A3 obligation (*every asserted value carries
//! a confidence*), and the [`Measurer`] engine that fills them in (lane
//! A0-3):
//!
//! * [`engine`] — [`Measurer`]: the §7.5 pipeline over a [`BinSpan`] region
//!   of a [`crate::tap::FrameView`] window, against a caller-supplied §7.1
//!   noise floor.
//! * [`config`] — [`MeasureConfig`]: OBW conventions (FR-AM2), the stated
//!   center method (FR-AM1), the window-ENBW power calibration term
//!   (FR-AM3 / D-006), and the presence/periodicity thresholds (FR-AM4/AM5).
//!
//! The internal stages: `psd` (presence + on-frame-averaged in-band PSD),
//! `obw` (power-fraction and dB-down widths), `temporal` (duty, burst
//! length, PRI via on/off autocorrelation), `behavior` (hopping, bursty,
//! drifting). All of it is deterministic (NFR-A4).

use serde::{Deserialize, Serialize};

mod behavior;
pub mod config;
pub mod engine;
mod obw;
mod psd;
#[cfg(test)]
mod seal_tests;
mod temporal;

pub use config::{CenterMethod, MeasureConfig};
pub use engine::{BinSpan, MeasureError, Measurer};

/// A value with the confidence it was asserted at (NFR-A3: phosphene never
/// prints a number it cannot defend; confidence is `0.0..=1.0`, and how it is
/// *presented* is a design/A1 decision — spec §13 Q7).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Measured<T> {
    /// The measured value.
    pub value: T,
    /// Confidence in `value`, `0.0..=1.0`.
    pub confidence: f32,
}

impl<T> Measured<T> {
    /// Wrap a value with its confidence.
    pub fn new(value: T, confidence: f32) -> Self {
        Measured { value, confidence }
    }
}

/// Which convention produced an occupied-bandwidth figure (FR-AM2: always
/// state the convention).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ObwConvention {
    /// Width containing the given fraction of total in-blob power
    /// (`0.99` = the classic 99%-power OBW).
    PowerFraction {
        /// Fraction of total power contained, `0.0..1.0`.
        fraction: f32,
    },
    /// Width between the points the given dB below the in-blob peak.
    DbDown {
        /// dB below peak defining the edges (positive number).
        db: f32,
    },
}

impl ObwConvention {
    /// 99%-power OBW.
    pub const POWER_99: ObwConvention = ObwConvention::PowerFraction { fraction: 0.99 };
    /// −26 dBc width — the ITU-aligned default convention (FR-AM2).
    pub const DB_DOWN_26: ObwConvention = ObwConvention::DbDown { db: 26.0 };
    /// −3 dB (half-power) width.
    pub const DB_DOWN_3: ObwConvention = ObwConvention::DbDown { db: 3.0 };
    /// −20 dB width.
    pub const DB_DOWN_20: ObwConvention = ObwConvention::DbDown { db: 20.0 };
}

/// An occupied-bandwidth figure together with the convention that produced
/// it (FR-AM2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OccupiedBandwidth {
    /// The bandwidth, Hz.
    pub hz: Measured<f64>,
    /// The convention that produced it.
    pub convention: ObwConvention,
}

/// Temporal descriptors of a track (FR-AM4), from its on/off transitions
/// over a window. A continuous signal has `duty` ≈ 1 and no `period_s`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Temporal {
    /// Fraction of the observation window the signal was present,
    /// `0.0..=1.0`.
    pub duty: Measured<f32>,
    /// Mean burst (on) duration in seconds; `None` for continuous signals.
    pub mean_burst_s: Option<Measured<f64>>,
    /// Dominant repetition interval / PRI in seconds when the on/off pattern
    /// is periodic; `None` otherwise ("continuous" or aperiodic).
    pub period_s: Option<Measured<f64>>,
}

/// Detail of resolved frequency-hopping behavior (FR-AM5: "hopping, with hop
/// set/rate if resolvable").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HopDetail {
    /// The resolved hop frequencies (Hz, same frame of reference as the
    /// owning measurement's center), if the set was resolved. Each carries
    /// the confidence its own evidence supports (NFR-A3): a frequency
    /// dwelled on many times is better defended than one seen twice, and
    /// they must not be presented as equally certain.
    pub set_hz: Vec<Measured<f64>>,
    /// Hops per second, if resolved.
    pub rate_hz: Option<Measured<f64>>,
}

/// Behavior flags for a track (FR-AM5). Each flag is asserted with the
/// confidence its evidence supports (NFR-A3) — whichever way it points: a
/// "not hopping" over five observed frames is a weaker claim than one over
/// sixty. There is deliberately no `Default`: a defaulted flag would be an
/// undefended assertion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorFlags {
    /// The emitter hops between frequencies (FR-AD5 clustering output).
    pub hopping: Measured<bool>,
    /// The emitter transmits in bursts rather than continuously.
    pub bursty: Measured<bool>,
    /// The center frequency is drifting.
    pub drifting: Measured<bool>,
    /// Hop set/rate detail when `hopping` and resolvable.
    pub hop: Option<HopDetail>,
}

/// A center-frequency figure together with the method that produced it —
/// power-weighted centroid or occupied-band midpoint (FR-AM1 / spec §7.5
/// "state which"). Mirrors [`OccupiedBandwidth`], on the same FR-AM2
/// principle: a center without its method is not a measurement, and a
/// consumer of the serialized set must not have to reach back into the
/// producing configuration to know what the number means.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CenterFrequency {
    /// The frequency, Hz.
    pub hz: Measured<f64>,
    /// The method that produced it.
    pub method: CenterMethod,
}

/// The full FR-AM measurement bundle for one track — everything the Signal
/// Inspector can say from spectrum frames alone (spec §5.2, §7.5). Computed
/// by the measure stage, embedded in [`crate::track::Track`], and rendered by
/// [`crate::annotate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementSet {
    /// Absolute center frequency, Hz (FR-AM1) — `None` when the band center
    /// is unknown (v1 clarification C5: honestly absent, never invented).
    pub center_hz: Option<CenterFrequency>,
    /// Carrier offset from the band center, Hz (FR-AM1). Always available;
    /// carries the same method as [`center_hz`](Self::center_hz).
    pub carrier_offset_hz: CenterFrequency,
    /// Occupied bandwidth under the configured headline convention
    /// (FR-AM2; default [`ObwConvention::DB_DOWN_26`]).
    pub obw: OccupiedBandwidth,
    /// The same width under additional conventions (FR-AM2 also exposes
    /// 99%-power and −3/−20 dB).
    pub obw_alt: Vec<OccupiedBandwidth>,
    /// In-band integrated power, calibrated to v1's dBFS reference (FR-AM3).
    pub power_dbfs: Measured<f32>,
    /// SNR versus the FR-AD1 noise floor, dB (FR-AM3).
    pub snr_db: Measured<f32>,
    /// Temporal descriptors (FR-AM4).
    pub temporal: Temporal,
    /// Behavior flags (FR-AM5).
    pub behavior: BehaviorFlags,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obw_convention_constants_match_fr_am2() {
        assert_eq!(
            ObwConvention::POWER_99,
            ObwConvention::PowerFraction { fraction: 0.99 }
        );
        assert_eq!(
            ObwConvention::DB_DOWN_26,
            ObwConvention::DbDown { db: 26.0 }
        );
    }
}
