// SPDX-License-Identifier: MIT

//! Configuration for the measure stage (FR-AM1–AM5).

use serde::{Deserialize, Serialize};

use super::ObwConvention;

/// Which convention produces the *reported* center frequency (FR-AM1 /
/// spec §7.5: "power-weighted centroid or occupied-band midpoint (state
/// which)"). Selected here; carried with every reported value in
/// [`crate::measure::CenterFrequency`], so the serialized measurement is
/// self-describing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CenterMethod {
    /// Power-weighted centroid of the in-band PSD (the default).
    PowerCentroid,
    /// Midpoint of the headline occupied-bandwidth edges.
    ObwMidpoint,
}

/// Configuration for a [`crate::measure::Measurer`].
///
/// All thresholds are deterministic: the same configuration over the same
/// frames yields bit-identical measurements (NFR-A4).
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureConfig {
    /// The convention for the headline OBW figure (FR-AM2; default
    /// [`ObwConvention::DB_DOWN_26`], the ITU-aligned −26 dBc width).
    pub headline_obw: ObwConvention,
    /// Additional conventions reported alongside the headline (FR-AM2 also
    /// exposes 99%-power and −3/−20 dB; default exactly those).
    pub alt_obw: Vec<ObwConvention>,
    /// How the reported center is derived (FR-AM1, "state which").
    pub center_method: CenterMethod,
    /// Equivalent noise bandwidth, in bins, of the analysis window that
    /// produced the spectrum frames (`phosphene_core::Window::enbw_bins`).
    ///
    /// D-006 normalises spectra by the window's *coherent* gain so a tone
    /// reads its level in one bin; by Parseval the same normalisation makes
    /// the linear sum over a signal's bins overcount its power by exactly
    /// ENBW. Dividing the integrated power by this value is what makes the
    /// FR-AM3 figure mean the same thing as v1's cursor readout — a
    /// full-scale tone integrates to 0 dBFS. Default `1.5`, the Hann window
    /// v1 uses by default; live wiring must supply the actual window's value.
    pub enbw_bins: f64,
    /// A frame counts as "signal present" when its in-band power exceeds the
    /// in-band noise-floor power by this many dB (drives FR-AM4/AM5).
    /// Default 6 dB.
    pub presence_margin_db: f32,
    /// Duty at or above which the signal is reported as continuous — no
    /// burst length, no PRI (FR-AM4: never imply periodicity that was not
    /// observed). Default 0.98.
    pub continuous_duty: f32,
    /// Minimum normalised on/off autocorrelation for a repetition interval
    /// to be asserted at all (FR-AM4). Default 0.5.
    pub min_pri_correlation: f32,
    /// Minimum centroid jump, in bins, treated as a frequency hop (FR-AM5).
    /// Deliberately not scaled by the measured OBW — a hopper's aggregate
    /// PSD spans its whole hop set. Wideband signals whose centroid wobbles
    /// beyond the default need a larger configured value. Default 3.0.
    pub hop_jump_bins: f64,
    /// Minimum regression-fitted centroid motion over the window, in bins,
    /// for the drifting flag (FR-AM5). Default 2.0.
    pub drift_min_bins: f64,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        MeasureConfig {
            headline_obw: ObwConvention::DB_DOWN_26,
            alt_obw: vec![
                ObwConvention::POWER_99,
                ObwConvention::DB_DOWN_3,
                ObwConvention::DB_DOWN_20,
            ],
            center_method: CenterMethod::PowerCentroid,
            enbw_bins: 1.5,
            presence_margin_db: 6.0,
            continuous_duty: 0.98,
            min_pri_correlation: 0.5,
            hop_jump_bins: 3.0,
            drift_min_bins: 2.0,
        }
    }
}

impl MeasureConfig {
    /// The default configuration with the ENBW of the given analysis window
    /// (use when the producing window is not the 1.5-bin Hann default).
    pub fn for_window(window: &phosphene_core::Window) -> Self {
        MeasureConfig {
            enbw_bins: window.enbw_bins(),
            ..MeasureConfig::default()
        }
    }
}
