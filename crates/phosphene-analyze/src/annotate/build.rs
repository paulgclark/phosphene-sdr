// SPDX-License-Identifier: MIT

//! Builds a render-ready [`Annotation`] from a [`Track`] (annotation design
//! `specs/002-nextgen-analyze/annotation-design.md` §2.2/§2.2.1). This is the
//! content contract with `phosphene-render`: `Annotation.label`/`.detail` are
//! opaque text to render, built here from the real measurement/classification
//! types (never the reverse — render never parses them back out, §2.2).
//!
//! `rate_known` threads the D-028 §2 / D-098 §7.3 rate-optional convention
//! through every Hz-suffixed token: hertz when the source declared a sample
//! rate, a bare cycles/sample fraction (mirroring the render axis's
//! `FreqValue::Normalized`) when it did not. It is passed explicitly rather
//! than read off `BandMeta`, because `BandMeta.center_freq_hz: None` alone
//! cannot distinguish "rate known, center unset" (v1 clarification C5) from
//! "no rate declared at all" (D-028 §2) — the caller (the live-integration
//! wiring, which owns the source metadata) already knows which case it is in.
//!
//! Every compact-number formatter here is a **local, independent**
//! reimplementation of the `phosphene-render` chrome's `format_hz` idiom —
//! this crate cannot depend on `phosphene-render` (spec §8.1 crate
//! boundaries), so the two are duplicated by necessity, not oversight.

use crate::measure::{
    BehaviorFlags, CenterMethod, HopDetail, Measured, MeasurementSet, ObwConvention,
    OccupiedBandwidth, Temporal,
};
use crate::tap::BandMeta;
use crate::track::{Classification, ModulationClass, Track};

use super::{Annotation, FreqSpanHz};

/// Build the render-ready annotation for `track`, or `None` before the
/// measure stage has produced anything for it yet — annotate has nothing to
/// say about a track it cannot measure.
pub fn build_annotation(track: &Track, meta: &BandMeta, rate_known: bool) -> Option<Annotation> {
    let m = track.measurements.as_ref()?;

    let anchor_hz = match &m.center_hz {
        Some(cf) => cf.hz.value,
        None => m.carrier_offset_hz.hz.value,
    };
    let half_bw = m.obw.hz.value / 2.0;
    let span = FreqSpanHz {
        low_hz: anchor_hz - half_bw,
        high_hz: anchor_hz + half_bw,
    };

    Some(Annotation {
        track: track.id,
        span,
        anchor_hz,
        label: build_label(track, m, rate_known),
        detail: build_detail(m, meta, rate_known),
        confidence: label_confidence(track, m),
        strength_db: m.snr_db.value,
    })
}

fn identity_token(class: &Option<Classification>) -> String {
    match class {
        None => "SIGNAL".to_string(),
        Some(c) => match &c.class {
            ModulationClass::Unknown => "SIGNAL".to_string(),
            ModulationClass::OokAsk => "OOK".to_string(),
            ModulationClass::Fsk { levels, .. } => format!("FSK{levels}"),
            ModulationClass::Ofdm { .. } => "OFDM".to_string(),
            ModulationClass::ChirpCss { .. } => "CHIRP".to_string(),
            ModulationClass::Linear => "LINEAR".to_string(),
        },
    }
}

/// The one behavior flag the headline shows, most specific/rare first
/// (§2.2): hopping, else drifting, else bursty, else none.
fn flag_token(behavior: &BehaviorFlags) -> Option<&'static str> {
    if behavior.hopping.value {
        Some("HOP")
    } else if behavior.drifting.value {
        Some("DRIFT")
    } else if behavior.bursty.value {
        Some("BURST")
    } else {
        None
    }
}

fn build_label(track: &Track, m: &MeasurementSet, rate_known: bool) -> String {
    let mut parts = vec![
        identity_token(&track.class),
        hz_mag(m.obw.hz.value, rate_known),
        format!("{}dB", round_i(m.snr_db.value)),
    ];
    if let Some(flag) = flag_token(&m.behavior) {
        parts.push(flag.to_string());
    }
    if let Some(sig) = track.signatures.first() {
        if sig.score >= 0.5 {
            parts.push(format!("~{}", sig.label));
        }
    }
    parts.join(" · ")
}

/// The headline's aggregate confidence (§2.2): the minimum of every value the
/// headline text actually asserts. The signature suffix's own `score` never
/// enters this fold — a weak signature guess does not drag down the measured
/// headline's stated confidence.
fn label_confidence(track: &Track, m: &MeasurementSet) -> f32 {
    let mut conf = m.obw.hz.confidence.min(m.snr_db.confidence);
    let resolved_class = track
        .class
        .as_ref()
        .filter(|c| !matches!(c.class, ModulationClass::Unknown));
    if let Some(c) = resolved_class {
        conf = conf.min(c.confidence);
    }
    conf
}

fn build_detail(m: &MeasurementSet, meta: &BandMeta, rate_known: bool) -> Vec<String> {
    let mut lines = vec![center_line(m, rate_known), bw_line(&m.obw, rate_known)];
    for alt in &m.obw_alt {
        lines.push(bw_line(alt, rate_known));
    }
    lines.push(power_line(m));
    lines.push(temporal_line(&m.temporal));
    if m.behavior.hopping.value {
        lines.push(hop_line(
            &m.behavior.hopping,
            m.behavior.hop.as_ref(),
            rate_known,
        ));
        if let Some(hd) = &m.behavior.hop {
            if let Some(follow) = hop_set_line(hd, meta, rate_known) {
                lines.push(follow);
            }
        }
    }
    if m.behavior.drifting.value {
        lines.push(format!(
            "DRIFT (conf {}%)",
            round_pct(m.behavior.drifting.confidence)
        ));
    }
    lines
}

fn center_line(m: &MeasurementSet, rate_known: bool) -> String {
    match &m.center_hz {
        Some(cf) => {
            let method = match cf.method {
                CenterMethod::PowerCentroid => "CENTROID",
                CenterMethod::ObwMidpoint => "MIDPOINT",
            };
            let conf = cf.hz.confidence.min(m.carrier_offset_hz.hz.confidence);
            format!(
                "CENTER {} {method} Δ{} (conf {}%)",
                hz_mag(cf.hz.value, rate_known),
                hz_signed(m.carrier_offset_hz.hz.value, rate_known),
                round_pct(conf)
            )
        }
        None => format!(
            "CENTER Δ{} (conf {}%)",
            hz_signed(m.carrier_offset_hz.hz.value, rate_known),
            round_pct(m.carrier_offset_hz.hz.confidence)
        ),
    }
}

fn bw_line(o: &OccupiedBandwidth, rate_known: bool) -> String {
    format!(
        "BW {} {} (conf {}%)",
        hz_mag(o.hz.value, rate_known),
        convention_token(o.convention),
        round_pct(o.hz.confidence)
    )
}

fn convention_token(c: ObwConvention) -> String {
    match c {
        ObwConvention::PowerFraction { fraction } => {
            format!("{}PWR%", round_i(fraction * 100.0))
        }
        ObwConvention::DbDown { db } => format!("-{}dB", round_i(db)),
    }
}

fn power_line(m: &MeasurementSet) -> String {
    format!(
        "POWER {:.1}dBFS SNR {}dB (conf {}%)",
        m.power_dbfs.value,
        round_i(m.snr_db.value),
        round_pct(m.power_dbfs.confidence.min(m.snr_db.confidence))
    )
}

fn temporal_line(t: &Temporal) -> String {
    let duty_pct = round_i(t.duty.value * 100.0);
    match (&t.mean_burst_s, &t.period_s) {
        (Some(burst), Some(period)) => format!(
            "DUTY {duty_pct}% BURST {} PRI {} (conf {}%)",
            compact_time(burst.value),
            compact_time(period.value),
            round_pct(
                t.duty
                    .confidence
                    .min(burst.confidence)
                    .min(period.confidence)
            )
        ),
        (Some(burst), None) => format!(
            "DUTY {duty_pct}% BURST {} (conf {}%)",
            compact_time(burst.value),
            round_pct(t.duty.confidence.min(burst.confidence))
        ),
        // Both `None` (continuous), and the type-shape-allowed but
        // unreachable `(None, Some)` — the §6 fallback rule: show less
        // (drop the impossible PRI) rather than invent a burst.
        (None, _) => format!(
            "CONTINUOUS DUTY {duty_pct}% (conf {}%)",
            round_pct(t.duty.confidence)
        ),
    }
}

/// The concise `HOP` line (§2.2.1 item 5). An empty (or absent) hop set
/// cannot derive `<n>`/`<span>`, so both fall back to the plainest form —
/// `hopping.confidence` alone is what's actually known either way.
fn hop_line(hopping: &Measured<bool>, hop: Option<&HopDetail>, rate_known: bool) -> String {
    let set_hz = hop.map_or(&[][..], |h| h.set_hz.as_slice());
    if set_hz.is_empty() {
        return format!("HOP suspected (conf {}%)", round_pct(hopping.confidence));
    }
    let n = set_hz.len();
    let min_v = set_hz.iter().map(|f| f.value).fold(f64::INFINITY, f64::min);
    let max_v = set_hz
        .iter()
        .map(|f| f.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = hz_mag(max_v - min_v, rate_known);
    let set_conf_min = set_hz
        .iter()
        .map(|f| f.confidence)
        .fold(f32::INFINITY, f32::min);
    let hd = hop.expect("a non-empty set implies `hop` is Some");
    match &hd.rate_hz {
        Some(rate) => {
            let conf = hopping.confidence.min(rate.confidence).min(set_conf_min);
            format!(
                "HOP {n}ch {}/s span {span} (conf {}%)",
                compact_rate(rate.value),
                round_pct(conf)
            )
        }
        None => {
            let conf = hopping.confidence.min(set_conf_min);
            format!("HOP {n}ch span {span} (conf {}%)", round_pct(conf))
        }
    }
}

/// The optional `HOP SET` follow-up line, only for a concise 1–6-entry set
/// (§2.2.1 item 5); `None` otherwise (an empty set already fell back in
/// [`hop_line`], and above 6 entries the concise line stands alone).
fn hop_set_line(hd: &HopDetail, meta: &BandMeta, rate_known: bool) -> Option<String> {
    if !(1..=6).contains(&hd.set_hz.len()) {
        return None;
    }
    let entries: Vec<String> = hd
        .set_hz
        .iter()
        .map(|f| {
            let tok = if meta.center_freq_hz.is_some() {
                hz_mag(f.value, rate_known)
            } else {
                hz_signed(f.value, rate_known)
            };
            format!("{tok}(conf {}%)", round_pct(f.confidence))
        })
        .collect();
    Some(format!("HOP SET {}", entries.join(", ")))
}

fn round_pct(c: f32) -> i64 {
    round_i(c.clamp(0.0, 1.0) * 100.0)
}

fn round_i(v: f32) -> i64 {
    v.round() as i64
}

/// Compact Hz **magnitude** (always non-negative) — `12.5K`, `915.1M` — or,
/// with `rate_known` false, the bare cycles/sample fraction (D-028 §2 / D-098
/// §7.3), never a fabricated Hz unit.
fn hz_mag(hz: f64, rate_known: bool) -> String {
    if !rate_known {
        return trimmed(hz.abs(), 4);
    }
    let (value, unit) = compact_unit(hz.abs());
    format!("{}{unit}", trimmed(value, 2))
}

/// Compact Hz, signed (`+100K`, `-960K`) — the Δ-offset / hop-set idiom — or,
/// with `rate_known` false, a signed bare fraction.
fn hz_signed(hz: f64, rate_known: bool) -> String {
    let sign = if hz < 0.0 { "-" } else { "+" };
    if !rate_known {
        return format!("{sign}{}", trimmed(hz.abs(), 4));
    }
    let (value, unit) = compact_unit(hz.abs());
    format!("{sign}{}{unit}", trimmed(value, 2))
}

fn compact_unit(a: f64) -> (f64, &'static str) {
    if a >= 1e9 {
        (a / 1e9, "G")
    } else if a >= 1e6 {
        (a / 1e6, "M")
    } else if a >= 1e3 {
        (a / 1e3, "K")
    } else {
        (a, "")
    }
}

/// `value` to `decimals` places, trailing fractional zeros trimmed (`"0"`
/// never becomes empty).
fn trimmed(value: f64, decimals: usize) -> String {
    let s = format!("{value:.decimals$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s.is_empty() || s == "-0" {
        "0".to_string()
    } else {
        s
    }
}

/// A per-second rate: one decimal, trimmed — no K/M/G scaling (hop rates stay
/// in the single/double digits in practice; the headline grammar names no
/// scaled form for it).
fn compact_rate(v: f64) -> String {
    trimmed(v, 1)
}

/// Compact time magnitude: µs below 1 ms, ms below 1 s, else s (§2.2.1 item
/// 4's picker).
fn compact_time(s: f64) -> String {
    let a = s.abs();
    let (value, unit) = if a < 1e-3 {
        (a * 1e6, "µs")
    } else if a < 1.0 {
        (a * 1e3, "ms")
    } else {
        (a, "s")
    };
    format!("{}{unit}", trimmed(value, 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::{SignatureCandidate, TrackId, TrackState};

    fn meta(center: Option<f64>) -> BandMeta {
        BandMeta {
            center_freq_hz: center,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        }
    }

    fn m(v: f64, c: f32) -> Measured<f64> {
        Measured::new(v, c)
    }

    fn base_measurements() -> MeasurementSet {
        MeasurementSet {
            center_hz: None,
            carrier_offset_hz: crate::measure::CenterFrequency {
                hz: m(0.0, 0.9),
                method: CenterMethod::PowerCentroid,
            },
            obw: OccupiedBandwidth {
                hz: m(12_500.0, 0.88),
                convention: ObwConvention::DB_DOWN_26,
            },
            obw_alt: Vec::new(),
            power_dbfs: Measured::new(-42.3, 0.94),
            snr_db: Measured::new(22.0, 0.90),
            temporal: Temporal {
                duty: Measured::new(0.99, 0.97),
                mean_burst_s: None,
                period_s: None,
            },
            behavior: BehaviorFlags {
                hopping: Measured::new(false, 0.9),
                bursty: Measured::new(false, 0.9),
                drifting: Measured::new(false, 0.9),
                hop: None,
            },
        }
    }

    fn track(measurements: Option<MeasurementSet>) -> Track {
        Track {
            id: TrackId(1),
            state: TrackState::Confirmed,
            center_offset_hz: 0.0,
            bandwidth_hz: 12_500.0,
            power_dbfs: -42.3,
            snr_db: 22.0,
            born_frame: 0,
            last_seen_frame: 10,
            measurements,
            class: None,
            signatures: Vec::new(),
        }
    }

    // --- AC-20: CENTER line -------------------------------------------

    #[test]
    fn ac20_center_line_with_known_center_folds_the_minimum_confidence() {
        let mut mset = base_measurements();
        mset.center_hz = Some(crate::measure::CenterFrequency {
            hz: m(915_100_000.0, 0.96),
            method: CenterMethod::PowerCentroid,
        });
        mset.carrier_offset_hz = crate::measure::CenterFrequency {
            hz: m(100_000.0, 0.99),
            method: CenterMethod::PowerCentroid,
        };
        assert_eq!(
            center_line(&mset, true),
            "CENTER 915.1M CENTROID Δ+100K (conf 96%)"
        );

        // Swap which side is weaker: the printed confidence follows.
        mset.center_hz = Some(crate::measure::CenterFrequency {
            hz: m(915_100_000.0, 0.99),
            method: CenterMethod::PowerCentroid,
        });
        mset.carrier_offset_hz = crate::measure::CenterFrequency {
            hz: m(100_000.0, 0.80),
            method: CenterMethod::PowerCentroid,
        };
        assert_eq!(
            center_line(&mset, true),
            "CENTER 915.1M CENTROID Δ+100K (conf 80%)"
        );
    }

    #[test]
    fn ac20_center_line_uses_midpoint_token() {
        let mut mset = base_measurements();
        mset.center_hz = Some(crate::measure::CenterFrequency {
            hz: m(2_450_000_000.0, 0.9),
            method: CenterMethod::ObwMidpoint,
        });
        assert!(center_line(&mset, true).contains("MIDPOINT"));
    }

    #[test]
    fn ac20_center_line_with_unknown_center_omits_the_absolute_token() {
        let mut mset = base_measurements();
        mset.center_hz = None;
        mset.carrier_offset_hz = crate::measure::CenterFrequency {
            hz: m(-50_000.0, 0.77),
            method: CenterMethod::PowerCentroid,
        };
        assert_eq!(center_line(&mset, true), "CENTER Δ-50K (conf 77%)");
    }

    // --- AC-21: BW lines -------------------------------------------------

    #[test]
    fn ac21_bw_lines_are_headline_first_then_alt_in_order_each_own_confidence() {
        let mut mset = base_measurements();
        mset.obw = OccupiedBandwidth {
            hz: m(12_500.0, 0.88),
            convention: ObwConvention::DB_DOWN_26,
        };
        mset.obw_alt = vec![
            OccupiedBandwidth {
                hz: m(14_000.0, 0.91),
                convention: ObwConvention::POWER_99,
            },
            OccupiedBandwidth {
                hz: m(8_000.0, 0.90),
                convention: ObwConvention::DB_DOWN_3,
            },
            OccupiedBandwidth {
                hz: m(20_000.0, 0.85),
                convention: ObwConvention::DB_DOWN_20,
            },
        ];
        let lines = build_detail(&mset, &meta(None), true);
        // CENTER is line 0; BW lines follow, headline then alt in order.
        assert_eq!(lines[1], "BW 12.5K -26dB (conf 88%)");
        assert_eq!(lines[2], "BW 14K 99PWR% (conf 91%)");
        assert_eq!(lines[3], "BW 8K -3dB (conf 90%)");
        assert_eq!(lines[4], "BW 20K -20dB (conf 85%)");
    }

    // --- AC-22: POWER line -------------------------------------------------

    #[test]
    fn ac22_power_line_confidence_is_the_minimum_not_average() {
        let mut mset = base_measurements();
        mset.power_dbfs = Measured::new(-42.3, 0.94);
        mset.snr_db = Measured::new(21.4, 0.80);
        assert_eq!(power_line(&mset), "POWER -42.3dBFS SNR 21dB (conf 80%)");
    }

    // --- AC-23: TEMPORAL three shapes ---------------------------------

    #[test]
    fn ac23_temporal_all_three_shapes() {
        let both = Temporal {
            duty: Measured::new(0.40, 0.90),
            mean_burst_s: Some(m(0.040, 0.85)),
            period_s: Some(m(1.90, 0.70)),
        };
        assert_eq!(
            temporal_line(&both),
            "DUTY 40% BURST 40ms PRI 1.9s (conf 70%)"
        );

        let burst_only = Temporal {
            duty: Measured::new(0.25, 0.9),
            mean_burst_s: Some(m(0.0004, 0.6)),
            period_s: None,
        };
        assert_eq!(
            temporal_line(&burst_only),
            "DUTY 25% BURST 400µs (conf 60%)"
        );

        let continuous = Temporal {
            duty: Measured::new(0.99, 0.97),
            mean_burst_s: None,
            period_s: None,
        };
        assert_eq!(temporal_line(&continuous), "CONTINUOUS DUTY 99% (conf 97%)");
    }

    // --- AC-24: HOP four cases + HOP SET follow-up ------------------------

    fn hop_measured(v: f64, c: f32) -> Measured<f64> {
        Measured::new(v, c)
    }

    #[test]
    fn ac24_hop_with_rate_and_set() {
        let hopping = Measured::new(true, 0.90);
        let hop = HopDetail {
            set_hz: vec![
                hop_measured(-375_000.0, 0.70),
                hop_measured(-125_000.0, 0.65),
                hop_measured(125_000.0, 0.60),
                hop_measured(375_000.0, 0.55),
            ],
            rate_hz: Some(hop_measured(66.7, 0.80)),
        };
        assert_eq!(
            hop_line(&hopping, Some(&hop), true),
            "HOP 4ch 66.7/s span 750K (conf 55%)"
        );
        let follow = hop_set_line(&hop, &meta(None), true).unwrap();
        assert_eq!(
            follow,
            "HOP SET -375K(conf 70%), -125K(conf 65%), +125K(conf 60%), +375K(conf 55%)"
        );
    }

    #[test]
    fn ac24_hop_without_rate() {
        let hopping = Measured::new(true, 0.90);
        let hop = HopDetail {
            set_hz: vec![
                hop_measured(-375_000.0, 0.70),
                hop_measured(375_000.0, 0.55),
            ],
            rate_hz: None,
        };
        assert_eq!(
            hop_line(&hopping, Some(&hop), true),
            "HOP 2ch span 750K (conf 55%)"
        );
    }

    #[test]
    fn ac24_hop_none_and_empty_set_are_byte_identical() {
        let hopping = Measured::new(true, 0.90);
        let none_case = hop_line(&hopping, None, true);
        let empty_set = HopDetail {
            set_hz: Vec::new(),
            rate_hz: Some(hop_measured(10.0, 0.5)),
        };
        let empty_case = hop_line(&hopping, Some(&empty_set), true);
        assert_eq!(none_case, "HOP suspected (conf 90%)");
        assert_eq!(none_case, empty_case);
        // An empty set never produces a HOP SET line either.
        assert!(hop_set_line(&empty_set, &meta(None), true).is_none());
    }

    #[test]
    fn ac24_hop_set_line_uses_absolute_hz_when_center_is_known() {
        let hop = HopDetail {
            set_hz: vec![
                hop_measured(99_800_000.0, 0.7),
                hop_measured(100_200_000.0, 0.7),
            ],
            rate_hz: None,
        };
        let follow = hop_set_line(&hop, &meta(Some(100_000_000.0)), true).unwrap();
        assert_eq!(follow, "HOP SET 99.8M(conf 70%), 100.2M(conf 70%)");
    }

    #[test]
    fn ac24_hop_set_line_absent_above_six_entries() {
        let hop = HopDetail {
            set_hz: (0..7)
                .map(|i| hop_measured(i as f64 * 1000.0, 0.5))
                .collect(),
            rate_hz: None,
        };
        assert!(hop_set_line(&hop, &meta(None), true).is_none());
    }

    // --- AC-25: DRIFT line -------------------------------------------------

    #[test]
    fn ac25_drift_line_states_no_numeric_rate() {
        let mut mset = base_measurements();
        mset.behavior.drifting = Measured::new(true, 0.83);
        let lines = build_detail(&mset, &meta(None), true);
        let drift = lines.iter().find(|l| l.starts_with("DRIFT")).unwrap();
        assert_eq!(drift, "DRIFT (conf 83%)");
    }

    // --- AC-26: producer-side headline label --------------------------

    #[test]
    fn ac26_identity_token_mapping_table() {
        let cases: &[(Option<Classification>, &str)] = &[
            (None, "SIGNAL"),
            (
                Some(Classification {
                    class: ModulationClass::Unknown,
                    symbol_rate_bd: None,
                    confidence: 0.5,
                    evidence: vec![],
                }),
                "SIGNAL",
            ),
            (
                Some(Classification {
                    class: ModulationClass::OokAsk,
                    symbol_rate_bd: None,
                    confidence: 0.9,
                    evidence: vec![],
                }),
                "OOK",
            ),
            (
                Some(Classification {
                    class: ModulationClass::Fsk {
                        levels: 4,
                        deviation_hz: m(4_500.0, 0.9),
                    },
                    symbol_rate_bd: None,
                    confidence: 0.9,
                    evidence: vec![],
                }),
                "FSK4",
            ),
            (
                Some(Classification {
                    class: ModulationClass::Ofdm {
                        useful_symbol_s: m(66.7e-6, 0.9),
                        cp_fraction: m(0.25, 0.9),
                    },
                    symbol_rate_bd: None,
                    confidence: 0.9,
                    evidence: vec![],
                }),
                "OFDM",
            ),
            (
                Some(Classification {
                    class: ModulationClass::ChirpCss {
                        chirp_rate_hz_per_s: m(1e9, 0.9),
                    },
                    symbol_rate_bd: None,
                    confidence: 0.9,
                    evidence: vec![],
                }),
                "CHIRP",
            ),
            (
                Some(Classification {
                    class: ModulationClass::Linear,
                    symbol_rate_bd: Some(m(9600.0, 0.8)),
                    confidence: 0.9,
                    evidence: vec![],
                }),
                "LINEAR",
            ),
        ];
        for (class, want) in cases {
            let mut t = track(Some(base_measurements()));
            t.class = class.clone();
            assert_eq!(
                build_label(&t, t.measurements.as_ref().unwrap(), true)
                    .split(" · ")
                    .next(),
                Some(*want)
            );
        }
    }

    #[test]
    fn ac26_signature_suffix_threshold_and_flag_priority() {
        let mset = base_measurements();
        let mut t = track(Some(mset.clone()));

        // No signature: no `~` suffix.
        assert!(!build_label(&t, &mset, true).contains('~'));

        // Below threshold: no suffix.
        t.signatures = vec![SignatureCandidate {
            label: "generic ISM".to_string(),
            score: 0.49,
            source: "test".to_string(),
        }];
        assert!(!build_label(&t, &mset, true).contains('~'));

        // At/above threshold: exactly one `~<label>` suffix, the best entry.
        t.signatures = vec![SignatureCandidate {
            label: "POCSAG".to_string(),
            score: 0.5,
            source: "test".to_string(),
        }];
        let label = build_label(&t, &mset, true);
        assert!(label.ends_with("~POCSAG"), "label: {label}");

        // Flag priority: hopping beats drifting beats bursty.
        let mut hopping_mset = mset.clone();
        hopping_mset.behavior.hopping = Measured::new(true, 0.9);
        hopping_mset.behavior.drifting = Measured::new(true, 0.9);
        hopping_mset.behavior.bursty = Measured::new(true, 0.9);
        let t2 = track(Some(hopping_mset.clone()));
        let label = build_label(&t2, &hopping_mset, true);
        assert!(label.contains("HOP") && !label.contains("DRIFT") && !label.contains("BURST"));
    }

    #[test]
    fn ac26_confidence_and_strength_are_derived_not_chosen() {
        let mut mset = base_measurements();
        mset.obw.hz.confidence = 0.88;
        mset.snr_db.confidence = 0.70;
        mset.snr_db.value = 22.0;
        let mut t = track(Some(mset.clone()));
        // Unresolved class: classification confidence must NOT enter the fold.
        t.class = Some(Classification {
            class: ModulationClass::Unknown,
            symbol_rate_bd: None,
            confidence: 0.01,
            evidence: vec![],
        });
        let ann = build_annotation(&t, &meta(None), true).unwrap();
        assert!((ann.confidence - 0.70).abs() < 1e-6, "{}", ann.confidence);
        assert_eq!(ann.strength_db, 22.0);

        // Resolved class: its confidence joins the fold.
        t.class = Some(Classification {
            class: ModulationClass::OokAsk,
            symbol_rate_bd: None,
            confidence: 0.5,
            evidence: vec![],
        });
        let ann = build_annotation(&t, &meta(None), true).unwrap();
        assert!((ann.confidence - 0.5).abs() < 1e-6, "{}", ann.confidence);
    }

    // --- AC-8: rate-less formatting -------------------------------------

    #[test]
    fn ac8_rateless_tokens_are_bare_fractions_with_no_hz_unit() {
        let mut mset = base_measurements();
        mset.center_hz = None;
        mset.carrier_offset_hz = crate::measure::CenterFrequency {
            hz: m(0.125, 0.9),
            method: CenterMethod::PowerCentroid,
        };
        mset.obw.hz = m(0.01, 0.88);
        let line = center_line(&mset, false);
        assert_eq!(line, "CENTER Δ+0.125 (conf 90%)");
        for bad in ["Hz", "hz", "K", "M", "G"] {
            assert!(!line.contains(bad), "line {line} smuggled a unit ({bad})");
        }
        let t = track(Some(mset.clone()));
        let label = build_label(&t, &mset, false);
        assert!(
            !label.to_lowercase().contains("hz") && !label.contains('K') && !label.contains('M'),
            "label {label} smuggled a Hz-flavoured unit"
        );
    }

    // --- Whole-pipeline smoke: build_annotation end to end ----------------

    #[test]
    fn build_annotation_is_none_before_measurement() {
        let t = track(None);
        assert!(build_annotation(&t, &meta(None), true).is_none());
    }

    #[test]
    fn build_annotation_spans_the_headline_obw_around_the_anchor() {
        let mut mset = base_measurements();
        mset.center_hz = Some(crate::measure::CenterFrequency {
            hz: m(915_100_000.0, 0.9),
            method: CenterMethod::PowerCentroid,
        });
        mset.obw.hz = m(12_500.0, 0.9);
        let t = track(Some(mset));
        let ann = build_annotation(&t, &meta(Some(915_000_000.0)), true).unwrap();
        assert_eq!(ann.anchor_hz, 915_100_000.0);
        assert_eq!(ann.span.low_hz, 915_100_000.0 - 6_250.0);
        assert_eq!(ann.span.high_hz, 915_100_000.0 + 6_250.0);
        assert_eq!(ann.track, TrackId(1));
    }
}
