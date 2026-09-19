// SPDX-License-Identifier: MIT

//! Seal tests for the A0-3 measure lane
//! (`specs/002-nextgen-analyze/lanes/a0-3-measure.md`), all against M0
//! signal-generator ground truth through the offline harness:
//!
//! * a signal of known width reports the right OBW under each convention,
//!   and the convention is named in the output (FR-AM2);
//! * integrated power matches the D-006 dBFS reference within tolerance
//!   (FR-AM3);
//! * a known duty-cycle burst train recovers duty, burst length and PRI, and
//!   a continuous signal reports "continuous" rather than a fabricated PRI
//!   (FR-AM4);
//! * behavior flags fire on the generator's hopper and chirp and stay quiet
//!   on a steady tone (FR-AM5);
//! * every measurement carries a confidence (NFR-A3) and the whole stage is
//!   deterministic (NFR-A4).

use phosphene_core::Complex;
use phosphene_sources::{HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig};

use super::{
    BinSpan, CenterMethod, MeasureConfig, MeasureError, MeasurementSet, Measurer, ObwConvention,
};
use crate::harness::{OfflineTap, OfflineTapConfig};
use crate::tap::{BandMeta, FrameView, SpectrumTap};

const RATE: f64 = 1_024_000.0;
const FFT: usize = 1024;

/// Run a generator scene through the offline harness and return the last
/// `n_frames` calibrated spectrum frames with their metadata.
fn frames_from_scene(scene: &SigGenConfig, n_frames: usize) -> (FrameView, BandMeta) {
    let mut config = OfflineTapConfig::new(scene.sample_rate_hz);
    config.center_freq_hz = scene.center_freq_hz;
    config.fft_size = FFT;
    config.spectrum_history_frames = n_frames;
    let mut tap = OfflineTap::new(config).unwrap();

    let mut gen = SigGen::new(scene.clone()).unwrap();
    let mut chunk = vec![Complex::new(0.0f32, 0.0); 1000];
    let mut remaining = n_frames * FFT;
    while remaining > 0 {
        let n = remaining.min(chunk.len());
        gen.fill(&mut chunk[..n]);
        tap.feed(&chunk[..n]);
        remaining -= n;
    }

    let mut view = FrameView::new(FFT, n_frames);
    tap.latest_frames(&mut view);
    assert_eq!(view.len(), n_frames);
    (view, tap.meta())
}

/// A flat per-bin noise floor at the given level (use −200 dBFS for
/// noise-free scenes: honest "no floor supplied" bottom-of-scale).
fn flat_floor(db: f32) -> Vec<f32> {
    vec![db; FFT]
}

/// The analytic per-bin floor of a generator noise component: total power
/// spread over N bins, times the window ENBW (the siggen module documents
/// exactly this: `level + 10·log10(ENBW/N)` dBFS/bin).
fn analytic_floor(total_dbfs: f32, enbw_bins: f64) -> Vec<f32> {
    flat_floor(total_dbfs + (10.0 * (enbw_bins / FFT as f64).log10()) as f32)
}

/// A span of bins centered on a frequency offset, ± `half` bins.
fn span_around(offset_hz: f64, half: usize) -> BinSpan {
    let center = (FFT / 2) as i64 + (offset_hz / (RATE / FFT as f64)).round() as i64;
    BinSpan {
        lo: (center - half as i64) as usize,
        hi: (center + half as i64) as usize,
    }
}

/// NFR-A3: every asserted value carries a usable confidence.
fn assert_confident(set: &MeasurementSet) {
    let check = |what: &str, c: f32| {
        assert!(
            c > 0.0 && c <= 1.0,
            "{what} confidence must be in (0, 1], got {c}"
        );
    };
    if let Some(center) = &set.center_hz {
        check("center", center.hz.confidence);
    }
    check("carrier offset", set.carrier_offset_hz.hz.confidence);
    check("headline OBW", set.obw.hz.confidence);
    for alt in &set.obw_alt {
        check("alt OBW", alt.hz.confidence);
    }
    check("power", set.power_dbfs.confidence);
    check("SNR", set.snr_db.confidence);
    check("duty", set.temporal.duty.confidence);
    if let Some(burst) = &set.temporal.mean_burst_s {
        check("mean burst", burst.confidence);
    }
    if let Some(pri) = &set.temporal.period_s {
        check("PRI", pri.confidence);
    }
    check("hopping flag", set.behavior.hopping.confidence);
    check("bursty flag", set.behavior.bursty.confidence);
    check("drifting flag", set.behavior.drifting.confidence);
    if let Some(hop) = &set.behavior.hop {
        for f in &hop.set_hz {
            check("hop frequency", f.confidence);
        }
        if let Some(rate) = &hop.rate_hz {
            check("hop rate", rate.confidence);
        }
    }
}

fn measurer() -> Measurer {
    Measurer::new(MeasureConfig::default()).unwrap()
}

/// FR-AM1 + FR-AM3 + FR-AM4 — the D-006 calibration seal: a full-scale tone
/// integrates to 0 dBFS (the same number v1's cursor readout shows), its
/// center comes back absolute from the tune metadata, and a continuous
/// signal is reported continuous — no burst length, no fabricated PRI.
#[test]
fn full_scale_tone_integrates_to_the_d006_reference() {
    let mut scene = SigGenConfig::new(RATE);
    scene.center_freq_hz = Some(100_000_000.0);
    scene.tones.push(ToneConfig {
        offset_hz: 300_000.0,
        level_dbfs: 0.0,
    });
    let (frames, meta) = frames_from_scene(&scene, 32);

    let set = measurer()
        .measure(
            span_around(300_000.0, 8),
            &frames,
            &meta,
            &flat_floor(-200.0),
        )
        .unwrap();

    let power = set.power_dbfs.value;
    assert!(
        power.abs() < 0.1,
        "full-scale tone integrated to {power} dBFS"
    );

    let offset = set.carrier_offset_hz.hz.value;
    assert!((offset - 300_000.0).abs() < 100.0, "offset {offset} Hz");
    let center = set.center_hz.expect("center is known from tune metadata");
    assert!((center.hz.value - 100_300_000.0).abs() < 100.0);
    assert_eq!(center.method, CenterMethod::PowerCentroid);
    assert_eq!(set.carrier_offset_hz.method, CenterMethod::PowerCentroid);

    assert!((set.temporal.duty.value - 1.0).abs() < 1e-6);
    assert!(
        set.temporal.mean_burst_s.is_none(),
        "continuous: no burst length"
    );
    assert!(
        set.temporal.period_s.is_none(),
        "continuous: no fabricated PRI"
    );
    assert!(
        !set.behavior.bursty.value && !set.behavior.hopping.value && !set.behavior.drifting.value
    );
    assert_confident(&set);
}

/// FR-AM3 — power stays calibrated at lower levels and the reported SNR
/// matches the analytically known signal-to-floor ratio of the scene.
#[test]
fn power_and_snr_match_the_known_scene() {
    const TONE_DBFS: f32 = -20.0;
    const NOISE_DBFS: f32 = -70.0;
    let config = MeasureConfig::default();
    let span = span_around(300_000.0, 8);

    let mut scene = SigGenConfig::new(RATE);
    scene.tones.push(ToneConfig {
        offset_hz: 300_000.0,
        level_dbfs: TONE_DBFS,
    });
    scene.noise = Some(NoiseConfig {
        level_dbfs: NOISE_DBFS,
    });
    let (frames, meta) = frames_from_scene(&scene, 32);
    let floor = analytic_floor(NOISE_DBFS, config.enbw_bins);

    let set = Measurer::new(config.clone())
        .unwrap()
        .measure(span, &frames, &meta, &floor)
        .unwrap();

    let power = set.power_dbfs.value;
    assert!(
        (power - TONE_DBFS).abs() < 0.3,
        "tone at {TONE_DBFS} dBFS integrated to {power} dBFS"
    );

    // Expected SNR: tone power over the integrated in-band floor.
    let floor_bin_lin = 10f64.powf(f64::from(floor[0]) / 10.0);
    let expected_snr = 10.0
        * (10f64.powf(f64::from(TONE_DBFS) / 10.0) / (floor_bin_lin * span.width() as f64)).log10();
    let snr = f64::from(set.snr_db.value);
    assert!(
        (snr - expected_snr).abs() < 2.0,
        "SNR {snr} dB, expected {expected_snr} dB"
    );
    assert_confident(&set);
}

/// FR-AM2 — a signal of known width (an 11-tone comb spanning 11 kHz)
/// reports the right OBW under each convention, each figure naming the
/// convention that produced it.
#[test]
fn known_width_comb_reports_obw_under_each_named_convention() {
    let mut scene = SigGenConfig::new(RATE);
    // 11 tones, 1.1 kHz apart (deliberately off the bin grid so frame
    // averaging decorrelates them), spanning 94.5–105.5 kHz.
    for k in 0..11 {
        scene.tones.push(ToneConfig {
            offset_hz: 94_500.0 + f64::from(k) * 1_100.0,
            level_dbfs: -20.0,
        });
    }
    let (frames, meta) = frames_from_scene(&scene, 32);

    let set = measurer()
        .measure(
            span_around(100_000.0, 25),
            &frames,
            &meta,
            &flat_floor(-200.0),
        )
        .unwrap();

    // The headline figure carries the default ITU convention.
    assert_eq!(set.obw.convention, ObwConvention::DB_DOWN_26);
    let by_convention = |c: ObwConvention| -> f64 {
        if set.obw.convention == c {
            return set.obw.hz.value;
        }
        set.obw_alt
            .iter()
            .find(|o| o.convention == c)
            .unwrap_or_else(|| panic!("{c:?} must be reported"))
            .hz
            .value
    };
    let w99 = by_convention(ObwConvention::POWER_99);
    let w3 = by_convention(ObwConvention::DB_DOWN_3);
    let w20 = by_convention(ObwConvention::DB_DOWN_20);
    let w26 = by_convention(ObwConvention::DB_DOWN_26);

    // True occupied span is 11 kHz; power-fraction and −3 dB hug it, the
    // deep dB-down conventions add the window skirts of the edge tones.
    assert!((10_000.0..=13_500.0).contains(&w99), "99% OBW {w99} Hz");
    assert!((10_500.0..=13_500.0).contains(&w3), "−3 dB width {w3} Hz");
    assert!(
        (11_000.0..=16_500.0).contains(&w26),
        "−26 dBc width {w26} Hz"
    );
    assert!(
        w3 <= w20 && w20 <= w26,
        "deeper conventions can never be narrower: {w3} {w20} {w26}"
    );

    let offset = set.carrier_offset_hz.hz.value;
    assert!(
        (offset - 100_000.0).abs() < 300.0,
        "comb centroid {offset} Hz"
    );
    assert_confident(&set);
}

/// FR-AM4 — the generator's burst train (single-frequency hopper: 5 ms
/// bursts every 15 ms) recovers duty, mean burst duration and PRI.
#[test]
fn burst_train_recovers_duty_burst_length_and_pri() {
    let mut scene = SigGenConfig::new(RATE);
    scene.hoppers.push(HopperConfig {
        offsets_hz: vec![200_000.0],
        dwell_s: 0.005,
        period_s: 0.015,
        level_dbfs: -20.0,
        order: HopOrder::Cycle,
    });
    let (frames, meta) = frames_from_scene(&scene, 64);

    let set = measurer()
        .measure(
            span_around(200_000.0, 8),
            &frames,
            &meta,
            &flat_floor(-200.0),
        )
        .unwrap();

    let duty = f64::from(set.temporal.duty.value);
    assert!((duty - 1.0 / 3.0).abs() < 0.08, "duty {duty}, truth 1/3");

    let burst = set
        .temporal
        .mean_burst_s
        .expect("burst train has a burst length");
    assert!(
        (burst.value - 0.005).abs() < 0.0012,
        "mean burst {} s, truth 0.005 s",
        burst.value
    );

    let pri = set.temporal.period_s.expect("periodic train has a PRI");
    assert!(
        (pri.value - 0.015).abs() < 0.0016,
        "PRI {} s, truth 0.015 s",
        pri.value
    );

    assert!(set.behavior.bursty.value, "burst train must flag bursty");
    assert!(!set.behavior.hopping.value, "single frequency: not hopping");
    assert_confident(&set);
}

/// FR-AM5 — the generator's frequency hopper raises the hopping flag with
/// its hop set and rate resolved.
#[test]
fn generator_hopper_flags_hopping_with_set_and_rate() {
    let offsets = [-200_000.0, 0.0, 200_000.0];
    let mut scene = SigGenConfig::new(RATE);
    scene.hoppers.push(HopperConfig {
        offsets_hz: offsets.to_vec(),
        dwell_s: 0.005,
        period_s: 0.005, // back-to-back: hops with no silence between
        level_dbfs: -20.0,
        order: HopOrder::Cycle,
    });
    let (frames, meta) = frames_from_scene(&scene, 60);

    let set = measurer()
        .measure(span_around(0.0, 220), &frames, &meta, &flat_floor(-200.0))
        .unwrap();

    assert!(set.behavior.hopping.value);
    assert!(
        !set.behavior.drifting.value,
        "hopping must not read as drift"
    );
    let hop = set
        .behavior
        .hop
        .as_ref()
        .expect("hop detail must be resolved");
    assert_eq!(hop.set_hz.len(), offsets.len(), "hop set {:?}", hop.set_hz);
    for (found, truth) in hop.set_hz.iter().zip(offsets.iter()) {
        assert!(
            (found.value - truth).abs() < 1_000.0,
            "hop {} vs {truth}",
            found.value
        );
    }
    let rate = hop.rate_hz.as_ref().expect("hop rate must be resolved");
    assert!(
        (rate.value - 200.0).abs() < 25.0,
        "hop rate {} /s, truth 200 /s",
        rate.value
    );
    assert_confident(&set);
}

/// FR-AM5 — the generator's chirp (a coherently moving center) raises the
/// drifting flag, not hopping.
#[test]
fn generator_chirp_flags_drifting() {
    let mut scene = SigGenConfig::new(RATE);
    scene.chirps.push(phosphene_sources::ChirpConfig {
        start_offset_hz: -100_000.0,
        stop_offset_hz: 100_000.0,
        sweep_time_s: 0.2,
        level_dbfs: -20.0,
    });
    let (frames, meta) = frames_from_scene(&scene, 64);

    let set = measurer()
        .measure(span_around(0.0, 110), &frames, &meta, &flat_floor(-200.0))
        .unwrap();

    assert!(
        set.behavior.drifting.value,
        "chirp must flag a drifting center"
    );
    assert!(!set.behavior.hopping.value);
    assert_confident(&set);
}

/// NFR-A4 — determinism: the same scene, freshly generated and freshly
/// measured, yields a bit-identical MeasurementSet.
#[test]
fn measurement_is_deterministic() {
    let run = || {
        let mut scene = SigGenConfig::new(RATE);
        scene.tones.push(ToneConfig {
            offset_hz: 300_000.0,
            level_dbfs: -20.0,
        });
        scene.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        scene.hoppers.push(HopperConfig {
            offsets_hz: vec![-100_000.0],
            dwell_s: 0.004,
            period_s: 0.012,
            level_dbfs: -25.0,
            order: HopOrder::Cycle,
        });
        let (frames, meta) = frames_from_scene(&scene, 48);
        let floor = analytic_floor(-70.0, 1.5);
        let m = measurer();
        (
            m.measure(span_around(300_000.0, 8), &frames, &meta, &floor)
                .unwrap(),
            m.measure(span_around(-100_000.0, 8), &frames, &meta, &floor)
                .unwrap(),
        )
    };
    let (tone_a, burst_a) = run();
    let (tone_b, burst_b) = run();
    assert_eq!(tone_a, tone_b);
    assert_eq!(burst_a, burst_b);
}

/// The honesty gates: a silent band is a `NoSignal` error, not a number;
/// inconsistent inputs and configurations are named rejections.
#[test]
fn refuses_silence_and_inconsistent_inputs() {
    let scene = SigGenConfig::new(RATE); // silence
    let (frames, meta) = frames_from_scene(&scene, 8);
    let m = measurer();

    assert!(matches!(
        m.measure(span_around(0.0, 8), &frames, &meta, &flat_floor(-200.0)),
        Err(MeasureError::NoSignal { frames_seen: 8 })
    ));

    let short_floor = vec![-200.0f32; FFT - 1];
    assert!(matches!(
        m.measure(span_around(0.0, 8), &frames, &meta, &short_floor),
        Err(MeasureError::InvalidInput(_))
    ));

    let bad_span = BinSpan { lo: 10, hi: FFT };
    assert!(matches!(
        m.measure(bad_span, &frames, &meta, &flat_floor(-200.0)),
        Err(MeasureError::InvalidInput(_))
    ));

    let empty = FrameView::new(FFT, 4);
    assert!(matches!(
        m.measure(span_around(0.0, 8), &empty, &meta, &flat_floor(-200.0)),
        Err(MeasureError::InvalidInput(_))
    ));

    let bad_config = MeasureConfig {
        enbw_bins: 0.5,
        ..MeasureConfig::default()
    };
    assert!(matches!(
        Measurer::new(bad_config),
        Err(MeasureError::InvalidConfig(_))
    ));
}

/// The convention plumbing end to end: a non-default headline convention is
/// the one named on the headline figure, and the stated center method is
/// honored.
#[test]
fn configured_convention_and_center_method_are_honored() {
    let mut scene = SigGenConfig::new(RATE);
    scene.tones.push(ToneConfig {
        offset_hz: 250_000.0,
        level_dbfs: -20.0,
    });
    let (frames, meta) = frames_from_scene(&scene, 16);
    let span = span_around(250_000.0, 8);
    let floor = flat_floor(-200.0);

    let config = MeasureConfig {
        headline_obw: ObwConvention::POWER_99,
        center_method: CenterMethod::ObwMidpoint,
        ..MeasureConfig::default()
    };
    let set = Measurer::new(config)
        .unwrap()
        .measure(span, &frames, &meta, &floor)
        .unwrap();

    assert_eq!(set.obw.convention, ObwConvention::POWER_99);
    let offset = set.carrier_offset_hz.hz.value;
    assert!(
        (offset - 250_000.0).abs() < 600.0,
        "midpoint center {offset} Hz"
    );
    // The value itself says which method produced it (FR-AM1 "state which").
    assert_eq!(set.carrier_offset_hz.method, CenterMethod::ObwMidpoint);
}

/// R2 blocker 2 seal — the center method survives serialization: a consumer
/// of the serialized set can tell PowerCentroid from ObwMidpoint without
/// reaching back into the producing configuration.
#[test]
fn center_method_survives_serialization() {
    let mut scene = SigGenConfig::new(RATE);
    scene.center_freq_hz = Some(100_000_000.0);
    scene.tones.push(ToneConfig {
        offset_hz: 250_000.0,
        level_dbfs: -20.0,
    });
    let (frames, meta) = frames_from_scene(&scene, 16);
    let span = span_around(250_000.0, 8);
    let floor = flat_floor(-200.0);

    for method in [CenterMethod::PowerCentroid, CenterMethod::ObwMidpoint] {
        let config = MeasureConfig {
            center_method: method,
            ..MeasureConfig::default()
        };
        let set = Measurer::new(config)
            .unwrap()
            .measure(span, &frames, &meta, &floor)
            .unwrap();

        let json = serde_json::to_string(&set).unwrap();
        let back: MeasurementSet = serde_json::from_str(&json).unwrap();
        // The provenance — the blocker — must survive exactly. (Values are
        // compared with a tolerance rather than bitwise: serde_json's
        // default float parsing is not ULP-exact without its
        // `float_roundtrip` feature, and that dev-dependency is A0-0's
        // manifest line, which D-019 forbids this lane to edit.)
        assert_eq!(back.carrier_offset_hz.method, method);
        assert_eq!(
            back.center_hz.expect("center known").method,
            method,
            "method must survive on the absolute center too"
        );
        assert_eq!(back.obw.convention, set.obw.convention);
        assert!(
            (back.carrier_offset_hz.hz.value - set.carrier_offset_hz.hz.value).abs() < 1e-6,
            "offset value must survive"
        );
        assert_eq!(
            back.carrier_offset_hz.hz.confidence,
            set.carrier_offset_hz.hz.confidence
        );
    }
}

/// R2 blocker 1+2 seal — NO field of a serialized MeasurementSet is an
/// undefended assertion. Walking the serialized JSON (rather than the Rust
/// struct) makes this a regression guard: any field added later without
/// either a confidence or an explicit not-applicable (`null`) fails here.
///
/// Rules: every numeric or boolean leaf must live inside a
/// `{value, confidence}` pair with confidence in (0, 1]; `null` is an
/// explicit not-applicable; strings are labels; subtrees under `convention`
/// or `method` are provenance discriminators (the FR-AM2/FR-AM1 "state
/// which"), not measurements.
#[test]
fn no_field_of_a_serialized_set_is_undefended() {
    fn assert_defended(v: &serde_json::Value, path: &str) {
        use serde_json::Value;
        match v {
            Value::Object(map) => {
                if map.contains_key("value") {
                    let c = map
                        .get("confidence")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or_else(|| panic!("{path}: value without confidence"));
                    assert!(c > 0.0 && c <= 1.0, "{path}: confidence {c} not in (0, 1]");
                    return;
                }
                for (k, val) in map {
                    if k == "convention" || k == "method" {
                        continue; // provenance, not a measurement
                    }
                    assert_defended(val, &format!("{path}.{k}"));
                }
            }
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    assert_defended(item, &format!("{path}[{i}]"));
                }
            }
            Value::Null | Value::String(_) => {}
            Value::Bool(b) => panic!("{path}: bare bool {b} — undefended assertion"),
            Value::Number(n) => panic!("{path}: bare number {n} — undefended assertion"),
        }
    }

    // A hopper scene with a known band center exercises every branch of the
    // set: absolute center, hop detail with set and rate, all three flags.
    let mut scene = SigGenConfig::new(RATE);
    scene.center_freq_hz = Some(100_000_000.0);
    scene.hoppers.push(HopperConfig {
        offsets_hz: vec![-200_000.0, 0.0, 200_000.0],
        dwell_s: 0.005,
        period_s: 0.005,
        level_dbfs: -20.0,
        order: HopOrder::Cycle,
    });
    let (frames, meta) = frames_from_scene(&scene, 60);
    let hopper_set = measurer()
        .measure(span_around(0.0, 220), &frames, &meta, &flat_floor(-200.0))
        .unwrap();
    assert!(
        hopper_set.behavior.hop.is_some(),
        "walker must see hop detail"
    );
    assert_defended(
        &serde_json::to_value(&hopper_set).unwrap(),
        "hopper MeasurementSet",
    );

    // A bursty single-frequency scene exercises the temporal branch.
    let mut scene = SigGenConfig::new(RATE);
    scene.hoppers.push(HopperConfig {
        offsets_hz: vec![200_000.0],
        dwell_s: 0.005,
        period_s: 0.015,
        level_dbfs: -20.0,
        order: HopOrder::Cycle,
    });
    let (frames, meta) = frames_from_scene(&scene, 64);
    let burst_set = measurer()
        .measure(
            span_around(200_000.0, 8),
            &frames,
            &meta,
            &flat_floor(-200.0),
        )
        .unwrap();
    assert!(
        burst_set.temporal.period_s.is_some(),
        "walker must see a PRI"
    );
    assert_defended(
        &serde_json::to_value(&burst_set).unwrap(),
        "burst MeasurementSet",
    );
}
