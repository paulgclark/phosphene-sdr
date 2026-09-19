// SPDX-License-Identifier: MIT

//! Seal tests for the A0-0 skeleton: the shared-type contract round-trips,
//! the annotation feed behaves as a snapshot swap, and the siggen/file
//! harness produces calibrated frames from known-ground-truth generator
//! input (lane spec `specs/002-nextgen-analyze/lanes/a0-analyze-skeleton.md`).

use phosphene_analyze::harness::{load_cf32, write_cf32, OfflineTap, OfflineTapConfig};
use phosphene_analyze::{
    AnalysisTap, Annotation, BehaviorFlags, CenterFrequency, CenterMethod, Classification,
    Detection, FrameView, FreqSpanHz, IqBuf, Measured, MeasurementSet, ModulationClass,
    ObwConvention, OccupiedBandwidth, SignatureCandidate, SpectrumTap, Temporal, TimeSpan, Track,
    TrackId, TrackState,
};
use phosphene_core::fftshift_index;
use phosphene_sources::{SigGen, SigGenConfig, ToneConfig};

/// A fully-populated Track exercising every field of the shared-type
/// contract, including the phase A1+/A3 fields.
fn full_track() -> Track {
    Track {
        id: TrackId(7),
        state: TrackState::Confirmed,
        center_offset_hz: 300_000.0,
        bandwidth_hz: 12_500.0,
        power_dbfs: -20.0,
        snr_db: 35.0,
        born_frame: 100,
        last_seen_frame: 250,
        measurements: Some(MeasurementSet {
            center_hz: Some(CenterFrequency {
                hz: Measured::new(915_300_000.0, 0.95),
                method: CenterMethod::PowerCentroid,
            }),
            carrier_offset_hz: CenterFrequency {
                hz: Measured::new(300_000.0, 0.95),
                method: CenterMethod::PowerCentroid,
            },
            obw: OccupiedBandwidth {
                hz: Measured::new(12_500.0, 0.9),
                convention: ObwConvention::DB_DOWN_26,
            },
            obw_alt: vec![OccupiedBandwidth {
                hz: Measured::new(11_000.0, 0.9),
                convention: ObwConvention::POWER_99,
            }],
            power_dbfs: Measured::new(-20.0, 0.9),
            snr_db: Measured::new(35.0, 0.9),
            temporal: Temporal {
                duty: Measured::new(0.4, 0.85),
                mean_burst_s: Some(Measured::new(0.08, 0.8)),
                period_s: Some(Measured::new(1.9, 0.8)),
            },
            behavior: BehaviorFlags {
                hopping: Measured::new(false, 0.9),
                bursty: Measured::new(true, 0.85),
                drifting: Measured::new(false, 0.9),
                hop: None,
            },
        }),
        class: Some(Classification {
            class: ModulationClass::Fsk {
                levels: 2,
                deviation_hz: Measured::new(4_500.0, 0.8),
            },
            symbol_rate_bd: Some(Measured::new(9_600.0, 0.75)),
            confidence: 0.8,
            evidence: vec![
                "constant envelope".to_string(),
                "two instantaneous-frequency modes".to_string(),
            ],
        }),
        signatures: vec![SignatureCandidate {
            label: "POCSAG paging".to_string(),
            score: 0.8,
            source: "ITU-R M.584 (illustrative)".to_string(),
        }],
    }
}

#[test]
fn shared_types_round_trip_through_serde() {
    let detection = Detection {
        bin_lo: 810,
        bin_hi: 814,
        frame_lo: 100,
        frame_hi: 109,
        centroid_bin: 812.3,
        peak_dbfs: -20.0,
        power_dbfs: -19.2,
        snr_db: 41.0,
    };
    let json = serde_json::to_string(&detection).unwrap();
    assert_eq!(serde_json::from_str::<Detection>(&json).unwrap(), detection);

    let track = full_track();
    let json = serde_json::to_string_pretty(&track).unwrap();
    assert_eq!(serde_json::from_str::<Track>(&json).unwrap(), track);

    let annotation = Annotation {
        track: TrackId(7),
        span: FreqSpanHz {
            low_hz: 915_293_750.0,
            high_hz: 915_306_250.0,
        },
        anchor_hz: 915_300_000.0,
        label: "FSK ~9.6 kBd — consistent with POCSAG".to_string(),
        detail: vec!["duty ≈ 40%, period ≈ 1.9 s".to_string()],
        confidence: 0.8,
        strength_db: 35.0,
    };
    let json = serde_json::to_string(&annotation).unwrap();
    assert_eq!(
        serde_json::from_str::<Annotation>(&json).unwrap(),
        annotation
    );
}

/// The harness turns a known generator scene into calibrated spectrum
/// frames: a −20 dBFS tone at +300 kHz must read −20 dBFS in exactly its
/// DC-centered bin (the D-006 calibration the whole detection chain leans
/// on).
#[test]
fn harness_produces_calibrated_frames_from_siggen_ground_truth() {
    const RATE: f64 = 1_024_000.0;
    const FFT: usize = 1024;
    const TONE_OFFSET_HZ: f64 = 300_000.0; // exactly bin +300 at 1 kHz/bin
    const TONE_LEVEL_DBFS: f32 = -20.0;

    let mut scene = SigGenConfig::new(RATE);
    scene.tones.push(ToneConfig {
        offset_hz: TONE_OFFSET_HZ,
        level_dbfs: TONE_LEVEL_DBFS,
    });
    let mut gen = SigGen::new(scene).unwrap();

    let mut config = OfflineTapConfig::new(RATE);
    config.fft_size = FFT;
    config.spectrum_history_frames = 4;
    let mut tap = OfflineTap::new(config).unwrap();

    // Feed 8 frames' worth in odd-sized chunks (chunking must not matter).
    let mut remaining = 8 * FFT;
    let mut chunk = vec![phosphene_core::Complex::new(0.0f32, 0.0); 1000];
    while remaining > 0 {
        let n = remaining.min(chunk.len());
        gen.fill(&mut chunk[..n]);
        tap.feed(&chunk[..n]);
        remaining -= n;
    }
    assert_eq!(tap.frames_produced(), 8);

    let meta = tap.meta();
    assert_eq!(meta.bins, FFT);
    assert!((meta.bin_hz - 1000.0).abs() < 1e-9);
    assert!((meta.frame_dt_s - FFT as f64 / RATE).abs() < 1e-12);

    let mut view = FrameView::new(FFT, 4);
    tap.latest_frames(&mut view);
    assert_eq!(view.len(), 4);
    // The 4 most recent of 8 produced frames are seqs 4..=7.
    assert_eq!(view.seq(0), 4);
    assert_eq!(view.seq(3), 7);

    let expected_bin = fftshift_index(300, FFT);
    for i in 0..view.len() {
        let frame = view.frame(i);
        let (peak_bin, &peak) = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!(peak_bin, expected_bin, "tone must land in its bin");
        assert!(
            (peak - TONE_LEVEL_DBFS).abs() < 0.1,
            "tone read {peak} dBFS, expected {TONE_LEVEL_DBFS}"
        );
        // Same math as BandMeta: the peak bin maps back to +300 kHz.
        assert!((meta.bin_offset_hz(peak_bin as f64) - TONE_OFFSET_HZ).abs() < 1e-9);
    }
}

/// The AnalysisTap side of the harness returns exactly the most recent raw
/// samples that were fed — verified against a second, independent generator
/// run (the siggen determinism contract).
#[test]
fn harness_snapshot_returns_the_raw_iq_tail() {
    const RATE: f64 = 1_024_000.0;
    let scene = {
        let mut s = SigGenConfig::new(RATE);
        s.tones.push(ToneConfig {
            offset_hz: -100_000.0,
            level_dbfs: -10.0,
        });
        s
    };

    let mut config = OfflineTapConfig::new(RATE);
    config.iq_history_s = 0.01;
    let mut tap = OfflineTap::new(config).unwrap();
    assert!((tap.caps().history_s - 0.01).abs() < 1e-12);

    const FED: usize = 8192;
    let mut fed = vec![phosphene_core::Complex::new(0.0f32, 0.0); FED];
    SigGen::new(scene.clone()).unwrap().fill(&mut fed);
    tap.feed(&fed);

    const WANT: usize = 4096;
    let mut buf = IqBuf::default();
    let meta = tap
        .snapshot(TimeSpan::last(WANT as f64 / RATE), &mut buf)
        .unwrap();
    assert_eq!(buf.samples.len(), WANT);
    assert!((meta.duration_s - WANT as f64 / RATE).abs() < 1e-12);

    // Ground truth: an identical generator's stream, last WANT samples.
    let mut truth = vec![phosphene_core::Complex::new(0.0f32, 0.0); FED];
    SigGen::new(scene).unwrap().fill(&mut truth);
    assert_eq!(buf.samples, truth[FED - WANT..].to_vec());
}

/// The file path: generator samples written as a raw `.cfile` and loaded
/// back feed the harness identically to the live fill — recorded IQ and
/// synthetic IQ share one code path.
#[test]
fn harness_reads_recorded_iq_files() {
    const RATE: f64 = 1_024_000.0;
    let mut scene = SigGenConfig::new(RATE);
    scene.tones.push(ToneConfig {
        offset_hz: 250_000.0,
        level_dbfs: -30.0,
    });
    let mut gen = SigGen::new(scene).unwrap();
    let mut samples = vec![phosphene_core::Complex::new(0.0f32, 0.0); 2048];
    gen.fill(&mut samples);

    let path = std::env::temp_dir().join(format!(
        "phosphene-analyze-skeleton-{}.cfile",
        std::process::id()
    ));
    write_cf32(&path, &samples).unwrap();
    let loaded = load_cf32(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded, samples);

    let mut config = OfflineTapConfig::new(RATE);
    config.fft_size = 1024;
    let mut tap = OfflineTap::new(config).unwrap();
    tap.feed(&loaded);
    assert_eq!(tap.frames_produced(), 2);

    let mut view = FrameView::new(1024, 2);
    tap.latest_frames(&mut view);
    let frame = view.frame(1);
    let peak_bin = frame
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(peak_bin, fftshift_index(250, 1024));
}
