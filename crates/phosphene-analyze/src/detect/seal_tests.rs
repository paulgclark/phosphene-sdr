// SPDX-License-Identifier: MIT

//! Lane A0-1 seal tests: the detection pipeline against M0 generator scenes
//! with known ground truth, through the offline harness (spec §2.3) —
//! exactly the path live v1 frames will take.
//!
//! What is verified, per the lane spec:
//!
//! * the **measured false-alarm rate** on noise-only input matches the
//!   configured Pfa within ±25% (the stated tolerance; statistical spread
//!   over ~10⁶ cells is a few percent, bin-to-bin mask correlation the
//!   rest). Counted in steady state — after the noise-floor window fills —
//!   since the D-023 detector thresholds against that floor and warmup
//!   thresholds are deliberately conservative;
//! * the **detection-probability curve** versus SNR is reported (printed),
//!   with only its endpoints and monotonicity asserted — the SNR where
//!   detection becomes unreliable is stated, not assumed;
//! * a **weak signal adjacent to a strong occupant is detected**, at a
//!   separation where a plain cell-average reference demonstrably masks it
//!   (both are measured side by side) — the D-023 property;
//! * the **robust noise floor** on a band with a strong continuous occupant
//!   and a bursty hopper stays on the generator's configured level;
//! * the whole pipeline is **deterministic** for a fixed seed and config.

use phosphene_core::Complex;
use phosphene_sources::{
    ChirpConfig, HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig,
};

use crate::detect::{Detection, Detector, DetectorConfig};
use crate::harness::{OfflineTap, OfflineTapConfig};
use crate::tap::{FrameView, SpectrumTap};

const RATE: f64 = 2_048_000.0;
const BINS: usize = 1024;
const BIN_HZ: f64 = RATE / BINS as f64;
const NOISE_TOTAL_DBFS: f32 = -70.0;
/// Frames until the default noise-floor window is full and per-cell
/// statistics are in steady state.
const WARMUP: usize = 64;

/// Displayed per-bin level of the scene's noise floor under the harness's
/// 1024-bin Hann analyzer: `level + 10·log10(ENBW/N)` (D-006 convention,
/// periodic-Hann ENBW = 1.5) ⇒ −98.34 dBFS/bin.
fn per_bin_floor() -> f32 {
    NOISE_TOTAL_DBFS + (10.0 * (1.5f64 / BINS as f64).log10()) as f32
}

/// The DC-centered bin a generator offset lands on (offsets in these tests
/// are exact bin multiples, so tones are bin-centered).
fn bin_of(offset_hz: f64) -> usize {
    (BINS as f64 / 2.0 + offset_hz / BIN_HZ).round() as usize
}

fn noise_scene(seed: u64) -> SigGenConfig {
    let mut c = SigGenConfig::new(RATE);
    c.seed = seed;
    c.noise = Some(NoiseConfig {
        level_dbfs: NOISE_TOTAL_DBFS,
    });
    c
}

/// Feed `frames` frames of the scene through the offline harness into the
/// detector, invoking `each(detector, frame_index, dbfs_frame)` after every
/// frame; returns the completed detections (call `det.finish()` for
/// still-open ones).
fn drive(
    scene: &SigGenConfig,
    frames: usize,
    det: &mut Detector,
    mut each: impl FnMut(&Detector, usize, &[f32]),
) -> Vec<Detection> {
    let mut gen = SigGen::new(scene.clone()).unwrap();
    let tap_cfg = OfflineTapConfig::new(RATE);
    let mut tap = OfflineTap::new(tap_cfg).unwrap();
    let mut chunk = vec![Complex::new(0.0f32, 0.0); BINS];
    let mut view = FrameView::new(BINS, 1);
    let mut out = Vec::new();
    for i in 0..frames {
        gen.fill(&mut chunk);
        tap.feed(&chunk);
        tap.latest_frames(&mut view);
        out.extend(det.push_frame(view.frame(0), view.seq(0)));
        each(det, i, view.frame(0));
    }
    out
}

/// Measured steady-state mask-cell false-alarm rate on a noise-only scene.
fn measure_pfa(pfa: f64, frames: usize, seed: u64) -> f64 {
    let mut cfg = DetectorConfig::new();
    cfg.cfar.pfa = pfa;
    let mut det = Detector::new(BINS, cfg).unwrap();
    let mut hits = 0u64;
    drive(&noise_scene(seed), frames, &mut det, |d, i, _| {
        if i >= WARMUP {
            hits += d.last_mask().iter().filter(|&&m| m).count() as u64;
        }
    });
    hits as f64 / ((frames - WARMUP) * BINS) as f64
}

#[test]
fn false_alarm_rate_matches_configured_pfa_on_noise_only_input() {
    for (pfa, frames, seed) in [(1e-2, 300, 11), (1e-3, 1000, 12)] {
        let measured = measure_pfa(pfa, frames, seed);
        println!("Pfa configured {pfa:.0e}: measured {measured:.3e} over {frames} frames");
        assert!(
            (measured / pfa - 1.0).abs() < 0.25,
            "measured false-alarm rate {measured:.3e} outside ±25% of configured {pfa:.0e}"
        );
    }
}

#[test]
fn detection_probability_curve_versus_snr_is_reported() {
    // Eight bin-centered tones, one per SNR step, far enough apart that no
    // tone sits in another's CFAR reference window (128 bins vs a 50-bin
    // reference span).
    let snrs_db: [f32; 8] = [-5.0, -2.0, 1.0, 4.0, 7.0, 10.0, 13.0, 20.0];
    let bins: Vec<usize> = (0..8).map(|i| 64 + 128 * i).collect();
    let mut scene = noise_scene(21);
    for (i, &snr) in snrs_db.iter().enumerate() {
        scene.tones.push(ToneConfig {
            offset_hz: (bins[i] as f64 - BINS as f64 / 2.0) * BIN_HZ,
            level_dbfs: per_bin_floor() + snr,
        });
    }
    let frames = 300usize;
    let cfg = DetectorConfig::new();
    let pfa = cfg.cfar.pfa;
    let mut det = Detector::new(BINS, cfg).unwrap();
    let mut hits = [0u32; 8];
    drive(&scene, frames, &mut det, |d, i, _| {
        if i >= WARMUP {
            for (h, &b) in hits.iter_mut().zip(&bins) {
                *h += u32::from(d.last_mask()[b]);
            }
        }
    });
    let pd: Vec<f64> = hits
        .iter()
        .map(|&h| h as f64 / (frames - WARMUP) as f64)
        .collect();

    println!("detection probability vs per-bin SNR (CA-CFAR, Pfa {pfa:.0e}, {frames} frames):");
    for (&snr, &p) in snrs_db.iter().zip(&pd) {
        println!("  SNR {snr:+5.1} dB -> Pd {p:.3}");
    }
    match snrs_db
        .iter()
        .zip(&pd)
        .find(|&(_, &p)| p >= 0.9)
        .map(|(&s, _)| s)
    {
        Some(s) => println!("  detection becomes unreliable below ≈ {s} dB per-bin SNR"),
        None => println!("  no tested SNR reached Pd 0.9"),
    }

    assert!(
        pd[7] >= 0.99,
        "Pd at 20 dB SNR should be ≈ 1, got {}",
        pd[7]
    );
    assert!(
        pd[0] <= 0.05,
        "Pd at −5 dB SNR should be near the false-alarm rate, got {}",
        pd[0]
    );
    for w in pd.windows(2) {
        assert!(
            w[1] >= w[0] - 0.1,
            "Pd curve should not decrease with SNR: {pd:?}"
        );
    }
}

#[test]
fn weak_signal_next_to_a_strong_occupant_is_detected_where_cell_averaging_masks_it() {
    // D-023's falsifiable test. A −20 dBFS occupant (78 dB over the floor)
    // sits 15 bins from a weak 18 dB-SNR tone — its mainlobe lands on one
    // of the weak bin's reference cells (distance 14 = guard 2 + stride
    // 3·4). A plain cell average over instantaneous reference powers
    // absorbs the occupant into the threshold and masks the weak tone; the
    // shipped robust-floor reference must not. Both are measured on the
    // same frames so the comparison proves the change did something.
    let strong_bin = bin_of(376_000.0);
    let weak_bin = strong_bin + 15;
    let weak_snr = 18.0f32;
    let mut scene = noise_scene(81);
    scene.tones.push(ToneConfig {
        offset_hz: 376_000.0,
        level_dbfs: -20.0,
    });
    scene.tones.push(ToneConfig {
        offset_hz: (weak_bin as f64 - BINS as f64 / 2.0) * BIN_HZ,
        level_dbfs: per_bin_floor() + weak_snr,
    });
    let frames = 300usize;
    let cfg = DetectorConfig::new();
    let (guard, stride, refs, pfa) = (
        cfg.cfar.guard_cells,
        cfg.cfar.ref_stride,
        cfg.cfar.ref_cells,
        cfg.cfar.pfa,
    );
    let mut det = Detector::new(BINS, cfg).unwrap();
    let (mut robust_hits, mut strong_hits, mut ca_hits, mut counted) = (0u32, 0u32, 0u32, 0u32);
    drive(&scene, frames, &mut det, |d, i, frame| {
        if i < WARMUP {
            return;
        }
        counted += 1;
        robust_hits += u32::from(d.last_mask()[weak_bin]);
        strong_hits += u32::from(d.last_mask()[strong_bin]);
        // Plain CA-CFAR baseline at the weak bin: same reference geometry,
        // textbook α = N(Pfa^(−1/N) − 1), but over instantaneous frame
        // powers — the masking-prone form D-023 rules out.
        let (mut sum, mut n) = (0.0f64, 0usize);
        for j in 1..=refs {
            let d = guard + stride * j;
            for b in [weak_bin - d, weak_bin + d] {
                sum += 10f64.powf(frame[b] as f64 / 10.0);
                n += 1;
            }
        }
        let alpha = n as f64 * (pfa.powf(-1.0 / n as f64) - 1.0);
        let cut = 10f64.powf(frame[weak_bin] as f64 / 10.0);
        ca_hits += u32::from(cut > alpha / n as f64 * sum);
    });
    let pd_robust = robust_hits as f64 / counted as f64;
    let pd_ca = ca_hits as f64 / counted as f64;
    println!("weak tone ({weak_snr} dB SNR) 15 bins from a −20 dBFS occupant:");
    println!("  robust-floor reference (shipped): Pd {pd_robust:.3}");
    println!("  plain cell-average baseline:      Pd {pd_ca:.3}");
    assert!(
        pd_ca < 0.1,
        "cell-average baseline no longer masks here (Pd {pd_ca:.3}) — \
         the separation stopped demonstrating the failure"
    );
    assert!(
        pd_robust >= 0.9,
        "weak signal masked by the strong occupant: Pd {pd_robust:.3}"
    );
    assert!(
        strong_hits == counted,
        "the strong occupant itself must stay detected"
    );
}

#[test]
fn robust_floor_is_not_pulled_up_by_strong_occupants() {
    // A −20 dBFS continuous tone (78 dB over the floor) and a 33%-duty
    // hopper share the band. The floor estimate away from the tone's
    // three-bin mainlobe must stay on the generator's configured level —
    // including in the hopper's own bins.
    let tone_bin = bin_of(600_000.0);
    let hop_bins = [bin_of(-600_000.0), bin_of(-200_000.0)];
    let mut scene = noise_scene(31);
    scene.tones.push(ToneConfig {
        offset_hz: 600_000.0,
        level_dbfs: -20.0,
    });
    scene.hoppers.push(HopperConfig {
        offsets_hz: vec![-600_000.0, -200_000.0],
        dwell_s: 0.005,
        period_s: 0.015,
        level_dbfs: -25.0,
        order: HopOrder::Cycle,
    });
    let mut det = Detector::new(BINS, DetectorConfig::new()).unwrap();
    drive(&scene, 200, &mut det, |_, _, _| {});

    let truth = per_bin_floor();
    let floor = det.noise_floor();
    let clear: Vec<f32> = (0..BINS)
        .filter(|k| k.abs_diff(tone_bin) > 3)
        .map(|k| floor[k])
        .collect();
    let mean = clear.iter().sum::<f32>() / clear.len() as f32;
    assert!(
        (mean - truth).abs() < 0.6,
        "band-average floor {mean:.2} dBFS vs generator truth {truth:.2}"
    );
    let max = clear.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        max < truth + 6.0,
        "some bin's floor was pulled up to {max:.2} dBFS (truth {truth:.2})"
    );
    for &hb in &hop_bins {
        assert!(
            (floor[hb] - truth).abs() < 4.0,
            "floor in hopper bin {hb} reads {:.2} dBFS vs truth {truth:.2}",
            floor[hb]
        );
    }
    let hop_mean = hop_bins.iter().map(|&b| floor[b]).sum::<f32>() / hop_bins.len() as f32;
    assert!(
        (hop_mean - truth).abs() < 2.0,
        "mean hopper-bin floor {hop_mean:.2} dBFS vs truth {truth:.2}"
    );
    // A continuous occupant is indistinguishable from noise at its level by
    // any per-bin temporal statistic: its own bin reads the occupant, and
    // only that bin's neighborhood (asserted clear above) is affected.
    assert!(
        floor[tone_bin] > -60.0,
        "expected the tone's own bin to read the occupant, got {:.2}",
        floor[tone_bin]
    );
}

#[test]
fn hopper_bursts_form_correctly_bounded_blobs() {
    // Cycle-order hopper: 10-frame bursts alternating between two known
    // bins every 30 frames, starting at frame 0. Over 200 frames, bursts
    // start at frames 0, 30, …, 180: seven complete blobs, alternating
    // A B A B A B A.
    let a = bin_of(-600_000.0);
    let b = bin_of(200_000.0);
    let mut scene = noise_scene(41);
    scene.hoppers.push(HopperConfig {
        offsets_hz: vec![-600_000.0, 200_000.0],
        dwell_s: 0.005,  // 10 frames at 1024 samples / 2.048 MS/s
        period_s: 0.015, // 30 frames
        level_dbfs: -25.0,
        order: HopOrder::Cycle,
    });
    let mut cfg = DetectorConfig::new();
    // Correlated neighbor bins make false alarms cluster, so lone-cell and
    // even few-cell noise blobs occur; 8 cells is far outside that while a
    // burst spans ≥ 30 (3 bins × 10 frames).
    cfg.min_blob_cells = 8;
    let mut det = Detector::new(BINS, cfg).unwrap();
    let mut dets = drive(&scene, 200, &mut det, |_, _, _| {});
    dets.extend(det.finish());
    dets.sort_by_key(|d| d.frame_lo);

    let snr_truth = -25.0 - per_bin_floor(); // ≈ 73.3 dB
    assert_eq!(dets.len(), 7, "expected 7 complete bursts: {dets:?}");
    for (i, d) in dets.iter().enumerate() {
        let expect_bin = if i % 2 == 0 { a } else { b };
        let start = i as u64 * 30;
        assert!(
            d.frame_lo.abs_diff(start) <= 1 && (9..=11).contains(&(d.frame_hi - d.frame_lo + 1)),
            "burst {i} time extent off: frames {}..={} (expected ≈{start}..={})",
            d.frame_lo,
            d.frame_hi,
            start + 9
        );
        assert!(
            (d.centroid_bin - expect_bin as f64).abs() < 1.5,
            "burst {i} centroid {:.1} vs expected bin {expect_bin}",
            d.centroid_bin
        );
        assert!(
            d.bin_hi - d.bin_lo <= 4,
            "burst {i} frequency extent too wide: {}..={}",
            d.bin_lo,
            d.bin_hi
        );
        assert!(
            (d.peak_dbfs - -25.0).abs() < 1.5,
            "burst {i} peak {} dBFS vs configured −25",
            d.peak_dbfs
        );
        // Burst 0 lives entirely inside the noise-floor warmup (its bin's
        // window holds nothing but the burst itself), so its SNR figure is
        // warmup-coarse by construction; every later burst must carry the
        // ground-truth SNR.
        if i > 0 {
            assert!(
                (d.snr_db - snr_truth).abs() < 3.0,
                "burst {i} SNR {} dB vs ground truth {snr_truth:.1}",
                d.snr_db
            );
        }
    }
}

#[test]
fn a_sweeping_chirp_stays_one_connected_component() {
    // 1 bin/frame sweep: strictly diagonal steps on the (freq, time) grid —
    // exactly what 8-connectivity exists for.
    let mut scene = noise_scene(51);
    scene.chirps.push(ChirpConfig {
        start_offset_hz: -400_000.0,
        stop_offset_hz: 400_000.0,
        sweep_time_s: 0.2, // 400 frames full sweep; we feed the first 100
        level_dbfs: -30.0,
    });
    let mut cfg = DetectorConfig::new();
    cfg.min_blob_cells = 8;
    let mut det = Detector::new(BINS, cfg).unwrap();
    let mut dets = drive(&scene, 100, &mut det, |_, _, _| {});
    dets.extend(det.finish());
    assert_eq!(dets.len(), 1, "chirp fragmented: {dets:?}");
    let d = &dets[0];
    let start_bin = bin_of(-400_000.0);
    assert!(
        d.frame_lo <= 1 && d.frame_hi >= 98,
        "chirp time extent lost"
    );
    // The extent may run a few bins past the swept range: at 65 dB SNR the
    // chirp's own window skirts clear the threshold (the D-023 reference
    // does not let the chirp mask its own sidelobes the way a cell average
    // did), so allow the skirt width on both edges.
    assert!(
        d.bin_lo.abs_diff(start_bin) <= 10 && d.bin_hi.abs_diff(start_bin + 100) <= 10,
        "chirp swept {}..={}, expected ≈{}..={}",
        d.bin_lo,
        d.bin_hi,
        start_bin,
        start_bin + 100
    );
}

#[test]
fn pipeline_is_deterministic_for_a_fixed_seed() {
    let run = || {
        let mut scene = noise_scene(61);
        scene.tones.push(ToneConfig {
            offset_hz: 300_000.0,
            level_dbfs: -60.0,
        });
        scene.hoppers.push(HopperConfig {
            offsets_hz: vec![-500_000.0, 100_000.0, 400_000.0],
            dwell_s: 0.004,
            period_s: 0.010,
            level_dbfs: -40.0,
            order: HopOrder::Pseudorandom,
        });
        let mut cfg = DetectorConfig::new();
        cfg.min_blob_cells = 2;
        let mut det = Detector::new(BINS, cfg).unwrap();
        let mut dets = drive(&scene, 150, &mut det, |_, _, _| {});
        dets.extend(det.finish());
        (dets, det.noise_floor().to_vec())
    };
    let (d1, f1) = run();
    let (d2, f2) = run();
    assert!(!d1.is_empty(), "scene should produce detections");
    assert_eq!(d1, d2, "detections must be bit-identical across runs");
    assert_eq!(f1, f2, "noise floor must be bit-identical across runs");
}

/// D-126/AN-4: the owner, on a real radio, "the DC spike in the center is
/// usually tagged as a bunch of signals". Reproduced with the generator's
/// DC tone (`offset_hz: 0.0`) standing in for LO leakage — the built-in
/// fixture the lane brief names for exactly this. **This is also the
/// negative control**: commenting out `Detector::push_frame`'s
/// `exclude_dc_band` call turns this red deterministically, on any
/// machine, unloaded, because the DC tone is far above the CFAR threshold
/// at every level tested (`DetectorConfig::new`'s doc has the width
/// measurement). Quoted red, captured by commenting out that one line and
/// rerunning this test before restoring it: of 159 detections over the
/// 150-frame scene (the other
/// 158 are ordinary Pfa background chatter across the band — expected
/// count at Pfa 1e-3 over 1024 bins × 150 frames ≈ 154), exactly one
/// touches the DC band — `Detection { bin_lo: 510, bin_hi: 513, frame_lo:
/// 0, frame_hi: 149, peak_dbfs: -9.999, snr_db: 13.5, .. }` — one
/// component alive for the *entire* capture, because the leakage never
/// stops being detected. That is the mechanism behind the owner's
/// complaint: a spurious, permanently-open "signal" sitting at center
/// frequency.
///
/// The "159 detections" count above was captured under `DetectorConfig::
/// new`'s pre-AN-1/AN-3 defaults (`Pfa` 1e-3, `min_blob_cells` 1) — both
/// retuned since (D-126, `CfarConfig::new`'s and `DetectorConfig::new`'s own
/// docs), so a fresh count today would be smaller. This test's own assertion
/// never depended on that count (only "no detection touches the DC band"),
/// and the DC-band exclusion it exercises runs unconditionally, before and
/// after the CFAR mask, so it is unaffected either way — the number is left
/// as a historical reading of the red control, not re-measured here.
#[test]
fn dc_spike_is_never_detected_or_blob_split() {
    let mut scene = noise_scene(201);
    scene.tones.push(ToneConfig {
        offset_hz: 0.0,
        level_dbfs: -10.0,
    });
    let frames = 150usize;
    let mut det = Detector::new(BINS, DetectorConfig::new()).unwrap();
    let mut dets = drive(&scene, frames, &mut det, |_, _, _| {});
    dets.extend(det.finish());

    let center = BINS / 2;
    assert!(
        dets.iter()
            .all(|d| d.bin_hi + 1 < center || d.bin_lo > center + 1),
        "DC spike was detected: {dets:?}"
    );
}

/// The exclusion is the smallest that removes the artefact (lane brief,
/// build step 3): a signal just outside the default guard band (center ± 1
/// bin) is detected exactly as any other tone would be, so nothing beyond
/// the measured leakage width is silently lost.
#[test]
fn a_signal_adjacent_to_the_guard_band_is_still_detected() {
    let center = BINS / 2;
    let near_bin = center + 2; // one bin outside the default guard
    let mut scene = noise_scene(202);
    scene.tones.push(ToneConfig {
        offset_hz: (near_bin as f64 - BINS as f64 / 2.0) * BIN_HZ,
        level_dbfs: per_bin_floor() + 20.0,
    });
    let frames = 150usize;
    let mut det = Detector::new(BINS, DetectorConfig::new()).unwrap();
    let mut dets = drive(&scene, frames, &mut det, |_, _, _| {});
    dets.extend(det.finish());
    assert!(
        dets.iter()
            .any(|d| d.bin_lo <= near_bin && d.bin_hi >= near_bin),
        "a real signal just outside the DC guard band went undetected: {dets:?}"
    );
}
