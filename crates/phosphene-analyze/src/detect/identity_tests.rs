// SPDX-License-Identifier: MIT

//! D-125 Amendment 3 item 3 (the owner's ruling on item 2): the negative
//! control for the incremental noise floor (`noise.rs`) and cached CFAR
//! (`cfar.rs`) is **identity, not tolerance** — the pixel goldens compare
//! within a tolerance (D-014), so they cannot prove the optimised code
//! produces the same output as before, only that it is close. This module
//! runs the frozen pre-optimisation reference implementations
//! (`noise::reference::NoiseFloorRef`, `cfar::reference::CfarRef`) side by
//! side with the real, optimised `NoiseFloor`/`Cfar`, over the same frames,
//! and asserts the floor, spread and occupancy mask are equal **bit for
//! bit** (`f32::to_bits`), never merely close.
//!
//! The scenes below reproduce the *shapes* of every committed golden scene
//! in `phosphene-app/tests/analyze_goldens.rs` (quiet noise, one steady
//! tone, one bursty signal, a ten-signal crowded band, a signal clipped at
//! the band edge) at that file's own rate/FFT (2.048 MS/s / 512), plus this
//! lane's own OOK burst scene at 10 MS/s / FFT 1024 (D-125's own concern)
//! and a long pure-noise stream. They are not byte-identical to that file's
//! scene helpers — those are private to a separate integration-test binary,
//! unreachable from this crate's own unit tests — but they exercise the
//! same signal shapes at the same parameters; see this module's own PR
//! description entry for the reasoning.

use phosphene_core::{Complex, SpectrumAnalyzer, WindowKind};
use phosphene_sources::siggen::OokConfig;
use phosphene_sources::{HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig};

use crate::detect::cfar::reference::CfarRef;
use crate::detect::noise::reference::NoiseFloorRef;
use crate::detect::{Cfar, CfarConfig, NoiseFloor, NoiseFloorConfig};

fn bursty_hopper(offset_hz: f64, level_dbfs: f32) -> HopperConfig {
    HopperConfig {
        offsets_hz: vec![offset_hz],
        dwell_s: 0.010,
        period_s: 0.030,
        level_dbfs,
        order: HopOrder::Cycle,
    }
}

/// Drives `gen`'s spectra through the reference (pre-optimisation) and
/// production (optimised) noise+CFAR chains side by side, for `frames`
/// frames of `fft` bins, asserting the floor, spread and occupancy mask are
/// bit-for-bit identical at every single frame. Panics with the scene name,
/// frame index and bin on the first mismatch.
fn assert_identity_over_scene(scene_name: &str, mut gen: SigGen, fft: usize, frames: usize) {
    let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();
    let mut prod_noise = NoiseFloor::new(fft, NoiseFloorConfig::new()).unwrap();
    let mut ref_noise = NoiseFloorRef::new(fft, NoiseFloorConfig::new());
    let mut prod_cfar = Cfar::new(fft, CfarConfig::new()).unwrap();
    let mut ref_cfar = CfarRef::new(fft, CfarConfig::new());
    let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
    let mut spectrum = vec![0.0f32; fft];
    let mut prod_mask = vec![false; fft];
    let mut ref_mask = vec![false; fft];

    for frame_idx in 0..frames {
        gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);

        prod_noise.push_frame(&spectrum);
        ref_noise.push_frame(&spectrum);

        for (bin, (&pf, &rf)) in prod_noise
            .floor_dbfs()
            .iter()
            .zip(ref_noise.floor_dbfs())
            .enumerate()
        {
            assert_eq!(
                pf.to_bits(),
                rf.to_bits(),
                "{scene_name}: floor mismatch at frame {frame_idx} bin {bin}: \
                 incremental {pf} vs reference {rf}"
            );
        }
        for (bin, (&ps, &rs)) in prod_noise
            .spread_db()
            .iter()
            .zip(ref_noise.spread_db())
            .enumerate()
        {
            assert_eq!(
                ps.to_bits(),
                rs.to_bits(),
                "{scene_name}: spread mismatch at frame {frame_idx} bin {bin}: \
                 incremental {ps} vs reference {rs}"
            );
        }

        prod_cfar.detect(&spectrum, prod_noise.floor_dbfs(), &mut prod_mask);
        ref_cfar.detect(&spectrum, ref_noise.floor_dbfs(), &mut ref_mask);
        assert_eq!(
            prod_mask, ref_mask,
            "{scene_name}: occupancy mask mismatch at frame {frame_idx}"
        );
    }
}

const RATE_HZ: f64 = 2_048_000.0;
const FFT: usize = 512;

#[test]
fn identity_quiet_noise_only() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 101;
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("quiet-noise", SigGen::new(cfg).unwrap(), FFT, 2_000);
}

#[test]
fn identity_one_steady_tone() {
    // 100% duty — the noise-floor estimator's documented blind spot
    // statistically, but the identity check is purely mechanical: the
    // incremental and reference windows must still agree bit for bit on
    // whatever (possibly biased) value they both compute.
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 102;
    cfg.tones.push(ToneConfig {
        offset_hz: 300_000.0,
        level_dbfs: -20.0,
    });
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("steady-tone", SigGen::new(cfg).unwrap(), FFT, 2_000);
}

#[test]
fn identity_one_bursty_signal() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 103;
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("one-bursty-signal", SigGen::new(cfg).unwrap(), FFT, 3_000);
}

#[test]
fn identity_crowded_band_ten_signals() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 104;
    for i in 0..10i64 {
        let offset = (i - 5) as f64 * 40_000.0;
        let level = -15.0 - i as f32;
        cfg.hoppers.push(bursty_hopper(offset, level));
    }
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("crowded-band", SigGen::new(cfg).unwrap(), FFT, 4_000);
}

#[test]
fn identity_signal_clipped_at_the_band_edge() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 105;
    cfg.hoppers.push(bursty_hopper(1_012_000.0, -15.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("edge-clipped", SigGen::new(cfg).unwrap(), FFT, 2_000);
}

#[test]
fn identity_ook_burst_at_10_ms_per_s() {
    // D-125's own scene shape: this lane exists because of exactly this
    // kind of burst, at the rate item 2's own measurement used.
    let mut cfg = SigGenConfig::new(10_000_000.0);
    cfg.seed = 106;
    cfg.ook.push(OokConfig {
        offset_hz: 300_000.0,
        symbol_rate_bd: 2000.0,
        level_dbfs: -20.0,
    });
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_identity_over_scene("ook-burst-10msps", SigGen::new(cfg).unwrap(), 1024, 3_000);
}

#[test]
fn identity_long_random_stream() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 107;
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    // A hopper too, so the stream is not pure noise throughout — many
    // window-slides worth of occupancy transitions over a long run.
    cfg.hoppers.push(bursty_hopper(250_000.0, -18.0));
    assert_identity_over_scene("long-random-stream", SigGen::new(cfg).unwrap(), FFT, 20_000);
}

/// D-125 Amendment 3 item 3: "a deliberately broken variant (for example an
/// off-by-one in the window removal) must turn that test red; quote the
/// red." A near-duplicate of [`NoiseFloor::push_frame`]'s incremental
/// eviction with exactly that bug: the outgoing value's own slot is left in
/// the shifted region (`idx..` instead of `idx + 1..`), so the window one
/// element too many gets shifted and the window silently keeps a stale
/// duplicate instead of truly evicting the outgoing sample. Exists only to
/// prove [`assert_identity_over_scene`] has power to catch a broken
/// incremental implementation — never used outside this one test.
struct NoiseFloorBroken {
    bins: usize,
    window_frames: usize,
    percentile: f64,
    h1: Vec<f64>,
    h2: Vec<f64>,
    ring: Vec<f32>,
    sorted: Vec<f32>,
    slot: usize,
    held: usize,
    floor: Vec<f32>,
    spread: Vec<f32>,
}

impl NoiseFloorBroken {
    // Duplicated from `noise.rs`'s own private consts (this type is a
    // deliberately separate, deliberately buggy copy, not a reader of the
    // real implementation's internals).
    const OCCUPIED_MARGIN_DB: f32 = 7.0;
    const NOISE_FLAG_RATE: f64 = 0.0310;
    const IQR_TO_SIGMA: f32 = 1.349;

    fn new(bins: usize, config: NoiseFloorConfig) -> Self {
        let w = config.window_frames;
        let (mut h1, mut h2) = (vec![0.0f64; w + 1], vec![0.0f64; w + 1]);
        for j in 1..=w {
            h1[j] = h1[j - 1] + 1.0 / j as f64;
            h2[j] = h2[j - 1] + 1.0 / (j as f64 * j as f64);
        }
        NoiseFloorBroken {
            bins,
            window_frames: w,
            percentile: config.percentile,
            h1,
            h2,
            ring: vec![0.0; bins * w],
            sorted: vec![0.0; bins * w],
            slot: 0,
            held: 0,
            floor: vec![0.0; bins],
            spread: vec![0.0; bins],
        }
    }

    fn push_frame(&mut self, frame: &[f32]) {
        let w = self.window_frames;
        let held_before = self.held;
        for (bin, &v) in frame.iter().enumerate() {
            let base = bin * w;
            if held_before == w {
                let outgoing = self.ring[base + self.slot];
                let region = &mut self.sorted[base..base + w];
                let idx = region
                    .binary_search_by(|probe| probe.total_cmp(&outgoing))
                    .expect("outgoing value must be present");
                // BUG (deliberate): should be `idx + 1..`, so the outgoing
                // slot itself is shifted onto its own position too, leaving
                // a stray duplicate instead of a true eviction.
                region.copy_within(idx.., idx);
                let ins = region[..w - 1]
                    .binary_search_by(|probe| probe.total_cmp(&v))
                    .unwrap_or_else(|e| e);
                region.copy_within(ins..w - 1, ins + 1);
                region[ins] = v;
            } else {
                let len = held_before;
                let region = &mut self.sorted[base..base + len + 1];
                let ins = region[..len]
                    .binary_search_by(|probe| probe.total_cmp(&v))
                    .unwrap_or_else(|e| e);
                region.copy_within(ins..len, ins + 1);
                region[ins] = v;
            }
            self.ring[base + self.slot] = v;
        }
        self.slot = (self.slot + 1) % w;
        self.held = (self.held + 1).min(w);

        let n = self.held;
        let rank = |q: f64, m: usize| ((q * m as f64).ceil() as usize).clamp(1, m) - 1;
        for bin in 0..self.bins {
            let sorted = &self.sorted[bin * w..bin * w + n];
            let cutoff = sorted[rank(0.5, n)] + Self::OCCUPIED_MARGIN_DB;
            let clean = sorted.partition_point(|&v| v <= cutoff);
            let r = rank(self.percentile, clean) + 1;
            let m = ((clean as f64 / (1.0 - Self::NOISE_FLAG_RATE)).round() as usize).clamp(r, n);
            let (mu, v) = (self.h1[m] - self.h1[m - r], self.h2[m] - self.h2[m - r]);
            let expected_db =
                10.0 * mu.log10() - (10.0 / std::f64::consts::LN_10) * v / (2.0 * mu * mu);
            self.floor[bin] = sorted[r - 1] - expected_db as f32;
            self.spread[bin] = (sorted[rank(0.75, n)] - sorted[rank(0.25, n)]) / Self::IQR_TO_SIGMA;
        }
    }

    fn floor_dbfs(&self) -> &[f32] {
        &self.floor
    }
}

#[test]
#[should_panic(expected = "floor mismatch")]
fn a_broken_incremental_variant_is_caught_by_the_identity_check() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 108;
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    let mut gen = SigGen::new(cfg).unwrap();

    let mut analyzer = SpectrumAnalyzer::new(FFT, WindowKind::Hann).unwrap();
    let mut broken = NoiseFloorBroken::new(FFT, NoiseFloorConfig::new());
    let mut reference = NoiseFloorRef::new(FFT, NoiseFloorConfig::new());
    let mut chunk = vec![Complex::new(0.0f32, 0.0); FFT];
    let mut spectrum = vec![0.0f32; FFT];

    // A full window's worth of frames plus enough churn afterward for at
    // least one eviction to occur and its consequence to surface: the
    // window fills at frame `window_frames`, so this comfortably covers it.
    for frame_idx in 0..200 {
        gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);
        broken.push_frame(&spectrum);
        reference.push_frame(&spectrum);
        for (bin, (&bf, &rf)) in broken
            .floor_dbfs()
            .iter()
            .zip(reference.floor_dbfs())
            .enumerate()
        {
            assert_eq!(
                bf.to_bits(),
                rf.to_bits(),
                "floor mismatch at frame {frame_idx} bin {bin}: broken {bf} vs reference {rf}"
            );
        }
    }
}

/// D-125 Amendment 4 item 2: "extend the identity module to compare the
/// parallel path against the single-thread one." `NoiseFloor::push_frame`
/// / `Cfar::detect` always dispatch through whichever rayon thread pool is
/// currently installed — the process-global one (real parallelism, however
/// many threads this machine has) by default, or a pool an `install` call
/// scopes to the closure it wraps. `seq_pool` below is a dedicated
/// single-thread pool, so `seq_noise`/`seq_cfar`'s calls run the identical
/// code — not a separate sequential branch — genuinely confined to one
/// thread, over the same scenes item 3's reference-vs-optimised check used.
/// A parallelisation bug and an algorithm bug are covered by separate
/// tests this way.
fn assert_parallel_matches_single_thread_over_scene(
    scene_name: &str,
    mut gen: SigGen,
    fft: usize,
    frames: usize,
) {
    let seq_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();
    let mut par_noise = NoiseFloor::new(fft, NoiseFloorConfig::new()).unwrap();
    let mut seq_noise = NoiseFloor::new(fft, NoiseFloorConfig::new()).unwrap();
    let mut par_cfar = Cfar::new(fft, CfarConfig::new()).unwrap();
    let mut seq_cfar = Cfar::new(fft, CfarConfig::new()).unwrap();
    let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
    let mut spectrum = vec![0.0f32; fft];
    let mut par_mask = vec![false; fft];
    let mut seq_mask = vec![false; fft];

    for frame_idx in 0..frames {
        gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);

        par_noise.push_frame(&spectrum);
        seq_pool.install(|| seq_noise.push_frame(&spectrum));

        for (bin, (&pf, &sf)) in par_noise
            .floor_dbfs()
            .iter()
            .zip(seq_noise.floor_dbfs())
            .enumerate()
        {
            assert_eq!(
                pf.to_bits(),
                sf.to_bits(),
                "{scene_name}: parallel/single-thread floor mismatch at frame {frame_idx} \
                 bin {bin}: parallel {pf} vs single-thread {sf}"
            );
        }
        for (bin, (&ps, &ss)) in par_noise
            .spread_db()
            .iter()
            .zip(seq_noise.spread_db())
            .enumerate()
        {
            assert_eq!(
                ps.to_bits(),
                ss.to_bits(),
                "{scene_name}: parallel/single-thread spread mismatch at frame {frame_idx} \
                 bin {bin}: parallel {ps} vs single-thread {ss}"
            );
        }

        par_cfar.detect(&spectrum, par_noise.floor_dbfs(), &mut par_mask);
        seq_pool.install(|| seq_cfar.detect(&spectrum, seq_noise.floor_dbfs(), &mut seq_mask));
        assert_eq!(
            par_mask, seq_mask,
            "{scene_name}: parallel/single-thread mask mismatch at frame {frame_idx}"
        );
    }
}

#[test]
fn parallel_matches_single_thread_quiet_noise_only() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 201;
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "quiet-noise",
        SigGen::new(cfg).unwrap(),
        FFT,
        2_000,
    );
}

#[test]
fn parallel_matches_single_thread_one_steady_tone() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 202;
    cfg.tones.push(ToneConfig {
        offset_hz: 300_000.0,
        level_dbfs: -20.0,
    });
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "steady-tone",
        SigGen::new(cfg).unwrap(),
        FFT,
        2_000,
    );
}

#[test]
fn parallel_matches_single_thread_one_bursty_signal() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 203;
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "one-bursty-signal",
        SigGen::new(cfg).unwrap(),
        FFT,
        3_000,
    );
}

#[test]
fn parallel_matches_single_thread_crowded_band_ten_signals() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 204;
    for i in 0..10i64 {
        let offset = (i - 5) as f64 * 40_000.0;
        let level = -15.0 - i as f32;
        cfg.hoppers.push(bursty_hopper(offset, level));
    }
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "crowded-band",
        SigGen::new(cfg).unwrap(),
        FFT,
        4_000,
    );
}

#[test]
fn parallel_matches_single_thread_signal_clipped_at_the_band_edge() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 205;
    cfg.hoppers.push(bursty_hopper(1_012_000.0, -15.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "edge-clipped",
        SigGen::new(cfg).unwrap(),
        FFT,
        2_000,
    );
}

#[test]
fn parallel_matches_single_thread_ook_burst_at_10_ms_per_s() {
    let mut cfg = SigGenConfig::new(10_000_000.0);
    cfg.seed = 206;
    cfg.ook.push(OokConfig {
        offset_hz: 300_000.0,
        symbol_rate_bd: 2000.0,
        level_dbfs: -20.0,
    });
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    assert_parallel_matches_single_thread_over_scene(
        "ook-burst-10msps",
        SigGen::new(cfg).unwrap(),
        1024,
        3_000,
    );
}

#[test]
fn parallel_matches_single_thread_long_random_stream() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 207;
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    cfg.hoppers.push(bursty_hopper(250_000.0, -18.0));
    assert_parallel_matches_single_thread_over_scene(
        "long-random-stream",
        SigGen::new(cfg).unwrap(),
        FFT,
        20_000,
    );
}

/// D-125 Amendment 4 item 2's power proof: "with a deliberately broken
/// parallel variant that goes red (quote it)." `Cfar::
/// detect_with_wrong_chunk_offsets` is identical to the real parallel
/// `detect` except every chunk computes as though it starts at bin 0 —
/// a plausible coordination bug (a hand-rolled offset-tracking loop
/// forgetting to add its chunk's own start), not a math bug. Forcing more
/// than one thread makes the bug observable (a single chunk covering every
/// bin has nothing to misalign).
#[test]
#[should_panic(expected = "mask mismatch")]
fn a_broken_parallel_variant_is_caught_by_the_identity_check() {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = 208;
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    let mut gen = SigGen::new(cfg).unwrap();

    let mut analyzer = SpectrumAnalyzer::new(FFT, WindowKind::Hann).unwrap();
    let mut noise = NoiseFloor::new(FFT, NoiseFloorConfig::new()).unwrap();
    let mut broken_cfar = Cfar::new(FFT, CfarConfig::new()).unwrap();
    let mut correct_cfar = Cfar::new(FFT, CfarConfig::new()).unwrap();
    let mut chunk = vec![Complex::new(0.0f32, 0.0); FFT];
    let mut spectrum = vec![0.0f32; FFT];
    let mut broken_mask = vec![false; FFT];
    let mut correct_mask = vec![false; FFT];

    for frame_idx in 0..200 {
        gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);
        noise.push_frame(&spectrum);
        broken_cfar.detect_with_wrong_chunk_offsets(
            &spectrum,
            noise.floor_dbfs(),
            &mut broken_mask,
            4,
        );
        correct_cfar.detect(&spectrum, noise.floor_dbfs(), &mut correct_mask);
        assert_eq!(
            broken_mask, correct_mask,
            "mask mismatch at frame {frame_idx}"
        );
    }
}
