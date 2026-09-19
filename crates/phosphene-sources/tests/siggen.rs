// SPDX-License-Identifier: MIT

//! Seal tests for the M0-B lane (lane spec §Seal):
//!
//! * a seeded siggen run is bit-identical across invocations (and invariant
//!   to output chunking);
//! * a full-scale tone at a known offset, fed through `phosphene-core`, reads
//!   0.0 dBFS ±0.1 in the expected bin — the two crates agree on D-006;
//! * the hopper visits its configured frequencies with the specified
//!   dwell/period;
//! * the noise floor comes out at the requested level within tolerance.

use phosphene_core::{fftshift_index, SpectrumAnalyzer, Window, WindowKind};
use phosphene_sources::{
    ChirpConfig, Complex, HopOrder, HopperConfig, NoiseConfig, SampleSink, SampleSource, SigGen,
    SigGenConfig, SinkFlow, SourceDesc, SourceMeta, ToneConfig,
};

fn generate(gen: &mut SigGen, n: usize) -> Vec<Complex<f32>> {
    let mut out = vec![Complex::new(0.0, 0.0); n];
    gen.fill(&mut out);
    out
}

/// Mean instantaneous frequency (Hz) over a run of consecutive samples,
/// estimated from sample-to-sample phase differences.
fn mean_freq_hz(samples: &[Complex<f32>], rate: f64) -> f64 {
    let sum: f64 = samples
        .windows(2)
        .map(|w| {
            let d = w[1] * w[0].conj();
            (d.im as f64).atan2(d.re as f64)
        })
        .sum();
    sum / (samples.len() - 1) as f64 * rate / std::f64::consts::TAU
}

#[test]
fn seeded_run_is_bit_identical_and_chunking_invariant() {
    let total = 1 << 15;

    // One shot...
    let mut a = SigGen::new(SigGenConfig::demo()).unwrap();
    let one_shot = generate(&mut a, total);

    // ...versus the same scene read through ragged chunk sizes.
    let mut b = SigGen::new(SigGenConfig::demo()).unwrap();
    let mut chunked: Vec<Complex<f32>> = Vec::with_capacity(total);
    let mut sizes = [1usize, 7, 977, 4096].iter().cycle();
    while chunked.len() < total {
        let n = (*sizes.next().unwrap()).min(total - chunked.len());
        chunked.extend(generate(&mut b, n));
    }

    assert_eq!(one_shot.len(), chunked.len());
    for (i, (x, y)) in one_shot.iter().zip(&chunked).enumerate() {
        assert_eq!(
            (x.re.to_bits(), x.im.to_bits()),
            (y.re.to_bits(), y.im.to_bits()),
            "streams diverge at sample {i}",
        );
    }
}

#[test]
fn tones_land_in_the_expected_bins_at_their_dbfs_levels() {
    // Bin-exact offsets: bin_offset · rate/N. Raw FFT bin for a negative
    // offset −m is N−m.
    let n = 1024;
    let rate = 1_024_000.0;
    let mut config = SigGenConfig::new(rate);
    config.tones = vec![
        // The lane-seal tone: full scale at +100 bins.
        ToneConfig {
            offset_hz: 100.0 * rate / n as f64,
            level_dbfs: 0.0,
        },
        // A second tone checks level handling below full scale.
        ToneConfig {
            offset_hz: -200.0 * rate / n as f64,
            level_dbfs: -20.0,
        },
    ];
    let mut gen = SigGen::new(config).unwrap();
    let frame = generate(&mut gen, n);

    let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
    let mut dbfs = vec![0.0f32; n];
    analyzer.process(&frame, &mut dbfs);

    // D-006 agreement: 0.0 dBFS ±0.1 in the expected (DC-centered) bin,
    // which is also the spectrum's peak.
    let peak_bin = fftshift_index(100, n);
    let peak = dbfs[peak_bin];
    assert!(peak.abs() < 0.1, "full-scale tone read {peak} dBFS");
    let argmax = (0..n).max_by(|&a, &b| dbfs[a].total_cmp(&dbfs[b])).unwrap();
    assert_eq!(argmax, peak_bin, "tone peak landed in the wrong bin");

    let second = dbfs[fftshift_index(n - 200, n)];
    assert!(
        (second + 20.0).abs() < 0.1,
        "−20 dBFS tone read {second} dBFS"
    );
}

#[test]
fn hopper_visits_its_frequencies_with_the_specified_dwell_and_period() {
    let rate = 1_000_000.0;
    let offsets = [100_000.0, -150_000.0, 250_000.0];
    let dwell_s = 0.001; // 1000 samples
    let period_s = 0.002; // 2000 samples
    let level_dbfs = -10.0f32;

    let mut config = SigGenConfig::new(rate);
    config.hoppers = vec![HopperConfig {
        offsets_hz: offsets.to_vec(),
        dwell_s,
        period_s,
        level_dbfs,
        order: HopOrder::Cycle,
    }];
    let mut gen = SigGen::new(config).unwrap();

    let dwell = (dwell_s * rate) as usize;
    let period = (period_s * rate) as usize;
    let periods = 6; // two full cycles through the three offsets
    let samples = generate(&mut gen, periods * period);
    let magnitude = 10f64.powf(level_dbfs as f64 / 20.0);

    for p in 0..periods {
        let burst = &samples[p * period..p * period + dwell];
        let gap = &samples[p * period + dwell..(p + 1) * period];

        // Dwell: the burst is on for exactly `dwell` samples at the cycle's
        // frequency and configured level...
        let expected = offsets[p % offsets.len()];
        let estimated = mean_freq_hz(burst, rate);
        assert!(
            (estimated - expected).abs() < 1.0,
            "burst {p}: estimated {estimated} Hz, expected {expected} Hz"
        );
        for (i, s) in burst.iter().enumerate() {
            let norm = ((s.re as f64).powi(2) + (s.im as f64).powi(2)).sqrt();
            assert!(
                (norm - magnitude).abs() < 1e-3,
                "burst {p} sample {i}: |x| = {norm}, expected {magnitude}"
            );
        }

        // ...and period: silent until the next hop.
        for (i, s) in gap.iter().enumerate() {
            assert!(
                s.re == 0.0 && s.im == 0.0,
                "burst {p} gap sample {i} is not silent: {s:?}"
            );
        }
    }
}

#[test]
fn pseudorandom_hopper_visits_every_configured_frequency() {
    let rate = 1_000_000.0;
    let offsets = [100_000.0, -150_000.0, 250_000.0, -300_000.0];
    let mut config = SigGenConfig::new(rate);
    config.seed = 0xFEED;
    config.hoppers = vec![HopperConfig {
        offsets_hz: offsets.to_vec(),
        dwell_s: 0.001,
        period_s: 0.002,
        level_dbfs: -10.0,
        order: HopOrder::Pseudorandom,
    }];
    let mut gen = SigGen::new(config).unwrap();

    let (dwell, period, periods) = (1000usize, 2000usize, 32usize);
    let samples = generate(&mut gen, periods * period);
    let mut visited = [false; 4];
    for p in 0..periods {
        let estimated = mean_freq_hz(&samples[p * period..p * period + dwell], rate);
        let nearest = offsets
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (estimated - **a).abs().total_cmp(&(estimated - **b).abs()))
            .map(|(i, offset)| {
                assert!(
                    (estimated - offset).abs() < 1.0,
                    "burst {p}: {estimated} Hz is not one of the configured offsets"
                );
                i
            })
            .unwrap();
        visited[nearest] = true;
    }
    assert_eq!(visited, [true; 4], "not every offset was visited");
}

#[test]
fn noise_floor_reads_at_the_requested_level() {
    // The config asks for total noise power in dBFS; through the D-006
    // dBFS/bin convention the expected per-bin level is σ²·ENBW/N (see
    // phosphene_core::Window::enbw_bins). Average many spectra in the linear
    // power domain and compare.
    let n = 1024;
    let rate = 1_000_000.0;
    let level_dbfs = -30.0f32;
    let mut config = SigGenConfig::new(rate);
    config.seed = 1;
    config.noise = Some(NoiseConfig { level_dbfs });
    let mut gen = SigGen::new(config).unwrap();

    let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
    let mut dbfs = vec![0.0f32; n];
    let frames = 200;
    let mut linear_sum = 0.0f64;
    for _ in 0..frames {
        let frame = generate(&mut gen, n);
        analyzer.process(&frame, &mut dbfs);
        linear_sum += dbfs
            .iter()
            .map(|&p| 10f64.powf(p as f64 / 10.0))
            .sum::<f64>();
    }
    let mean_bin_power = linear_sum / (frames * n) as f64;

    let sigma_sq = 10f64.powf(level_dbfs as f64 / 10.0);
    let enbw = Window::new(WindowKind::Hann, n).enbw_bins();
    let expected = sigma_sq * enbw / n as f64;
    let error_db = 10.0 * (mean_bin_power / expected).log10();
    assert!(
        error_db.abs() < 0.2,
        "noise floor off by {error_db} dB (measured {mean_bin_power:e}, expected {expected:e})"
    );
}

#[test]
fn chirp_sweeps_from_start_to_stop_frequency() {
    let rate = 1_000_000.0;
    let sweep_len = 10_000usize;
    let mut config = SigGenConfig::new(rate);
    config.chirps = vec![ChirpConfig {
        start_offset_hz: 0.0,
        stop_offset_hz: 400_000.0,
        sweep_time_s: sweep_len as f64 / rate,
        level_dbfs: 0.0,
    }];
    let mut gen = SigGen::new(config).unwrap();
    let samples = generate(&mut gen, sweep_len);

    let early = mean_freq_hz(&samples[0..100], rate);
    let late = mean_freq_hz(&samples[sweep_len - 200..sweep_len - 100], rate);
    assert!(early < 5_000.0, "chirp starts at {early} Hz, expected ~0");
    assert!(
        (380_000.0..400_000.0).contains(&late),
        "chirp ends near {late} Hz, expected ~392 kHz"
    );
}

/// A sink that records everything and stops after a target sample count.
struct CollectingSink {
    samples: Vec<Complex<f32>>,
    metas: Vec<SourceMeta>,
    target: usize,
}

impl SampleSink for CollectingSink {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        assert!(
            !self.metas.is_empty(),
            "push before meta_changed violates the stream contract"
        );
        self.samples.extend_from_slice(samples);
        if self.samples.len() >= self.target {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }

    fn meta_changed(&mut self, meta: &SourceMeta) {
        self.metas.push(meta.clone());
    }
}

#[test]
fn streaming_through_the_trait_matches_direct_generation() {
    let config = SigGenConfig::demo();
    let mut source = SigGen::open(&SourceDesc::SigGen(config.clone())).unwrap();

    let mut sink = CollectingSink {
        samples: Vec::new(),
        metas: Vec::new(),
        target: 10_000,
    };
    source.stream(&mut sink).unwrap();

    assert_eq!(sink.metas.len(), 1);
    assert_eq!(sink.metas[0].sample_rate_hz, Some(config.sample_rate_hz));
    assert_eq!(sink.metas[0].label, "siggen");
    assert_eq!(source.meta(), sink.metas[0]);

    // The streamed samples are the same deterministic sequence fill() gives.
    let mut twin = SigGen::new(config).unwrap();
    let expected = generate(&mut twin, sink.samples.len());
    for (i, (x, y)) in sink.samples.iter().zip(&expected).enumerate() {
        assert_eq!(
            (x.re.to_bits(), x.im.to_bits()),
            (y.re.to_bits(), y.im.to_bits()),
            "streamed sample {i} differs from direct generation",
        );
    }
}
