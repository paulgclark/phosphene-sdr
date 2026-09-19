// SPDX-License-Identifier: MIT

//! Calibration goldens (spec §10, D-006).
//!
//! The one that matters: a full-scale complex tone reads 0.0 dBFS ±0.1 dB in
//! its bin — and reads the *same* number at FFT sizes 1024 and 2048 and under
//! both Hann and Blackman-Harris. That invariance is the whole reason D-006
//! corrects coherent gain out.

use phosphene_core::{fftshift_index, Complex, SpectrumAnalyzer, WindowKind, DBFS_FLOOR};

/// Synthesise a complex tone at `cycles` cycles per frame (i.e. raw FFT bin
/// `cycles`), amplitude `amp`, phases evaluated in f64 before the f32 cast.
fn tone(n: usize, cycles: usize, amp: f32) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * cycles as f64 * i as f64 / n as f64;
            Complex::new(amp * phase.cos() as f32, amp * phase.sin() as f32)
        })
        .collect()
}

fn add(a: &mut [Complex<f32>], b: &[Complex<f32>]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

fn spectrum(n: usize, window: WindowKind, frame: &[Complex<f32>]) -> Vec<f32> {
    let mut analyzer = SpectrumAnalyzer::new(n, window).expect("valid size");
    let mut dbfs = vec![0.0f32; n];
    analyzer.process(frame, &mut dbfs);
    dbfs
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Deterministic SplitMix64 → Box-Muller complex Gaussian noise source.
struct NoiseGen {
    state: u64,
}

impl NoiseGen {
    fn new(seed: u64) -> Self {
        NoiseGen { state: seed }
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Complex Gaussian sample with E[|x|²] = `power` (variance split evenly
    /// between the real and imaginary components).
    fn complex_gaussian(&mut self, power: f64) -> Complex<f32> {
        let sigma_c = (power / 2.0).sqrt();
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * self.uniform();
        Complex::new(
            (sigma_c * r * theta.cos()) as f32,
            (sigma_c * r * theta.sin()) as f32,
        )
    }
}

/// D-006 golden: full-scale tone → 0.0 dBFS ±0.1 dB in its bin, and the SAME
/// reading across FFT sizes 1024/2048 and windows Hann/Blackman-Harris
/// (rectangular included for good measure).
#[test]
fn full_scale_tone_reads_zero_dbfs_invariantly() {
    let mut readings = Vec::new();
    for n in [1024, 2048] {
        for window in [
            WindowKind::Hann,
            WindowKind::BlackmanHarris4,
            WindowKind::Rectangular,
        ] {
            let dbfs = spectrum(n, window, &tone(n, 137, 1.0));
            let peak_bin = fftshift_index(137, n);
            assert_eq!(
                argmax(&dbfs),
                peak_bin,
                "peak in the wrong bin (N={n}, {window:?})"
            );
            let peak = dbfs[peak_bin];
            assert!(
                peak.abs() <= 0.1,
                "full-scale tone read {peak} dBFS (N={n}, {window:?})"
            );
            readings.push(peak);
        }
    }
    // The invariance clause: every configuration reports the same number.
    let spread = readings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - readings.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        spread <= 0.02,
        "calibration varies across size/window by {spread} dB: {readings:?}"
    );
}

/// Two-tone golden: both tones land in the right (DC-centered) bins at the
/// right levels — 0 dBFS at +100 bins, −20 dBFS at −200 bins.
#[test]
fn two_tone_lands_in_the_right_bins_at_the_right_levels() {
    let n = 1024;
    for window in [WindowKind::Hann, WindowKind::BlackmanHarris4] {
        let mut frame = tone(n, 100, 1.0);
        add(&mut frame, &tone(n, n - 200, 0.1)); // −200 bins == raw bin N−200
        let dbfs = spectrum(n, window, &frame);

        let strong = dbfs[fftshift_index(100, n)];
        let weak = dbfs[fftshift_index(n - 200, n)];
        assert_eq!(argmax(&dbfs), fftshift_index(100, n), "{window:?}");
        assert!(
            strong.abs() <= 0.1,
            "strong tone read {strong} dBFS ({window:?})"
        );
        assert!(
            (weak + 20.0).abs() <= 0.1,
            "−20 dBFS tone read {weak} dBFS ({window:?})"
        );
        // And the negative-frequency tone sits left of center, mirrored.
        assert_eq!(fftshift_index(n - 200, n), n / 2 - 200);
    }
}

/// Noise golden: white noise of known per-sample power σ² averages to
/// σ²·ENBW/N per bin (in linear power) — the dBFS/bin convention of D-006
/// with the window's equivalent noise bandwidth.
#[test]
fn noise_floor_matches_enbw_analytics() {
    let n = 1024;
    let frames = 400;
    let power = 0.01; // −20 dBFS per-sample power

    for window in [
        WindowKind::Hann,
        WindowKind::BlackmanHarris4,
        WindowKind::Rectangular,
    ] {
        let mut analyzer = SpectrumAnalyzer::new(n, window).expect("valid size");
        let mut rng = NoiseGen::new(0x0D06_F00D);
        let mut dbfs = vec![0.0f32; n];
        let mut linear_sum = 0.0f64;
        for _ in 0..frames {
            let frame: Vec<Complex<f32>> = (0..n).map(|_| rng.complex_gaussian(power)).collect();
            analyzer.process(&frame, &mut dbfs);
            for &p in &dbfs {
                linear_sum += 10f64.powf(p as f64 / 10.0);
            }
        }
        let measured_db = 10.0 * (linear_sum / (frames * n) as f64).log10();
        let expected_db = 10.0 * power.log10() + 10.0 * analyzer.window().enbw_bins().log10()
            - 10.0 * (n as f64).log10();
        assert!(
            (measured_db - expected_db).abs() <= 0.1,
            "noise floor {measured_db:.3} dBFS, expected {expected_db:.3} ({window:?})"
        );
    }
}

/// Determinism: fixed seed, fixed inputs → bit-identical outputs, including
/// across independently constructed analyzers.
#[test]
fn fixed_inputs_give_bit_identical_spectra() {
    let n = 2048;
    let mut rng = NoiseGen::new(42);
    let mut frame = tone(n, 300, 0.5);
    for x in frame.iter_mut() {
        *x += rng.complex_gaussian(1e-4);
    }

    let mut runs: Vec<Vec<u32>> = Vec::new();
    for _ in 0..2 {
        let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).expect("valid size");
        for _ in 0..2 {
            let mut dbfs = vec![0.0f32; n];
            analyzer.process(&frame, &mut dbfs);
            runs.push(dbfs.iter().map(|p| p.to_bits()).collect());
        }
    }
    for run in &runs[1..] {
        assert_eq!(*run, runs[0], "spectra are not bit-identical");
    }
}

/// Silence clamps to the −200 dB floor instead of −inf (§7.2).
#[test]
fn silence_reads_the_floor() {
    let n = 512;
    let frame = vec![Complex::new(0.0, 0.0); n];
    let dbfs = spectrum(n, WindowKind::Hann, &frame);
    assert!(
        dbfs.iter().all(|&p| p == DBFS_FLOOR),
        "expected all bins at the floor"
    );
}
