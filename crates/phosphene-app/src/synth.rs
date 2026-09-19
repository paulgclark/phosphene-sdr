// SPDX-License-Identifier: MIT

//! Local synthetic IQ source driving the M0-C live trace.
//!
//! The lane contract says to render synthetic data locally rather than block
//! on M0-B's signal generator; this stays deliberately small (the real
//! generator lives in `phosphene-sources`). It synthesizes a deterministic
//! complex baseband scene — two fixed tones, a slow wander tone, a bursty
//! carrier, and a Gaussian noise floor — and runs it through
//! `phosphene-core`'s §7.2 pipeline, then applies the §7.4 live-trace EMA:
//! `L_k ← L_k + α·(P_k − L_k)`, α = 1 − exp(−Δt/τ_live).

use phosphene_core::{Complex, SpectrumAnalyzer, WindowKind};

/// Live-trace time constant τ_live, seconds (§7.4 default 0.1 s).
const TAU_LIVE: f32 = 0.1;

/// Amplitude for a tone that should read `db` dBFS at its bin (D-006:
/// magnitude 1.0 reads 0 dBFS).
fn amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// `xorshift64*` PRNG — deterministic, seedable, no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 1.0) / (1u64 << 24) as f32
    }

    /// Standard Gaussian pair via Box–Muller.
    fn gaussian_pair(&mut self) -> (f32, f32) {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * self.uniform();
        (r * theta.cos(), r * theta.sin())
    }
}

/// One steady complex tone at a fixed relative frequency.
struct Tone {
    /// Frequency relative to the sample rate, −0.5..0.5.
    rel_freq: f32,
    amplitude: f32,
    phase: f32,
}

/// Deterministic synthetic spectrum source.
pub struct Synth {
    analyzer: SpectrumAnalyzer,
    fft_size: usize,
    sample_rate: f64,
    tones: Vec<Tone>,
    /// Wandering tone: relative center and phase of its slow sweep.
    wander_phase: f32,
    /// Burst tone carrier phase.
    burst_phase: f32,
    /// Burst gate phase in seconds.
    time: f64,
    rng: Rng,
    frame: Vec<Complex<f32>>,
    spectrum: Vec<f32>,
    /// §7.4 live-trace EMA state, dBFS/bin.
    live: Vec<f32>,
    live_initialized: bool,
    /// FFTs computed so far — the honest basis for the headless HUD's
    /// FFTs/s readout (FR-D11): counted, not estimated.
    ffts: u64,
}

impl Synth {
    /// Nominal sample rate of the synthetic stream, Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Analysis window name for the HUD.
    pub fn window_name(&self) -> &'static str {
        "HANN"
    }

    /// Build the M0 demo scene at the given FFT size.
    pub fn new(fft_size: usize) -> Result<Self, phosphene_core::CoreError> {
        let analyzer = SpectrumAnalyzer::new(fft_size, WindowKind::Hann)?;
        Ok(Self {
            analyzer,
            fft_size,
            sample_rate: 2_400_000.0,
            tones: vec![
                Tone {
                    rel_freq: -0.30,
                    amplitude: amp(-25.0),
                    phase: 0.0,
                },
                Tone {
                    rel_freq: 0.10,
                    amplitude: amp(-50.0),
                    phase: 0.0,
                },
            ],
            wander_phase: 0.0,
            burst_phase: 0.0,
            time: 0.0,
            rng: Rng(0x_5EED_CAFE_F00D_D00D),
            frame: vec![Complex::new(0.0, 0.0); fft_size],
            spectrum: vec![0.0; fft_size],
            live: vec![0.0; fft_size],
            live_initialized: false,
            ffts: 0,
        })
    }

    /// FFTs computed since construction. The synthetic scene processes every
    /// sample it generates, in whole frames — so 100% processed and zero
    /// drops are exact facts here, not placeholders.
    pub fn ffts(&self) -> u64 {
        self.ffts
    }

    /// Advance the scene by `dt` seconds and return the updated live trace
    /// (DC-centered dBFS/bin, one value per FFT bin), handing every computed
    /// dBFS/bin spectrum to `per_spectrum` on the way — the headless loop's
    /// seam for the §7.3 per-spectrum accumulator (D-033), which needs each
    /// spectrum, not the EMA the live trace keeps.
    ///
    /// Processes every sample of the interval in whole N-sample frames
    /// (§7.1's non-overlapping segmentation), capped to keep pathological
    /// `dt` values from stalling a frame.
    pub fn step_with(&mut self, dt: f32, mut per_spectrum: impl FnMut(&[f32])) -> &[f32] {
        let n = self.fft_size;
        let frames = ((dt as f64 * self.sample_rate / n as f64).round() as usize).clamp(1, 64);
        let alpha = 1.0 - (-(n as f32 / self.sample_rate as f32) / TAU_LIVE).exp();
        for _ in 0..frames {
            self.fill_frame();
            let (frame, spectrum) = (&self.frame, &mut self.spectrum);
            self.analyzer.process(frame, spectrum);
            self.ffts += 1;
            per_spectrum(&self.spectrum);
            if self.live_initialized {
                for (l, p) in self.live.iter_mut().zip(&self.spectrum) {
                    *l += alpha * (p - *l);
                }
            } else {
                self.live.copy_from_slice(&self.spectrum);
                self.live_initialized = true;
            }
        }
        &self.live
    }

    /// Synthesize the next N samples of the scene.
    fn fill_frame(&mut self) {
        let n = self.fft_size;
        let dt_sample = 1.0 / self.sample_rate;
        let two_pi = std::f32::consts::TAU;

        // Noise floor: complex Gaussian, σ chosen to land the per-bin floor
        // in the −90s dBFS at the default FFT size.
        let sigma = 3e-4;
        for s in self.frame.iter_mut() {
            let (re, im) = self.rng.gaussian_pair();
            *s = Complex::new(re * sigma, im * sigma);
        }

        // Steady tones.
        for tone in &mut self.tones {
            let dphi = two_pi * tone.rel_freq;
            let mut phase = tone.phase;
            for s in self.frame.iter_mut() {
                *s += Complex::new(phase.cos(), phase.sin()) * tone.amplitude;
                phase = (phase + dphi) % two_pi;
            }
            tone.phase = phase;
        }

        // Wander tone: slowly sweeps ±0.35 of the span (20 s period).
        let wander_freq = 0.35 * (two_pi * (self.time / 20.0) as f32).sin();
        let a_wander = amp(-38.0);
        let mut phase = self.wander_phase;
        let dphi = two_pi * wander_freq;
        // Burst tone: on 40% of a 1.6 s cycle.
        let burst_on = (self.time / 1.6).fract() < 0.4;
        let a_burst = amp(-45.0);
        let dphi_burst = two_pi * -0.12;
        let mut phase_burst = self.burst_phase;
        for s in self.frame.iter_mut() {
            *s += Complex::new(phase.cos(), phase.sin()) * a_wander;
            phase = (phase + dphi) % two_pi;
            // Phase advances through the off period so the burst carrier
            // stays coherent.
            if burst_on {
                *s += Complex::new(phase_burst.cos(), phase_burst.sin()) * a_burst;
            }
            phase_burst = (phase_burst + dphi_burst) % two_pi;
        }
        self.wander_phase = phase;
        self.burst_phase = phase_burst;

        self.time += n as f64 * dt_sample;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phosphene_core::fftshift_index;

    #[test]
    fn trace_shows_the_strong_tone_at_its_bin() {
        let mut synth = Synth::new(1024).unwrap();
        // Settle the EMA over a second of scene time.
        let mut trace: Vec<f32> = Vec::new();
        for _ in 0..60 {
            trace = synth.step_with(1.0 / 60.0, |_| {}).to_vec();
        }
        // Strong tone: −0.30 relative → raw bin 1024·0.70 (negative freqs
        // alias to the top half), fftshifted.
        let raw_bin = (1024.0 * 0.70f32).round() as usize;
        let idx = fftshift_index(raw_bin, 1024);
        let peak = trace[idx];
        assert!(
            (peak - -25.0).abs() < 2.0,
            "tone bin read {peak} dBFS, expected ≈ −25"
        );
        // The noise floor sits far below the tone.
        let median = {
            let mut s = trace.clone();
            s.sort_by(f32::total_cmp);
            s[s.len() / 2]
        };
        assert!(
            median < -80.0,
            "noise floor median {median} dBFS unexpectedly high"
        );
    }

    #[test]
    fn synth_is_deterministic() {
        let mut a = Synth::new(1024).unwrap();
        let mut b = Synth::new(1024).unwrap();
        for _ in 0..5 {
            let ta = a.step_with(1.0 / 60.0, |_| {}).to_vec();
            let tb = b.step_with(1.0 / 60.0, |_| {}).to_vec();
            assert_eq!(ta, tb);
        }
    }

    #[test]
    fn invalid_fft_size_is_rejected() {
        assert!(Synth::new(1000).is_err());
    }
}
