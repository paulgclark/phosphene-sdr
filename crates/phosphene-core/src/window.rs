// SPDX-License-Identifier: MIT

//! Analysis window functions (FR-D10, spec §7.1).
//!
//! All windows are the *periodic* (DFT-even) variants — §7.1 specifies the
//! periodic Hann; the other windows follow the same convention so that their
//! coherent gain and equivalent noise bandwidth take their textbook values
//! exactly. Coefficients are evaluated in `f64` at construction time and
//! stored as `f32` for the hot path (§7.7).

/// Which analysis window to apply before the FFT (FR-D10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowKind {
    /// Periodic Hann — the default (§7.1).
    Hann,
    /// 4-term Blackman-Harris (−92 dB sidelobes), periodic form.
    BlackmanHarris4,
    /// Rectangular (no weighting).
    Rectangular,
}

impl WindowKind {
    /// The default window (FR-D10: Hann).
    pub const DEFAULT: WindowKind = WindowKind::Hann;
}

/// 4-term Blackman-Harris coefficients (Harris 1978, minimum 4-term window).
const BH4: [f64; 4] = [0.35875, 0.48829, 0.14128, 0.01168];

/// A window of a fixed length, with its precomputed coefficients and the
/// derived gains the calibration convention (D-006) and noise analytics need.
#[derive(Debug, Clone)]
pub struct Window {
    kind: WindowKind,
    coefficients: Vec<f32>,
    coherent_gain: f64,
    enbw_bins: f64,
}

impl Window {
    /// Build a window of `len` samples.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0`.
    pub fn new(kind: WindowKind, len: usize) -> Self {
        assert!(len > 0, "window length must be non-zero");
        let n = len as f64;
        let coefficients: Vec<f32> = (0..len)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n;
                let w = match kind {
                    WindowKind::Hann => 0.5 - 0.5 * theta.cos(),
                    WindowKind::BlackmanHarris4 => {
                        BH4[0] - BH4[1] * theta.cos() + BH4[2] * (2.0 * theta).cos()
                            - BH4[3] * (3.0 * theta).cos()
                    }
                    WindowKind::Rectangular => 1.0,
                };
                w as f32
            })
            .collect();

        // Gains are derived from the stored f32 coefficients (summed in f64)
        // so they describe exactly the window the hot path applies.
        let sum: f64 = coefficients.iter().map(|&w| w as f64).sum();
        let sum_sq: f64 = coefficients.iter().map(|&w| (w as f64) * (w as f64)).sum();
        Window {
            kind,
            coefficients,
            coherent_gain: sum / n,
            enbw_bins: n * sum_sq / (sum * sum),
        }
    }

    /// Which window this is.
    pub fn kind(&self) -> WindowKind {
        self.kind
    }

    /// Number of samples in the window.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// True if the window has zero length (never, by construction).
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The per-sample coefficients `w[n]`.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Coherent gain `CG = (Σ w[n]) / N` — the factor D-006 corrects out of
    /// the power spectrum so a full-scale tone reads 0 dBFS under any window.
    ///
    /// The correction itself is applied inside `spectrum.rs`; this accessor
    /// exists so the calibration tests can assert the D-006 invariant
    /// against the window's own figure of merit (kept by M1-K on that
    /// ground: an accessor that makes an invariant checkable is not dead).
    pub fn coherent_gain(&self) -> f64 {
        self.coherent_gain
    }

    /// Equivalent noise bandwidth in bins, `ENBW = N·Σw² / (Σw)²`.
    ///
    /// Under the D-006 dBFS/bin convention, white noise of per-sample power
    /// σ² has an expected per-bin level of `σ² · ENBW / N` — the relation
    /// the D-006 calibration tests assert through this accessor, and the
    /// value `phosphene-analyze`'s `MeasureConfig::for_window` consumes.
    pub fn enbw_bins(&self) -> f64 {
        self.enbw_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_gains_are_the_textbook_values() {
        // Periodic Hann: CG = 0.5 and ENBW = 1.5 exactly (up to f32 rounding).
        let w = Window::new(WindowKind::Hann, 1024);
        assert!((w.coherent_gain() - 0.5).abs() < 1e-6);
        assert!((w.enbw_bins() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn blackman_harris_gains_are_the_textbook_values() {
        // CG = a0; ENBW = (a0² + (a1²+a2²+a3²)/2) / a0² ≈ 2.0044.
        let w = Window::new(WindowKind::BlackmanHarris4, 2048);
        assert!((w.coherent_gain() - 0.35875).abs() < 1e-6);
        assert!((w.enbw_bins() - 2.0044).abs() < 1e-3);
    }

    #[test]
    fn rectangular_is_unity() {
        let w = Window::new(WindowKind::Rectangular, 512);
        assert!(w.coefficients().iter().all(|&c| c == 1.0));
        assert!((w.coherent_gain() - 1.0).abs() < 1e-9);
        assert!((w.enbw_bins() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn periodic_windows_are_dft_even() {
        // Periodic form: w[0] is the (near-)zero end sample and w[N/2] the
        // peak; w[k] == w[N-k] for k = 1..N/2 (symmetry about N/2).
        for kind in [WindowKind::Hann, WindowKind::BlackmanHarris4] {
            let n = 256;
            let w = Window::new(kind, n);
            let c = w.coefficients();
            for k in 1..n / 2 {
                assert!(
                    (c[k] - c[n - k]).abs() < 1e-6,
                    "{kind:?} not DFT-even at {k}"
                );
            }
        }
    }
}
