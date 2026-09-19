// SPDX-License-Identifier: MIT

//! Power spectrum in dBFS/bin (spec §7.2, calibration per D-006).
//!
//! [`SpectrumAnalyzer`] wraps `rustfft` behind the pipeline stage
//! `window → FFT → |·|² → dBFS → fftshift`. All buffers are preallocated at
//! construction; [`SpectrumAnalyzer::process`] performs no allocation (§7.7).

use std::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::window::{Window, WindowKind};

/// Smallest supported FFT size (FR-D10).
pub const MIN_FFT_SIZE: usize = 512;
/// Largest supported FFT size (FR-D10).
pub const MAX_FFT_SIZE: usize = 32768;
/// Default FFT size (FR-D10, clarification C6/§13 item 4).
pub const DEFAULT_FFT_SIZE: usize = 1024;
/// Lower clamp on reported power, to avoid −inf (§7.2).
pub const DBFS_FLOOR: f32 = -200.0;

/// Errors from configuring the DSP core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreError {
    /// The requested FFT size is not a power of two in
    /// [`MIN_FFT_SIZE`]..=[`MAX_FFT_SIZE`].
    InvalidFftSize(usize),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::InvalidFftSize(n) => write!(
                f,
                "invalid FFT size {n}: must be a power of two in \
                 {MIN_FFT_SIZE}..={MAX_FFT_SIZE}"
            ),
        }
    }
}

impl std::error::Error for CoreError {}

/// Map a raw FFT bin index to its position in the DC-centered (fftshifted)
/// spectrum of length `size`: DC lands at `size / 2`, positive frequencies to
/// its right, negative frequencies to its left (§7.2).
///
/// # Panics
///
/// Panics if `bin >= size`.
pub fn fftshift_index(bin: usize, size: usize) -> usize {
    assert!(bin < size, "bin {bin} out of range for FFT size {size}");
    (bin + size / 2) % size
}

/// Preallocated `window → FFT → |·|² → dBFS → fftshift` pipeline stage.
///
/// Output spectra are DC-centered and calibrated to the D-006 convention:
/// **0 dBFS = full-scale complex sinusoid**, FFT normalised by N with the
/// window's coherent gain corrected out, reported as **dBFS/bin**. The reading
/// for a given signal is therefore invariant across FFT sizes and window
/// choices. Per bin:
///
/// ```text
/// P_k = 10·log10( |X_k|² / (Σ w[n])² )      floored at −200 dB
/// ```
///
/// where `(Σw)² = (N·CG)²` folds the 1/N FFT normalisation and the coherent
/// gain correction into a single constant (§7.2's `C_cal`).
pub struct SpectrumAnalyzer {
    size: usize,
    window: Window,
    fft: Arc<dyn Fft<f32>>,
    /// Windowed frame; transformed in place by the FFT.
    buffer: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    /// `1 / (Σ w[n])²` — applied to |X_k|² before the log.
    power_scale: f32,
}

impl SpectrumAnalyzer {
    /// Build an analyzer for `size`-point frames under the given window.
    ///
    /// `size` must be a power of two in [`MIN_FFT_SIZE`]..=[`MAX_FFT_SIZE`]
    /// (FR-D10 names powers of two throughout; other sizes are rejected
    /// rather than guessed at).
    pub fn new(size: usize, window: WindowKind) -> Result<Self, CoreError> {
        if !(MIN_FFT_SIZE..=MAX_FFT_SIZE).contains(&size) || !size.is_power_of_two() {
            return Err(CoreError::InvalidFftSize(size));
        }
        let window = Window::new(window, size);
        let sum: f64 = window.coefficients().iter().map(|&w| w as f64).sum();
        let fft = FftPlanner::new().plan_fft_forward(size);
        let scratch_len = fft.get_inplace_scratch_len();
        Ok(SpectrumAnalyzer {
            size,
            window,
            buffer: vec![Complex::new(0.0, 0.0); size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            fft,
            power_scale: (1.0 / (sum * sum)) as f32,
        })
    }

    /// FFT size N (also the frame length).
    pub fn size(&self) -> usize {
        self.size
    }

    /// The analysis window in use.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Compute the DC-centered dBFS/bin spectrum of one frame.
    ///
    /// `frame` is one batch of N consecutive complex baseband samples;
    /// `dbfs_out[fftshift_index(k, N)]` receives `P_k`. Allocation-free.
    ///
    /// # Panics
    ///
    /// Panics if `frame.len()` or `dbfs_out.len()` differs from
    /// [`size`](Self::size) — a programmer error, not a runtime condition.
    pub fn process(&mut self, frame: &[Complex<f32>], dbfs_out: &mut [f32]) {
        assert_eq!(frame.len(), self.size, "frame length must equal FFT size");
        assert_eq!(
            dbfs_out.len(),
            self.size,
            "output length must equal FFT size"
        );

        for ((out, &sample), &w) in self
            .buffer
            .iter_mut()
            .zip(frame)
            .zip(self.window.coefficients())
        {
            *out = sample * w;
        }
        self.fft
            .process_with_scratch(&mut self.buffer, &mut self.scratch);

        let half = self.size / 2;
        for (k, x) in self.buffer.iter().enumerate() {
            let power = (x.re * x.re + x.im * x.im) * self.power_scale;
            // log10(0) = −inf; max() clamps it to the floor (§7.2).
            let dbfs = (10.0 * power.log10()).max(DBFS_FLOOR);
            dbfs_out[(k + half) % self.size] = dbfs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_and_non_power_of_two_sizes() {
        for bad in [0, 256, 511, 1000, 1023, 65536] {
            assert_eq!(
                SpectrumAnalyzer::new(bad, WindowKind::Hann).err(),
                Some(CoreError::InvalidFftSize(bad)),
            );
        }
        for good in [512, 1024, 4096, 32768] {
            assert!(SpectrumAnalyzer::new(good, WindowKind::Hann).is_ok());
        }
    }

    #[test]
    fn fftshift_maps_dc_to_center_and_nyquist_to_edge() {
        assert_eq!(fftshift_index(0, 1024), 512);
        assert_eq!(fftshift_index(512, 1024), 0);
        assert_eq!(fftshift_index(1, 1024), 513);
        assert_eq!(fftshift_index(1023, 1024), 511);
    }

    #[test]
    fn error_message_names_the_size() {
        let msg = CoreError::InvalidFftSize(1000).to_string();
        assert!(msg.contains("1000"));
    }
}
