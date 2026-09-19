// SPDX-License-Identifier: MIT

//! # phosphene-benches — the M0-D benchmark harness (never shipped)
//!
//! ## Architecture
//!
//! This crate hosts the criterion benches that re-baseline the feasibility
//! numbers of product spec §3.2 on real hardware (D-005, Appendix B) and
//! report throughput across the full FFT-size range 512–32768 (D-012). It is
//! `publish = false`, is excluded from release artifacts, and carries no
//! product code — the shared helpers below exist so the bench targets and
//! their sanity tests use the exact same signal construction.
//!
//! Baselines live in `benches/baselines/<class>.json` and are written and
//! checked by `cargo run -p xtask -- bench-baseline | bench-check` (spec §10:
//! a >10% median regression against the stored baseline for the runner class
//! fails CI).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use phosphene_core::Complex;

/// Deterministic full-scale complex tone at `cycles` cycles per frame of
/// length `n` — the same fixture class the D-006 calibration goldens use.
/// Benches must be deterministic (no RNG, no time-seeded state) so a run on
/// the same hardware measures the same work every time.
pub fn tone_frame(n: usize, cycles: f32) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * cycles * i as f32 / n as f32;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect()
}

/// The §7.2 magnitude²+log10 stage in isolation, as spec §3.2 row 2 measured
/// it: `10·log10(re² + im²)` per bin, floored well below the −200 dB clamp so
/// the bench never takes `log10(0)`.
pub fn mag_log(spectrum: &[Complex<f32>], dbfs_out: &mut [f32]) {
    for (out, s) in dbfs_out.iter_mut().zip(spectrum) {
        let power = (s.re * s.re + s.im * s.im).max(1e-30);
        *out = 10.0 * power.log10();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_frame_is_full_scale() {
        let frame = tone_frame(1024, 100.0);
        assert_eq!(frame.len(), 1024);
        for s in &frame {
            assert!((s.norm() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mag_log_of_unit_magnitude_is_zero_db() {
        let spectrum = vec![Complex::new(1.0f32, 0.0); 8];
        let mut out = vec![0.0f32; 8];
        mag_log(&spectrum, &mut out);
        for v in out {
            assert!(v.abs() < 1e-4);
        }
    }
}
