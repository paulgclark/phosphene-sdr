// SPDX-License-Identifier: MIT

//! Shared helpers for the accumulator seal tests (spec §10): exponential
//! time-constant fitting for the step-response assertions.

/// Fit `v(t) = v₀·exp(−t/τ)` to samples by least-squares regression of
/// `ln v` against `t`, returning the fitted τ in seconds.
///
/// Panics if fewer than 3 samples or any non-positive value is supplied —
/// the tests must select their fit windows so the data is usable.
pub fn fit_exponential_tau(samples: &[(f32, f32)]) -> f64 {
    assert!(
        samples.len() >= 3,
        "need at least 3 samples to fit, got {}",
        samples.len()
    );
    let n = samples.len() as f64;
    let (mut sum_t, mut sum_y, mut sum_tt, mut sum_ty) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &(t, v) in samples {
        assert!(v > 0.0, "exponential fit needs positive values, got {v}");
        let t = t as f64;
        let y = (v as f64).ln();
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }
    let slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t);
    assert!(slope < 0.0, "fit produced a non-decaying slope {slope}");
    -1.0 / slope
}

/// Assert a fitted τ is within `tol` relative error of the configured τ.
pub fn assert_tau_close(fitted: f64, expected: f32, tol: f64, what: &str) {
    let expected = expected as f64;
    let rel = (fitted - expected).abs() / expected;
    assert!(
        rel <= tol,
        "{what}: fitted τ = {fitted:.4} s vs configured {expected:.4} s \
         (relative error {rel:.3} > {tol})"
    );
}
