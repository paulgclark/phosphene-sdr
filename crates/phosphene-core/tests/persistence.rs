// SPDX-License-Identifier: MIT

//! Persistence-histogram seal tests (spec §10, lane M1-A).
//!
//! The step response is §10's own test: feed a signal that appears then
//! disappears, fit the intensity curve, and assert τ_rise and τ_decay within
//! tolerance — at two τ settings each, so the fit is testing the parameter
//! rather than a coincidence. Everything runs in deterministic mode: fixed
//! synthetic spectra, fixed Δt, no clocks.

mod common;

use common::{assert_tau_close, fit_exponential_tau};
use phosphene_core::{fftshift_index, Complex, PersistenceHistogram, SpectrumAnalyzer, WindowKind};

/// Display-frame interval Δt. Small enough that even the fastest §7.3 τ_rise
/// (0.02 s) yields plenty of fit points before the curve saturates.
const DT: f32 = 1.0 / 240.0;
/// Spectra folded in per display frame (K_total).
const SPECTRA_PER_TICK: usize = 8;

const BINS: usize = 4;
const LEVELS: usize = 128;
const P_BOTTOM: f32 = -100.0;
const P_TOP: f32 = 0.0;
/// The bin carrying the test tone and its on/off power levels.
const TONE_BIN: usize = 1;
const P_ON: f32 = -20.0;

fn quiet_spectrum() -> [f32; BINS] {
    [P_BOTTOM; BINS]
}

fn tone_spectrum() -> [f32; BINS] {
    let mut s = quiet_spectrum();
    s[TONE_BIN] = P_ON;
    s
}

fn histogram(tau_rise: f32, tau_decay: f32) -> PersistenceHistogram {
    let mut h = PersistenceHistogram::new(BINS, LEVELS, P_BOTTOM, P_TOP).unwrap();
    h.set_tau_rise(tau_rise).unwrap();
    h.set_tau_decay(tau_decay).unwrap();
    h
}

/// Advance one display frame: `SPECTRA_PER_TICK` spectra, then a Δt tick.
fn drive(h: &mut PersistenceHistogram, spectrum: &[f32; BINS], ticks: usize) {
    for _ in 0..ticks {
        for _ in 0..SPECTRA_PER_TICK {
            h.accumulate(spectrum);
        }
        h.tick(DT);
    }
}

/// §10 step response at one (τ_rise, τ_decay) setting.
fn step_response_case(tau_rise: f32, tau_decay: f32) {
    let mut h = histogram(tau_rise, tau_decay);
    let cell = h.level_index(P_ON);

    // Signal appears: the tone cell sees hit ratio T = 1 every frame, so
    // I(t) = 1 − exp(−t/τ_rise). Fit ln(1 − I) against t.
    let mut rise_samples = Vec::new();
    let mut t = 0.0f32;
    while rise_samples.len() < 400 {
        drive(&mut h, &tone_spectrum(), 1);
        t += DT;
        let residual = 1.0 - h.intensity_at(TONE_BIN, cell);
        // Keep the fit window where f32 still resolves 1 − I cleanly.
        if residual < 0.02 {
            break;
        }
        rise_samples.push((t, residual));
    }
    let fitted_rise = fit_exponential_tau(&rise_samples);
    assert_tau_close(fitted_rise, tau_rise, 0.05, "rise");

    // Saturate, then the signal disappears: T = 0 for the tone cell, so
    // I(t) = I₀·exp(−t/τ_decay). Fit ln(I) against t.
    drive(&mut h, &tone_spectrum(), (10.0 * tau_rise / DT) as usize);
    let mut decay_samples = Vec::new();
    let mut t = 0.0f32;
    while decay_samples.len() < 4000 {
        drive(&mut h, &quiet_spectrum(), 1);
        t += DT;
        let intensity = h.intensity_at(TONE_BIN, cell);
        if intensity < 1e-3 {
            break;
        }
        decay_samples.push((t, intensity));
    }
    let fitted_decay = fit_exponential_tau(&decay_samples);
    assert_tau_close(fitted_decay, tau_decay, 0.05, "decay");
}

#[test]
fn step_response_fits_tau_at_fast_setting() {
    // Both constants from the low end of their §7.3 ranges.
    step_response_case(0.03, 0.3);
}

#[test]
fn step_response_fits_tau_at_slow_setting() {
    // Both constants from the high end of their §7.3 ranges.
    step_response_case(0.06, 1.5);
}

#[test]
fn steady_tone_reaches_a_stable_intensity() {
    let mut h = histogram(0.05, 1.0);
    let cell = h.level_index(P_ON);

    // Run well past 5·τ_rise; the constantly-hit cell must sit at I ≈ 1.
    drive(&mut h, &tone_spectrum(), 200);
    let settled = h.intensity_at(TONE_BIN, cell);
    assert!(
        (settled - 1.0).abs() < 0.01,
        "steady tone settled at I = {settled}, expected ≈ 1"
    );

    // And be stable: one more frame moves it by essentially nothing.
    drive(&mut h, &tone_spectrum(), 1);
    let next = h.intensity_at(TONE_BIN, cell);
    assert!(
        (next - settled).abs() < 1e-4,
        "steady intensity still moving: {settled} → {next}"
    );

    // A never-hit cell in the same bin stays dark.
    assert_eq!(h.intensity_at(TONE_BIN, cell / 2), 0.0);
}

#[test]
fn rare_transient_flashes_then_fades() {
    // §3.1's whole point: a one-off burst produces a visible excursion that
    // then decays, rather than vanishing into an average.
    let mut h = histogram(0.05, 0.5);
    let cell = h.level_index(P_ON);

    // Quiet background, then a single burst spectrum within one frame.
    drive(&mut h, &quiet_spectrum(), 50);
    h.accumulate(&tone_spectrum());
    for _ in 1..SPECTRA_PER_TICK {
        h.accumulate(&quiet_spectrum());
    }
    h.tick(DT);
    let peak = h.intensity_at(TONE_BIN, cell);
    // One hit in K = 8 spectra is T = 1/8, folded by α_rise — a real,
    // nonzero excursion the renderer can show.
    let expected = (1.0 - (-DT / 0.05f32).exp()) / SPECTRA_PER_TICK as f32;
    assert!(
        peak > 0.5 * expected && peak > 1e-3,
        "one-hit excursion too dim to see: I = {peak}, expected ≈ {expected}"
    );

    // The excursion decays back toward dark.
    drive(&mut h, &quiet_spectrum(), (10.0 * 0.5 / DT) as usize);
    let faded = h.intensity_at(TONE_BIN, cell);
    assert!(
        faded < 1e-3,
        "transient failed to fade: I = {faded} after 10·τ_decay"
    );
    assert!(faded < peak);
}

#[test]
fn tick_with_no_spectra_holds_the_grid() {
    // K_total = 0 leaves T undefined; the fold is skipped and the grid holds
    // rather than fabricating an all-zero observation.
    let mut h = histogram(0.05, 1.0);
    drive(&mut h, &tone_spectrum(), 10);
    let before: Vec<f32> = h.intensity().to_vec();
    for _ in 0..100 {
        h.tick(DT);
    }
    assert_eq!(h.intensity(), &before[..]);
}

#[test]
fn calibrated_spectrum_feeds_the_histogram_unrescaled() {
    // D-006 coupling: a full-scale tone reads 0.0 dBFS out of the analyzer,
    // so with the grid topped at 0 dBFS its energy must land in the TOP level
    // of its (DC-centered) bin — the level entering the histogram is the
    // D-006 dBFS/bin figure, not a rescaled one.
    let n = 1024;
    let cycles = 100;
    let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
    let frame: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * cycles as f64 * i as f64 / n as f64;
            Complex::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect();
    let mut dbfs = vec![0.0f32; n];
    analyzer.process(&frame, &mut dbfs);

    let mut h = PersistenceHistogram::new(n, 128, -100.0, 0.0).unwrap();
    h.accumulate(&dbfs);
    h.tick(1.0 / 60.0);

    let bin = fftshift_index(cycles, n);
    assert!(
        h.intensity_at(bin, 127) > 0.0,
        "0 dBFS missed the top level"
    );
}
