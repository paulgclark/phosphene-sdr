// SPDX-License-Identifier: MIT

//! # phosphene-core — DSP core (product spec §7, normative)
//!
//! ## Architecture
//!
//! This crate is the pure-computation heart of phosphene: it turns frames of
//! complex baseband samples into calibrated power spectra. It sits at the head
//! of the §7.0 pipeline — `window → FFT → |·|² → dBFS → fftshift` — and owns
//! nothing else: **no GUI, no I/O, no devices, no threads, no `unsafe`**
//! (NFR-Q1/Q2). Sources feed it, accumulators and renderers consume it; both
//! live in other crates.
//!
//! Modules:
//!
//! * [`window`] — analysis window functions (Hann default, Blackman-Harris
//!   4-term, rectangular — FR-D10) with their coherent gain and equivalent
//!   noise bandwidth.
//! * [`spectrum`] — the [`SpectrumAnalyzer`]: a preallocated, allocation-free
//!   wrapper over `rustfft` that produces DC-centered dBFS/bin spectra.
//! * [`accumulate`] — the display accumulators (§7.3–§7.5): the
//!   [`PersistenceHistogram`] (FR-D1, the centrepiece), the [`LiveTrace`] EMA
//!   (FR-D2), and the [`MaxHold`] trace with both D-007 modes (FR-D3).
//!   Callers pass Δt explicitly — the crate never reads a clock — so fixed
//!   inputs and fixed Δt give bit-identical grids (deterministic mode).
//! * [`health`] — the NFR-P3 degradation contract and its D-013 accounting:
//!   whole-batch assembly ([`Batcher`]), batch-resolution drop/completion
//!   records and the FR-D11 figures ([`PipelineHealth`]), and the NFR-P3
//!   ring-capacity arithmetic — all clock-free and allocation-free per batch.
//! * [`tap`] — the frozen next-gen analysis interfaces (D-017): the read-only
//!   [`SpectrumTap`] and [`AnalysisTap`] traits and their supporting types.
//!   Signatures only for now — live wiring is a later integration lane.
//!
//! ## Calibration convention (D-006 — the load-bearing invariant)
//!
//! **0 dBFS is a full-scale complex sinusoid.** A complex tone of magnitude
//! 1.0 reads 0.0 dBFS in its bin. The FFT is normalised by N and the analysis
//! window's *coherent gain* is corrected out, so the reading is independent of
//! both FFT size and window choice. Concretely, per bin `k`:
//!
//! ```text
//! P_k = 10·log10( |X_k|² / (Σ w[n])² )        [dBFS/bin]
//! ```
//!
//! since `Σw = N·CG`, dividing |X_k|² by `(Σw)²` is exactly "normalise by N,
//! then divide out the coherent gain CG". Values are floored at −200 dB
//! (§7.2), all hot-path math is `f32` (§7.7), and the scale is **dBFS/bin** —
//! noise-like signals read power per bin, not per Hz, and the UI must label it
//! as such (never dBm).
//!
//! ## Example
//!
//! ```
//! use phosphene_core::{Complex, SpectrumAnalyzer, WindowKind, fftshift_index};
//!
//! let n = 1024;
//! let mut analyzer = SpectrumAnalyzer::new(n, WindowKind::Hann).unwrap();
//!
//! // Full-scale complex tone, 100 cycles per frame (i.e. bin 100).
//! let frame: Vec<Complex<f32>> = (0..n)
//!     .map(|i| {
//!         let phase = 2.0 * std::f32::consts::PI * 100.0 * i as f32 / n as f32;
//!         Complex::new(phase.cos(), phase.sin())
//!     })
//!     .collect();
//!
//! let mut dbfs = vec![0.0f32; n];
//! analyzer.process(&frame, &mut dbfs);
//!
//! // D-006: the tone reads 0.0 dBFS in its (DC-centered) bin.
//! let peak = dbfs[fftshift_index(100, n)];
//! assert!(peak.abs() < 0.1, "full-scale tone read {peak} dBFS");
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod accumulate;
pub mod health;
pub mod spectrum;
pub mod tap;
pub mod window;

pub use accumulate::{
    AccumulatorError, LiveTrace, MaxHold, MaxHoldMode, PersistenceHistogram, DEFAULT_LEVELS,
    DEFAULT_TAU_DECAY, DEFAULT_TAU_LIVE, DEFAULT_TAU_MAX_HOLD, DEFAULT_TAU_RISE,
};
pub use health::{
    min_ring_samples, Batcher, HealthError, HealthSnapshot, PipelineHealth, PushOutcome,
    RING_HEADROOM_S,
};
pub use rustfft::num_complex::Complex;
pub use spectrum::{
    fftshift_index, CoreError, SpectrumAnalyzer, DBFS_FLOOR, DEFAULT_FFT_SIZE, MAX_FFT_SIZE,
    MIN_FFT_SIZE,
};
pub use tap::{
    AnalysisTap, BandMeta, CaptureMeta, FrameView, IqBuf, SpectrumTap, TapCaps, TapError, TimeSpan,
};
pub use window::{Window, WindowKind};
