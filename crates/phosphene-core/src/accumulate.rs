// SPDX-License-Identifier: MIT

//! Display accumulators (spec §7.3–§7.5): persistence histogram, live trace,
//! max hold.
//!
//! This is the "accumulate" stage of the §7.0 pipeline. All three accumulators
//! consume the same calibrated dBFS/bin spectra produced by
//! [`SpectrumAnalyzer`](crate::SpectrumAnalyzer) — a level entering the
//! histogram is the D-006 figure, never a rescaled one — and they share the
//! document's smoothing convention `α = 1 − exp(−Δt/τ)`.
//!
//! Two cadences run through this module, decoupled per §7.0:
//!
//! * **Per spectrum** — [`PersistenceHistogram::accumulate`],
//!   [`LiveTrace::accumulate`], [`MaxHold::accumulate`] fold one spectrum into
//!   counts / EMA / running max.
//! * **Per display frame** — [`PersistenceHistogram::tick`] folds the
//!   accumulated counts into the intensity grid with the §7.3 asymmetric
//!   exponential, and [`MaxHold::tick`] applies the D-007 decay toward the
//!   live trace. `MaxHold::tick` decays only the trace retained from earlier
//!   frames and must run **before** the frame's spectra are accumulated —
//!   §7.5's `M_k = max(M_k · d, batch max)` never decays the current batch's
//!   maximum (see [`MaxHold`]). `Δt` is passed in by the caller, which is what makes
//!   deterministic mode (fixed Δt → bit-identical grids) possible: nothing in
//!   this crate reads a clock.
//!
//! Time-constant defaults marked *tune* in §7 carry starting values from the
//! spec's ranges; final values are chosen during M1 visual tuning and every
//! constant here is runtime-settable.

use std::fmt;

use crate::spectrum::DBFS_FLOOR;

/// The power-level count L (§7.3, D-045: fixed, not user-configurable —
/// vertical resolution beyond 128 levels is not visible at any realistic
/// panel height).
pub const DEFAULT_LEVELS: usize = 128;

/// Default rise time constant τ_rise in seconds (§7.3 range 0.02–0.07, *tune*).
pub const DEFAULT_TAU_RISE: f32 = 0.05;
/// Default decay time constant τ_decay in seconds — the user "persistence"
/// (§7.3 range 0.25–2, *tune*).
pub const DEFAULT_TAU_DECAY: f32 = 1.0;
/// Default live-trace averaging time constant τ_live in seconds (§7.4).
pub const DEFAULT_TAU_LIVE: f32 = 0.1;
/// Default max-hold decay-toward-live time constant in seconds (D-007: its own
/// constant, separate from the persistence τ; *tune* — §7.5 gives no range, so
/// this is a placeholder pending M1 visual tuning).
pub const DEFAULT_TAU_MAX_HOLD: f32 = 2.0;

/// Errors from configuring an accumulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccumulatorError {
    /// The frequency-bin count must be non-zero (it is the FFT size N).
    InvalidBinCount(usize),
    /// The power-level count L must equal [`DEFAULT_LEVELS`] (D-045: fixed,
    /// not user-configurable).
    InvalidLevelCount(usize),
    /// The displayed dB range must be finite with `top > bottom`.
    InvalidDbRange {
        /// Bottom of the displayed range, dBFS.
        bottom: f32,
        /// Top of the displayed range (the reference level), dBFS.
        top: f32,
    },
    /// A time constant τ must be finite and positive.
    InvalidTimeConstant(f32),
    /// The per-spectrum interval (N / sample_rate) must be finite and positive.
    InvalidSpectrumInterval(f32),
}

impl fmt::Display for AccumulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccumulatorError::InvalidBinCount(n) => {
                write!(f, "invalid bin count {n}: must be non-zero")
            }
            AccumulatorError::InvalidLevelCount(l) => write!(
                f,
                "invalid level count {l}: must be exactly {DEFAULT_LEVELS} (D-045: fixed, not configurable)"
            ),
            AccumulatorError::InvalidDbRange { bottom, top } => write!(
                f,
                "invalid dB range [{bottom}, {top}]: must be finite with top > bottom"
            ),
            AccumulatorError::InvalidTimeConstant(tau) => {
                write!(f, "invalid time constant {tau}: must be finite and > 0")
            }
            AccumulatorError::InvalidSpectrumInterval(dt) => {
                write!(f, "invalid spectrum interval {dt}: must be finite and > 0")
            }
        }
    }
}

impl std::error::Error for AccumulatorError {}

/// `α = 1 − exp(−Δt/τ)` — the smoothing coefficient convention used
/// throughout §7.3–§7.5.
fn alpha(dt: f32, tau: f32) -> f32 {
    1.0 - (-dt / tau).exp()
}

fn check_tau(tau: f32) -> Result<f32, AccumulatorError> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(AccumulatorError::InvalidTimeConstant(tau));
    }
    Ok(tau)
}

fn check_dt(dt: f32) {
    assert!(
        dt.is_finite() && dt > 0.0,
        "Δt must be finite and positive, got {dt}"
    );
}

/// Persistence histogram accumulator (§7.3, FR-D1 — the centrepiece).
///
/// State is an intensity grid `I[k][ℓ]` over `k = 0..N−1` frequency bins and
/// `ℓ = 0..L−1` power levels spanning the displayed dB range
/// `[p_bottom, p_top]` (i.e. `[P_ref − 10·div·dB/div, P_ref]` — the caller
/// supplies the resolved endpoints). Stored row-major by bin:
/// `intensity()[k * levels + ℓ]`.
///
/// Per spectrum, [`accumulate`](Self::accumulate) quantises each bin's dBFS
/// power to a level and increments the hit count `C[k][ℓ]`. Per display frame,
/// [`tick`](Self::tick) folds counts into intensity:
///
/// ```text
/// T[k][ℓ] = C[k][ℓ] / K_total_this_frame          hit ratio 0..1
/// α_rise  = 1 − exp(−Δt / τ_rise)
/// α_decay = 1 − exp(−Δt / τ_decay)
/// I += (T ≥ I ? α_rise : α_decay) · (T − I);  clamp to [0,1];  clear C
/// ```
///
/// Rendering (colormap, gamma, the below-ε background cut) is the render
/// crate's job; this struct owns only the normative accumulation math. All
/// buffers are preallocated; `accumulate` and `tick` never allocate (§7.7).
#[derive(Debug, Clone)]
pub struct PersistenceHistogram {
    bins: usize,
    levels: usize,
    p_bottom: f32,
    /// `(L−1) / (p_top − p_bottom)` — dB → level scale.
    level_scale: f32,
    p_top: f32,
    tau_rise: f32,
    tau_decay: f32,
    counts: Vec<u32>,
    intensity: Vec<f32>,
    spectra_since_tick: u32,
}

impl PersistenceHistogram {
    /// Build a histogram over `bins` frequency bins × `levels` power levels
    /// spanning `[p_bottom, p_top]` dBFS, with default time constants
    /// ([`DEFAULT_TAU_RISE`], [`DEFAULT_TAU_DECAY`]).
    pub fn new(
        bins: usize,
        levels: usize,
        p_bottom: f32,
        p_top: f32,
    ) -> Result<Self, AccumulatorError> {
        if bins == 0 {
            return Err(AccumulatorError::InvalidBinCount(bins));
        }
        // D-045: the level count is fixed, not user-configurable — accepting
        // any nonzero value here would silently reopen the exact
        // configurability the owner ruled to drop.
        if levels != DEFAULT_LEVELS {
            return Err(AccumulatorError::InvalidLevelCount(levels));
        }
        if !p_bottom.is_finite() || !p_top.is_finite() || p_top <= p_bottom {
            return Err(AccumulatorError::InvalidDbRange {
                bottom: p_bottom,
                top: p_top,
            });
        }
        Ok(PersistenceHistogram {
            bins,
            levels,
            p_bottom,
            level_scale: (levels - 1) as f32 / (p_top - p_bottom),
            p_top,
            tau_rise: DEFAULT_TAU_RISE,
            tau_decay: DEFAULT_TAU_DECAY,
            counts: vec![0; bins * levels],
            intensity: vec![0.0; bins * levels],
            spectra_since_tick: 0,
        })
    }

    /// Frequency-bin count N.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Power-level count L.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Bottom of the displayed dB range, dBFS.
    pub fn p_bottom(&self) -> f32 {
        self.p_bottom
    }

    /// Top of the displayed dB range (reference level), dBFS.
    pub fn p_top(&self) -> f32 {
        self.p_top
    }

    /// Rise time constant τ_rise, seconds.
    pub fn tau_rise(&self) -> f32 {
        self.tau_rise
    }

    /// Decay time constant τ_decay (the user "persistence"), seconds.
    pub fn tau_decay(&self) -> f32 {
        self.tau_decay
    }

    /// Set τ_rise (finite, > 0).
    ///
    /// Test-only today, and deliberately so (audited by AUDIT-1, kept by
    /// M1-K): §7.3 marks τ_rise **tune** — not a user control, so unlike
    /// τ_decay (D-035) no requirement wires it — but the accumulator's two
    /// time constants belong together, and the step-response tests drive
    /// the rise side through this setter.
    pub fn set_tau_rise(&mut self, tau: f32) -> Result<(), AccumulatorError> {
        self.tau_rise = check_tau(tau)?;
        Ok(())
    }

    /// Set τ_decay — the user-facing persistence control (finite, > 0).
    pub fn set_tau_decay(&mut self, tau: f32) -> Result<(), AccumulatorError> {
        self.tau_decay = check_tau(tau)?;
        Ok(())
    }

    /// The §7.3 level quantiser:
    /// `ℓ = clamp( round( (P − P_bottom) / (P_top − P_bottom) · (L−1) ), 0, L−1 )`.
    pub fn level_index(&self, p_dbfs: f32) -> usize {
        let raw = ((p_dbfs - self.p_bottom) * self.level_scale).round();
        raw.clamp(0.0, (self.levels - 1) as f32) as usize
    }

    /// Fold one dBFS/bin spectrum into the hit counts.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum.len() != bins` — a programmer error.
    pub fn accumulate(&mut self, spectrum: &[f32]) {
        assert_eq!(
            spectrum.len(),
            self.bins,
            "spectrum length must equal bin count"
        );
        for (k, &p) in spectrum.iter().enumerate() {
            let level = self.level_index(p);
            self.counts[k * self.levels + level] += 1;
        }
        self.spectra_since_tick += 1;
    }

    /// Fold the counts accumulated since the last tick into the intensity
    /// grid with the §7.3 asymmetric exponential, then clear them. `dt` is the
    /// display-frame interval Δt in seconds, supplied by the caller.
    ///
    /// If no spectra arrived since the last tick, `T = C / K_total` has no
    /// value (`K_total = 0`) and the fold is skipped: the grid holds its state
    /// rather than fabricating an all-zero observation. §7.3 defines the fold
    /// only for `K_total ≥ 1`; this guard is defensive, not display math.
    ///
    /// # Panics
    ///
    /// Panics if `dt` is not finite and positive.
    pub fn tick(&mut self, dt: f32) {
        check_dt(dt);
        if self.spectra_since_tick == 0 {
            return;
        }
        let k_total = self.spectra_since_tick as f32;
        let a_rise = alpha(dt, self.tau_rise);
        let a_decay = alpha(dt, self.tau_decay);
        for (i, c) in self.intensity.iter_mut().zip(self.counts.iter_mut()) {
            let t = *c as f32 / k_total;
            let a = if t >= *i { a_rise } else { a_decay };
            *i = (*i + a * (t - *i)).clamp(0.0, 1.0);
            *c = 0;
        }
        self.spectra_since_tick = 0;
    }

    /// The intensity grid `I ∈ [0,1]`, row-major by bin:
    /// `intensity()[k * levels + ℓ]`.
    pub fn intensity(&self) -> &[f32] {
        &self.intensity
    }

    /// Intensity of one cell.
    ///
    /// # Panics
    ///
    /// Panics if `bin >= bins` or `level >= levels`.
    pub fn intensity_at(&self, bin: usize, level: usize) -> f32 {
        assert!(bin < self.bins, "bin {bin} out of range");
        assert!(level < self.levels, "level {level} out of range");
        self.intensity[bin * self.levels + level]
    }
}

/// Live spectrum trace (§7.4, FR-D2): per-bin EMA across spectra,
/// `L_k ← L_k + α_live·(P_k − L_k)`, **computed incrementally per batch in
/// closed form** — [`accumulate_batch`](Self::accumulate_batch) is the
/// production path.
///
/// `α_live` derives from τ_live through the document's convention
/// `α = 1 − exp(−Δt/τ)` with Δt the per-spectrum interval (N / sample_rate,
/// §7.1's non-overlapping frames), which the caller supplies at construction.
///
/// ## The closed form (§7.4)
///
/// Unrolling the recurrence over a batch of K spectra `P_1..P_K` with
/// `r = 1 − α_live`:
///
/// ```text
/// L ← r^K·L + (1 − r^K)·M,   M = Σᵢ r^(K−i)·P_i  /  Σᵢ r^(K−i)
/// ```
///
/// `M` is the batch's exponentially-weighted mean, computed in the one pass
/// over the batch data that the pipeline already makes; the blend against the
/// trace then costs O(N) **per batch, not per spectrum** — at NFR-P1 rates
/// (~19.5k spectra/s at the D-012 default FFT size) that is the difference
/// the guarantee needs. [`accumulate`](Self::accumulate) applies the naive
/// per-spectrum recurrence and is retained **as the test oracle only**: the
/// two are identical in exact arithmetic, and the seal asserts they agree in
/// `f32` within tolerance over realistic batch lengths.
///
/// The trace seeds from the first observed spectrum rather than ramping up
/// from the −200 dB floor, so the EMA converges on signal statistics, not on
/// an artefact of the empty initial state.
#[derive(Debug, Clone)]
pub struct LiveTrace {
    bins: usize,
    spectrum_interval: f32,
    tau: f32,
    alpha: f32,
    trace: Vec<f32>,
    /// Batch-local Horner accumulator for `Σ r^(K−i)·P_i` (preallocated —
    /// §7.7 forbids hot-path allocation).
    weighted_sum: Vec<f32>,
    seeded: bool,
}

impl LiveTrace {
    /// Build a live trace over `bins` frequency bins. `spectrum_interval` is
    /// the time between consecutive spectra in seconds (N / sample_rate);
    /// `tau` is the averaging time constant τ_live ([`DEFAULT_TAU_LIVE`] is
    /// the spec default).
    pub fn new(bins: usize, spectrum_interval: f32, tau: f32) -> Result<Self, AccumulatorError> {
        if bins == 0 {
            return Err(AccumulatorError::InvalidBinCount(bins));
        }
        if !spectrum_interval.is_finite() || spectrum_interval <= 0.0 {
            return Err(AccumulatorError::InvalidSpectrumInterval(spectrum_interval));
        }
        let tau = check_tau(tau)?;
        Ok(LiveTrace {
            bins,
            spectrum_interval,
            tau,
            alpha: alpha(spectrum_interval, tau),
            trace: vec![DBFS_FLOOR; bins],
            weighted_sum: vec![0.0; bins],
            seeded: false,
        })
    }

    /// Frequency-bin count N.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Averaging time constant τ_live, seconds.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Set τ_live (finite, > 0). Takes effect from the next spectrum.
    pub fn set_tau(&mut self, tau: f32) -> Result<(), AccumulatorError> {
        self.tau = check_tau(tau)?;
        self.alpha = alpha(self.spectrum_interval, self.tau);
        Ok(())
    }

    /// Fold one batch of dBFS/bin spectra into the EMA via the §7.4 closed
    /// form — **the production path**. `spectra` is K spectra stored
    /// back-to-back (oldest first), `K·bins` values in all; the trace is
    /// blended exactly once, so the per-trace cost is O(N) per batch. An
    /// empty batch is a no-op. Allocation-free.
    ///
    /// Equivalent in exact arithmetic to K calls of
    /// [`accumulate`](Self::accumulate); for K = 1 the two are identical
    /// even in `f32`.
    ///
    /// # Panics
    ///
    /// Panics if `spectra.len()` is not a multiple of `bins`.
    pub fn accumulate_batch(&mut self, spectra: &[f32]) {
        assert_eq!(
            spectra.len() % self.bins,
            0,
            "batch length must be a whole number of spectra"
        );
        let mut chunks = spectra.chunks_exact(self.bins);
        if !self.seeded {
            match chunks.next() {
                Some(first) => {
                    self.trace.copy_from_slice(first);
                    self.seeded = true;
                }
                None => return,
            }
        }
        let Some(first) = chunks.next() else {
            return;
        };

        // One pass: Horner-fold the batch into Σ r^(K−i)·P_i.
        self.weighted_sum.copy_from_slice(first);
        let mut k = 1u32;
        let r = 1.0 - self.alpha as f64;
        let r32 = r as f32;
        for spectrum in chunks {
            for (s, &p) in self.weighted_sum.iter_mut().zip(spectrum) {
                *s = r32 * *s + p;
            }
            k += 1;
        }

        // Then blend, once: L ← r^K·L + (1 − r^K)·M. The scalar factors are
        // evaluated in f64 so K-fold exponentiation costs no precision.
        let r_pow_k = r.powi(k as i32);
        let beta = (1.0 - r_pow_k) as f32;
        // Σ r^(K−i) = (1 − r^K) / (1 − r) = (1 − r^K) / α.
        let inv_weight = ((1.0 - r) / (1.0 - r_pow_k)) as f32;
        for (l, &s) in self.trace.iter_mut().zip(&self.weighted_sum) {
            *l += beta * (s * inv_weight - *l);
        }
    }

    /// Fold one dBFS/bin spectrum into the EMA via the naive per-spectrum
    /// recurrence — the normative §7.4 reference, retained **as the test
    /// oracle** for [`accumulate_batch`](Self::accumulate_batch). Production
    /// code must use the batch path: per-spectrum trace updates are exactly
    /// the cost §7.4 rules out at NFR-P1 rates.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum.len() != bins`.
    pub fn accumulate(&mut self, spectrum: &[f32]) {
        assert_eq!(
            spectrum.len(),
            self.bins,
            "spectrum length must equal bin count"
        );
        if self.seeded {
            for (l, &p) in self.trace.iter_mut().zip(spectrum) {
                *l += self.alpha * (p - *l);
            }
        } else {
            self.trace.copy_from_slice(spectrum);
            self.seeded = true;
        }
    }

    /// The live trace in dBFS/bin ([`DBFS_FLOOR`] before any spectrum).
    pub fn trace(&self) -> &[f32] {
        &self.trace
    }
}

/// Max-hold decay behaviour (§7.5, D-007).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxHoldMode {
    /// Decay toward the live trace at the accumulator's own time constant —
    /// **the default** (D-007).
    DecayTowardLive,
    /// Pure hold: the trace never decreases until [`MaxHold::reset`].
    PureHold,
}

/// Max-hold trace (§7.5, FR-D3): `M_k = max(M_k · d, max over batch of P_k)`,
/// with the D-007 mode choice.
///
/// The decay factor `d` applies **only to the retained `M_k`** — the
/// current batch's maximum is never decayed. The calling order per display
/// frame therefore matters and is part of this type's contract:
///
/// 1. [`tick`](Self::tick) **first** — decays the trace retained from
///    previous frames;
/// 2. then [`accumulate`](Self::accumulate) the frame's spectra.
///
/// A peak appearing for the first time in the current frame's batch is thus
/// reported at its **exact, undecayed** value on that frame, and only starts
/// relaxing on subsequent frames. Accumulate-then-tick would pull a brand-new
/// peak toward the live trace before it was ever displayed, so the true peak
/// value would never be read — the inversion §7.5's formula rules out.
///
/// * [`MaxHoldMode::PureHold`] — §7.5's `d = 1`: the per-bin running max,
///   never decreasing until [`reset`](Self::reset); `tick` is a no-op, so
///   the ordering is irrelevant in this mode.
/// * [`MaxHoldMode::DecayTowardLive`] (default) — per display frame,
///   `M_k ← M_k + α·(L_k − M_k)` with `α = 1 − exp(−Δt/τ)` relaxes the
///   retained trace toward the live trace at the accumulator's own time
///   constant τ, separate from the persistence τ.
///
/// Like [`LiveTrace`], the trace seeds from the first observed spectrum
/// (after construction or reset); until then `tick` is a no-op, so leading
/// with a tick on the very first frame is safe.
#[derive(Debug, Clone)]
pub struct MaxHold {
    bins: usize,
    mode: MaxHoldMode,
    tau: f32,
    trace: Vec<f32>,
    seeded: bool,
}

impl MaxHold {
    /// Build a max-hold trace over `bins` frequency bins, in the default
    /// decay-toward-live mode (D-007) with τ = [`DEFAULT_TAU_MAX_HOLD`].
    pub fn new(bins: usize) -> Result<Self, AccumulatorError> {
        if bins == 0 {
            return Err(AccumulatorError::InvalidBinCount(bins));
        }
        Ok(MaxHold {
            bins,
            mode: MaxHoldMode::DecayTowardLive,
            tau: DEFAULT_TAU_MAX_HOLD,
            trace: vec![DBFS_FLOOR; bins],
            seeded: false,
        })
    }

    /// Frequency-bin count N.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// The current decay mode.
    pub fn mode(&self) -> MaxHoldMode {
        self.mode
    }

    /// Select the decay mode (user-selectable per D-007). The trace carries
    /// over; only future ticks change behaviour.
    pub fn set_mode(&mut self, mode: MaxHoldMode) {
        self.mode = mode;
    }

    /// Decay-toward-live time constant, seconds.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Set the decay-toward-live time constant (finite, > 0).
    pub fn set_tau(&mut self, tau: f32) -> Result<(), AccumulatorError> {
        self.tau = check_tau(tau)?;
        Ok(())
    }

    /// Fold one dBFS/bin spectrum: per-bin `M_k = max(M_k, P_k)`.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum.len() != bins`.
    pub fn accumulate(&mut self, spectrum: &[f32]) {
        assert_eq!(
            spectrum.len(),
            self.bins,
            "spectrum length must equal bin count"
        );
        if self.seeded {
            for (m, &p) in self.trace.iter_mut().zip(spectrum) {
                *m = m.max(p);
            }
        } else {
            self.trace.copy_from_slice(spectrum);
            self.seeded = true;
        }
    }

    /// Apply one display frame of decay to the **retained** trace — call at
    /// the START of the frame, before [`accumulate`](Self::accumulate) folds
    /// in the frame's spectra, so the current batch's maxima are never
    /// decayed (§7.5: `d` multiplies `M_k` only). In
    /// [`MaxHoldMode::DecayTowardLive`], relaxes each bin toward `live` (the
    /// [`LiveTrace`] output) by `α = 1 − exp(−Δt/τ)`; in
    /// [`MaxHoldMode::PureHold`] this is a no-op. Before the first spectrum
    /// seeds the trace (and after [`reset`](Self::reset)) there is nothing
    /// retained to decay and this is a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `dt` is not finite and positive, or if
    /// `live.len() != bins`.
    pub fn tick(&mut self, dt: f32, live: &[f32]) {
        check_dt(dt);
        assert_eq!(live.len(), self.bins, "live length must equal bin count");
        if self.mode != MaxHoldMode::DecayTowardLive || !self.seeded {
            return;
        }
        let a = alpha(dt, self.tau);
        for (m, &l) in self.trace.iter_mut().zip(live) {
            *m += a * (l - *m);
        }
    }

    /// Reset (FR-D3's reset key): clear the trace to [`DBFS_FLOOR`] and
    /// re-seed from the next spectrum.
    pub fn reset(&mut self) {
        self.trace.fill(DBFS_FLOOR);
        self.seeded = false;
    }

    /// The max-hold trace in dBFS/bin ([`DBFS_FLOOR`] before any spectrum).
    pub fn trace(&self) -> &[f32] {
        &self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_rejects_invalid_config() {
        assert_eq!(
            PersistenceHistogram::new(0, 128, -100.0, 0.0).err(),
            Some(AccumulatorError::InvalidBinCount(0)),
        );
        // D-045: the level count is fixed at DEFAULT_LEVELS (128) — anything
        // else is rejected, not just the degenerate zero case. MIN_LEVELS/
        // MAX_LEVELS are gone; this is a hard equality, not a range.
        for bad_levels in [0, 1, 64, 127, 129, 256, 257] {
            assert_eq!(
                PersistenceHistogram::new(16, bad_levels, -100.0, 0.0).err(),
                Some(AccumulatorError::InvalidLevelCount(bad_levels)),
                "level count {bad_levels} must be rejected (only {DEFAULT_LEVELS} is accepted)"
            );
        }
        assert!(PersistenceHistogram::new(16, DEFAULT_LEVELS, -100.0, 0.0).is_ok());
        for (bottom, top) in [(0.0, 0.0), (0.0, -100.0), (f32::NEG_INFINITY, 0.0)] {
            assert_eq!(
                PersistenceHistogram::new(16, 128, bottom, top).err(),
                Some(AccumulatorError::InvalidDbRange { bottom, top }),
            );
        }
        let mut h = PersistenceHistogram::new(16, 128, -100.0, 0.0).unwrap();
        for bad_tau in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(h.set_tau_rise(bad_tau).is_err());
            assert!(h.set_tau_decay(bad_tau).is_err());
        }
    }

    #[test]
    fn level_index_maps_endpoints_and_clamps() {
        let h = PersistenceHistogram::new(4, 128, -100.0, 0.0).unwrap();
        assert_eq!(h.level_index(-100.0), 0);
        assert_eq!(h.level_index(0.0), 127);
        // Below/above the displayed range clamps to the edge levels (§7.3).
        assert_eq!(h.level_index(-200.0), 0);
        assert_eq!(h.level_index(10.0), 127);
        // One level spans 100/127 dB here; the midpoint rounds to nearest.
        assert_eq!(h.level_index(-50.0), 64); // 0.5·127 = 63.5 → rounds to 64
    }

    #[test]
    fn histogram_counts_land_in_the_quantised_cell() {
        let mut h = PersistenceHistogram::new(4, 128, -100.0, 0.0).unwrap();
        h.accumulate(&[-100.0, -20.0, -100.0, -100.0]);
        h.tick(1.0 / 60.0);
        let level = h.level_index(-20.0);
        assert!(h.intensity_at(1, level) > 0.0);
        // The same bin's other levels saw no hits.
        assert_eq!(h.intensity_at(1, 0), 0.0);
    }

    #[test]
    fn live_trace_rejects_invalid_config() {
        assert!(LiveTrace::new(0, 1e-3, 0.1).is_err());
        for bad in [0.0, -1.0, f32::NAN] {
            assert!(LiveTrace::new(8, bad, 0.1).is_err());
            assert!(LiveTrace::new(8, 1e-3, bad).is_err());
        }
    }

    #[test]
    fn live_trace_seeds_then_follows_the_ema_recurrence() {
        let mut lt = LiveTrace::new(2, 1e-3, DEFAULT_TAU_LIVE).unwrap();
        assert_eq!(lt.trace(), &[DBFS_FLOOR, DBFS_FLOOR]);
        lt.accumulate(&[-30.0, -60.0]);
        assert_eq!(lt.trace(), &[-30.0, -60.0]);
        let a = alpha(1e-3, DEFAULT_TAU_LIVE);
        lt.accumulate(&[-20.0, -60.0]);
        assert_eq!(lt.trace()[0], -30.0 + a * (-20.0 - -30.0));
        assert_eq!(lt.trace()[1], -60.0);
    }

    #[test]
    fn max_hold_defaults_to_decay_toward_live() {
        // D-007: decay-toward-live is the default; its τ is its own constant.
        let mh = MaxHold::new(8).unwrap();
        assert_eq!(mh.mode(), MaxHoldMode::DecayTowardLive);
        assert_eq!(mh.tau(), DEFAULT_TAU_MAX_HOLD);
    }

    #[test]
    fn accumulator_errors_name_the_offending_value() {
        assert!(AccumulatorError::InvalidLevelCount(63)
            .to_string()
            .contains("63"));
        assert!(AccumulatorError::InvalidTimeConstant(-2.0)
            .to_string()
            .contains("-2"));
    }
}
