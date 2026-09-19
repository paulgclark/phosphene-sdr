// SPDX-License-Identifier: MIT

//! The deterministic signal generator (`--sdr siggen`) — demo mode and the
//! known-truth test fixture for every later lane (plan M0-B).
//!
//! A [`SigGen`] synthesises a scene of independent components summed into one
//! `cf32` stream:
//!
//! * **Tones** ([`ToneConfig`]) — fixed-frequency complex sinusoids at chosen
//!   offsets and levels. A 0 dBFS tone has magnitude 1.0, so through
//!   `phosphene-core` it reads 0.0 dBFS in its bin — the two crates share the
//!   D-006 calibration by construction.
//! * **Noise floor** ([`NoiseConfig`]) — circular complex white Gaussian noise
//!   at a chosen *total* power. Displayed per-bin level follows the D-006
//!   dBFS/bin convention: `level + 10·log10(ENBW/N)` dB (see
//!   `phosphene_core::Window::enbw_bins`).
//! * **Chirps** ([`ChirpConfig`]) — linear frequency sweeps, repeating.
//! * **Bursty frequency hoppers** ([`HopperConfig`]) — a burst of `dwell_s`
//!   seconds at one of a set of offsets, once per `period_s`, silent between
//!   bursts. This is the component that shows off what a persistence display
//!   exists for: short transients visible rather than statistically invisible.
//! * **OOK/ASK carriers** ([`OokConfig`]) — a continuous-phase carrier keyed
//!   on and off once per symbol by a deterministic bit stream (D-037: the
//!   known-ground-truth fixture CL-1's classifier develops against).
//! * **Phase-continuous M-ary FSK carriers** ([`FskConfig`]) — a carrier
//!   whose instantaneous frequency steps between `levels` tones spaced
//!   `deviation_hz` apart once per symbol, chosen by a deterministic symbol
//!   stream, with no phase reset at symbol boundaries (D-037).
//!
//! ## Determinism (the contract every later lane's tests lean on)
//!
//! The sample stream is a pure function of the configuration (seed included):
//! the same [`SigGenConfig`] yields a **bit-identical** stream on every run,
//! and the stream is invariant to how it is chunked — [`SigGen::fill`] called
//! with any sequence of buffer sizes produces the same concatenated samples.
//! All oscillator state advances in `f64` and is quantised to `f32` only at
//! the output.
//!
//! Levels are specified in dBFS per D-006 (0 dBFS = full-scale complex
//! sinusoid). The generator does not clip: a scene whose components sum above
//! magnitude 1.0 simply exceeds full scale.

mod rng;

use phosphene_core::Complex;

use crate::source::{
    Control, ControlCaps, SampleSink, SampleSource, SinkFlow, SourceDesc, SourceError, SourceMeta,
};
use rng::{mix, SplitMix64};

/// A fixed complex sinusoid.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneConfig {
    /// Frequency offset from center, Hz. Must lie within ±sample_rate/2.
    pub offset_hz: f64,
    /// Level in dBFS (0.0 = full scale, D-006).
    pub level_dbfs: f32,
}

/// A circular complex white Gaussian noise floor.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseConfig {
    /// Total noise power in dBFS — i.e. mean `|x|²` relative to a full-scale
    /// sinusoid. The *displayed* per-bin floor is lower by `10·log10(N/ENBW)`.
    pub level_dbfs: f32,
}

/// A repeating linear chirp.
#[derive(Debug, Clone, PartialEq)]
pub struct ChirpConfig {
    /// Sweep start offset from center, Hz (within ±sample_rate/2).
    pub start_offset_hz: f64,
    /// Sweep end offset from center, Hz (within ±sample_rate/2).
    pub stop_offset_hz: f64,
    /// Seconds per sweep; the chirp then restarts at `start_offset_hz`.
    pub sweep_time_s: f64,
    /// Level in dBFS.
    pub level_dbfs: f32,
}

/// How a hopper walks its frequency list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HopOrder {
    /// Visit the configured offsets in order, wrapping around.
    Cycle,
    /// Deterministic pseudorandom order derived from the generator seed.
    Pseudorandom,
}

/// A bursty frequency hopper: one burst per period, silent the rest of it.
#[derive(Debug, Clone, PartialEq)]
pub struct HopperConfig {
    /// The offsets (Hz, within ±sample_rate/2) the hopper visits.
    pub offsets_hz: Vec<f64>,
    /// Burst length in seconds — the transmit time at each visited offset.
    pub dwell_s: f64,
    /// Seconds from one hop to the next; must be ≥ `dwell_s`. The hopper is
    /// silent for `period_s − dwell_s` after each burst.
    pub period_s: f64,
    /// Level in dBFS while bursting.
    pub level_dbfs: f32,
    /// Hop order.
    pub order: HopOrder,
}

/// A keyed on-off carrier: OOK/ASK, the ground truth CL-1's classifier
/// develops against (D-037). The carrier's phase runs continuously; only its
/// amplitude is keyed, once per symbol, by a deterministic bit stream drawn
/// from the generator's own seeded RNG.
#[derive(Debug, Clone, PartialEq)]
pub struct OokConfig {
    /// Carrier offset from center, Hz (within ±sample_rate/2).
    pub offset_hz: f64,
    /// Symbol rate, baud (symbols/second). Must be finite, positive, and no
    /// faster than one symbol per sample at the configured rate.
    pub symbol_rate_bd: f64,
    /// Level in dBFS while the keyed bit is on (0.0 = full scale, D-006).
    pub level_dbfs: f32,
}

/// A phase-continuous M-ary FSK carrier: the other half of the D-037 ground
/// truth. Each symbol period, the carrier's instantaneous frequency jumps to
/// one of `levels` tones spaced `deviation_hz` apart, chosen by a
/// deterministic symbol stream; phase is never reset at a symbol boundary
/// (continuous-phase FSK), so amplitude stays constant throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct FskConfig {
    /// Center offset from the generator's center, Hz — the midpoint of the
    /// tone ladder (within ±sample_rate/2, as is every individual tone).
    pub offset_hz: f64,
    /// Number of frequency levels M (at least 2).
    pub levels: u32,
    /// Spacing between adjacent frequency levels, Hz. Must be finite and
    /// positive.
    pub deviation_hz: f64,
    /// Symbol rate, baud. Must be finite, positive, and no faster than one
    /// symbol per sample at the configured rate.
    pub symbol_rate_bd: f64,
    /// Level in dBFS (0.0 = full scale, D-006).
    pub level_dbfs: f32,
}

/// Full description of a signal-generator scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SigGenConfig {
    /// Sample rate in Hz (must be finite and positive).
    pub sample_rate_hz: f64,
    /// Nominal center frequency for axis labelling, if any (FR-C3: `None`
    /// labels the axis in relative Hz).
    pub center_freq_hz: Option<f64>,
    /// Seed for every stochastic component (noise, pseudorandom hop order).
    pub seed: u64,
    /// Fixed tones.
    pub tones: Vec<ToneConfig>,
    /// Noise floor, if any.
    pub noise: Option<NoiseConfig>,
    /// Repeating chirps.
    pub chirps: Vec<ChirpConfig>,
    /// Bursty frequency hoppers.
    pub hoppers: Vec<HopperConfig>,
    /// Keyed on-off carriers (D-037 ground truth for CL-1).
    pub ook: Vec<OokConfig>,
    /// Phase-continuous M-ary FSK carriers (D-037 ground truth for CL-1).
    pub fsk: Vec<FskConfig>,
}

impl SigGenConfig {
    /// An empty (silent) scene at the given sample rate, seed 0 — the starting
    /// point tests build on.
    pub fn new(sample_rate_hz: f64) -> Self {
        SigGenConfig {
            sample_rate_hz,
            center_freq_hz: None,
            seed: 0,
            tones: Vec::new(),
            noise: None,
            chirps: Vec::new(),
            hoppers: Vec::new(),
            ook: Vec::new(),
            fsk: Vec::new(),
        }
    }

    /// The default demo scene for `--sdr siggen`: two tones, a noise floor, a
    /// slow full-span chirp, and a pseudorandom bursty hopper — everything the
    /// display has to prove it can show (plan M0).
    pub fn demo() -> Self {
        SigGenConfig {
            sample_rate_hz: 2_048_000.0,
            center_freq_hz: None,
            seed: 0x0DD_BA11,
            tones: vec![
                ToneConfig {
                    offset_hz: 300_000.0,
                    level_dbfs: -20.0,
                },
                ToneConfig {
                    offset_hz: -500_000.0,
                    level_dbfs: -40.0,
                },
            ],
            noise: Some(NoiseConfig { level_dbfs: -70.0 }),
            chirps: vec![ChirpConfig {
                start_offset_hz: -800_000.0,
                stop_offset_hz: 800_000.0,
                sweep_time_s: 2.0,
                level_dbfs: -30.0,
            }],
            hoppers: vec![HopperConfig {
                offsets_hz: vec![-600_000.0, -200_000.0, 200_000.0, 600_000.0],
                dwell_s: 0.005,
                period_s: 0.015,
                level_dbfs: -25.0,
                order: HopOrder::Pseudorandom,
            }],
            ook: Vec::new(),
            fsk: Vec::new(),
        }
    }
}

/// dBFS level → linear magnitude of a sinusoid at that level.
fn magnitude(level_dbfs: f32) -> f64 {
    10f64.powf(level_dbfs as f64 / 20.0)
}

/// Advance-and-wrap for a phase accumulator. Increments are bounded to
/// [−π, π] by the Nyquist checks at construction, so one correction suffices
/// to keep the phase in [0, 2π).
fn wrap_phase(p: f64) -> f64 {
    use std::f64::consts::TAU;
    if p >= TAU {
        p - TAU
    } else if p < 0.0 {
        p + TAU
    } else {
        p
    }
}

/// `mag·e^{jφ}` quantised to the output format.
fn polar(mag: f64, phase: f64) -> Complex<f32> {
    Complex::new((mag * phase.cos()) as f32, (mag * phase.sin()) as f32)
}

/// Decorrelated per-component seed streams derived from the scene seed.
fn stream_seed(seed: u64, stream: u64) -> u64 {
    mix(seed ^ mix(stream))
}

/// Stream-id offsets keeping OOK/FSK symbol streams decorrelated from the
/// existing noise (stream `0`) and hopper (streams `1 + i`) streams, with
/// enough headroom that no realistic scene count collides.
const OOK_STREAM_BASE: u64 = 1_000_000_000;
const FSK_STREAM_BASE: u64 = 2_000_000_000;

/// Seconds → samples at `rate`, requiring at least one full sample period.
fn to_symbol_samples(what: &str, symbol_rate_bd: f64, rate: f64) -> Result<u64, SourceError> {
    if !symbol_rate_bd.is_finite() || symbol_rate_bd <= 0.0 {
        return Err(SourceError::InvalidConfig(format!(
            "{what} symbol rate must be finite and positive, got {symbol_rate_bd} Bd"
        )));
    }
    let samples = to_samples(1.0 / symbol_rate_bd, rate);
    if samples < 1 {
        return Err(SourceError::InvalidConfig(format!(
            "{what} symbol rate {symbol_rate_bd} Bd is faster than one sample at {rate} Hz"
        )));
    }
    Ok(samples)
}

/// Seconds → samples at `rate`, rounded to nearest.
fn to_samples(seconds: f64, rate: f64) -> u64 {
    (seconds * rate).round() as u64
}

#[derive(Debug, Clone)]
struct ToneState {
    magnitude: f64,
    phase: f64,
    inc: f64,
}

#[derive(Debug, Clone)]
struct ChirpState {
    magnitude: f64,
    phase: f64,
    /// Phase increment at sweep position 0.
    inc0: f64,
    /// Per-sample change of the increment (linear sweep).
    dinc: f64,
    sweep_len: u64,
    pos: u64,
}

#[derive(Debug, Clone)]
struct HopperState {
    magnitude: f64,
    phase: f64,
    /// Phase increment per configured offset.
    incs: Vec<f64>,
    dwell: u64,
    period: u64,
    /// Sample position within the current period.
    pos: u64,
    /// Which hop we are on (monotonic).
    slot: u64,
    order: HopOrder,
    hash_seed: u64,
    current_inc: f64,
}

#[derive(Debug, Clone)]
struct OokState {
    magnitude: f64,
    phase: f64,
    inc: f64,
    samples_per_symbol: u64,
    /// Sample position within the current symbol.
    pos: u64,
    /// Which symbol we are on (monotonic).
    slot: u64,
    hash_seed: u64,
    /// Amplitude multiplier for the current symbol: 0.0 (off) or 1.0 (on).
    current_level: f64,
}

#[derive(Debug, Clone)]
struct FskState {
    magnitude: f64,
    phase: f64,
    /// Phase increment per configured level.
    incs: Vec<f64>,
    samples_per_symbol: u64,
    levels: u32,
    pos: u64,
    slot: u64,
    hash_seed: u64,
    current_inc: f64,
}

#[derive(Debug, Clone)]
struct NoiseState {
    /// Standard deviation of each of I and Q (σ²/2 per component).
    sigma: f64,
    rng: SplitMix64,
}

/// The signal generator. See the [module docs](self) for the scene model and
/// the determinism contract.
#[derive(Debug, Clone)]
pub struct SigGen {
    config: SigGenConfig,
    tones: Vec<ToneState>,
    chirps: Vec<ChirpState>,
    hoppers: Vec<HopperState>,
    ook: Vec<OokState>,
    fsk: Vec<FskState>,
    noise: Option<NoiseState>,
}

/// Samples per [`SampleSink::push`] when streaming.
const STREAM_CHUNK: usize = 4096;

impl SigGen {
    /// Build a generator, validating the whole scene up front.
    pub fn new(config: SigGenConfig) -> Result<Self, SourceError> {
        let rate = config.sample_rate_hz;
        if !rate.is_finite() || rate <= 0.0 {
            return Err(SourceError::InvalidConfig(format!(
                "sample rate must be finite and positive, got {rate} Hz"
            )));
        }
        if let Some(center) = config.center_freq_hz {
            if !center.is_finite() {
                return Err(SourceError::InvalidConfig(format!(
                    "center frequency must be finite, got {center} Hz"
                )));
            }
        }
        let nyquist = rate / 2.0;
        let check_offset = |what: String, offset: f64| -> Result<(), SourceError> {
            if !offset.is_finite() || offset.abs() > nyquist {
                return Err(SourceError::InvalidConfig(format!(
                    "{what} offset {offset} Hz is outside ±{nyquist} Hz \
                     (Nyquist at {rate} Hz sample rate)"
                )));
            }
            Ok(())
        };
        let check_level = |what: String, level: f32| -> Result<(), SourceError> {
            if !level.is_finite() {
                return Err(SourceError::InvalidConfig(format!(
                    "{what} level must be finite, got {level} dBFS"
                )));
            }
            Ok(())
        };

        /// Phase increment per sample for an offset at the given rate.
        fn phase_inc(offset_hz: f64, rate: f64) -> f64 {
            std::f64::consts::TAU * offset_hz / rate
        }

        let mut tones = Vec::with_capacity(config.tones.len());
        for (i, t) in config.tones.iter().enumerate() {
            check_offset(format!("tone {i}"), t.offset_hz)?;
            check_level(format!("tone {i}"), t.level_dbfs)?;
            tones.push(ToneState {
                magnitude: magnitude(t.level_dbfs),
                phase: 0.0,
                inc: phase_inc(t.offset_hz, rate),
            });
        }

        let mut chirps = Vec::with_capacity(config.chirps.len());
        for (i, c) in config.chirps.iter().enumerate() {
            check_offset(format!("chirp {i} start"), c.start_offset_hz)?;
            check_offset(format!("chirp {i} stop"), c.stop_offset_hz)?;
            check_level(format!("chirp {i}"), c.level_dbfs)?;
            if !c.sweep_time_s.is_finite() || c.sweep_time_s <= 0.0 {
                return Err(SourceError::InvalidConfig(format!(
                    "chirp {i} sweep time must be finite and positive, got {} s",
                    c.sweep_time_s
                )));
            }
            let sweep_len = to_samples(c.sweep_time_s, rate);
            if sweep_len < 1 {
                return Err(SourceError::InvalidConfig(format!(
                    "chirp {i} sweep time {} s is shorter than one sample at {rate} Hz",
                    c.sweep_time_s
                )));
            }
            let inc0 = phase_inc(c.start_offset_hz, rate);
            let inc1 = phase_inc(c.stop_offset_hz, rate);
            chirps.push(ChirpState {
                magnitude: magnitude(c.level_dbfs),
                phase: 0.0,
                inc0,
                dinc: (inc1 - inc0) / sweep_len as f64,
                sweep_len,
                pos: 0,
            });
        }

        let mut hoppers = Vec::with_capacity(config.hoppers.len());
        for (i, h) in config.hoppers.iter().enumerate() {
            if h.offsets_hz.is_empty() {
                return Err(SourceError::InvalidConfig(format!(
                    "hopper {i} has no offsets to hop between"
                )));
            }
            for &offset in &h.offsets_hz {
                check_offset(format!("hopper {i}"), offset)?;
            }
            check_level(format!("hopper {i}"), h.level_dbfs)?;
            if !h.dwell_s.is_finite() || h.dwell_s <= 0.0 {
                return Err(SourceError::InvalidConfig(format!(
                    "hopper {i} dwell must be finite and positive, got {} s",
                    h.dwell_s
                )));
            }
            if !h.period_s.is_finite() || h.period_s < h.dwell_s {
                return Err(SourceError::InvalidConfig(format!(
                    "hopper {i} period ({} s) must be finite and at least the dwell ({} s)",
                    h.period_s, h.dwell_s
                )));
            }
            let dwell = to_samples(h.dwell_s, rate);
            let period = to_samples(h.period_s, rate);
            if dwell < 1 {
                return Err(SourceError::InvalidConfig(format!(
                    "hopper {i} dwell {} s is shorter than one sample at {rate} Hz",
                    h.dwell_s
                )));
            }
            hoppers.push(HopperState {
                magnitude: magnitude(h.level_dbfs),
                phase: 0.0,
                incs: h
                    .offsets_hz
                    .iter()
                    .map(|&offset| phase_inc(offset, rate))
                    .collect(),
                dwell,
                period,
                pos: 0,
                slot: 0,
                order: h.order,
                hash_seed: stream_seed(config.seed, 1 + i as u64),
                current_inc: 0.0,
            });
        }

        let mut ook = Vec::with_capacity(config.ook.len());
        for (i, o) in config.ook.iter().enumerate() {
            check_offset(format!("ook {i}"), o.offset_hz)?;
            check_level(format!("ook {i}"), o.level_dbfs)?;
            let samples_per_symbol =
                to_symbol_samples(&format!("ook {i}"), o.symbol_rate_bd, rate)?;
            ook.push(OokState {
                magnitude: magnitude(o.level_dbfs),
                phase: 0.0,
                inc: phase_inc(o.offset_hz, rate),
                samples_per_symbol,
                pos: 0,
                slot: 0,
                hash_seed: stream_seed(config.seed, OOK_STREAM_BASE + i as u64),
                current_level: 0.0,
            });
        }

        let mut fsk = Vec::with_capacity(config.fsk.len());
        for (i, f) in config.fsk.iter().enumerate() {
            if f.levels < 2 {
                return Err(SourceError::InvalidConfig(format!(
                    "fsk {i} needs at least 2 levels, got {}",
                    f.levels
                )));
            }
            if !f.deviation_hz.is_finite() || f.deviation_hz <= 0.0 {
                return Err(SourceError::InvalidConfig(format!(
                    "fsk {i} deviation must be finite and positive, got {} Hz",
                    f.deviation_hz
                )));
            }
            check_level(format!("fsk {i}"), f.level_dbfs)?;
            let samples_per_symbol =
                to_symbol_samples(&format!("fsk {i}"), f.symbol_rate_bd, rate)?;
            let mid = (f.levels - 1) as f64 / 2.0;
            let mut incs = Vec::with_capacity(f.levels as usize);
            for level in 0..f.levels {
                let tone_offset = f.offset_hz + f.deviation_hz * (level as f64 - mid);
                check_offset(format!("fsk {i} level {level}"), tone_offset)?;
                incs.push(phase_inc(tone_offset, rate));
            }
            fsk.push(FskState {
                magnitude: magnitude(f.level_dbfs),
                phase: 0.0,
                incs,
                samples_per_symbol,
                levels: f.levels,
                pos: 0,
                slot: 0,
                hash_seed: stream_seed(config.seed, FSK_STREAM_BASE + i as u64),
                current_inc: 0.0,
            });
        }

        let noise = match &config.noise {
            Some(n) => {
                check_level("noise".to_string(), n.level_dbfs)?;
                Some(NoiseState {
                    sigma: (10f64.powf(n.level_dbfs as f64 / 10.0) / 2.0).sqrt(),
                    rng: SplitMix64::new(stream_seed(config.seed, 0)),
                })
            }
            None => None,
        };

        Ok(SigGen {
            config,
            tones,
            chirps,
            hoppers,
            ook,
            fsk,
            noise,
        })
    }

    /// The validated scene this generator was built from.
    pub fn config(&self) -> &SigGenConfig {
        &self.config
    }

    /// Synthesise the next `out.len()` samples of the stream into `out`.
    ///
    /// Deterministic and chunking-invariant: the concatenation of successive
    /// `fill` outputs depends only on the configuration, never on the buffer
    /// sizes used to read it.
    pub fn fill(&mut self, out: &mut [Complex<f32>]) {
        for s in out.iter_mut() {
            *s = Complex::new(0.0, 0.0);
        }

        for t in &mut self.tones {
            for s in out.iter_mut() {
                *s += polar(t.magnitude, t.phase);
                t.phase = wrap_phase(t.phase + t.inc);
            }
        }

        for c in &mut self.chirps {
            for s in out.iter_mut() {
                *s += polar(c.magnitude, c.phase);
                let inc = c.inc0 + c.dinc * c.pos as f64;
                c.phase = wrap_phase(c.phase + inc);
                c.pos += 1;
                if c.pos == c.sweep_len {
                    c.pos = 0;
                }
            }
        }

        for h in &mut self.hoppers {
            for s in out.iter_mut() {
                if h.pos == 0 {
                    let n = h.incs.len() as u64;
                    let idx = match h.order {
                        HopOrder::Cycle => h.slot % n,
                        HopOrder::Pseudorandom => mix(h.hash_seed ^ h.slot) % n,
                    };
                    h.current_inc = h.incs[idx as usize];
                }
                if h.pos < h.dwell {
                    *s += polar(h.magnitude, h.phase);
                    h.phase = wrap_phase(h.phase + h.current_inc);
                }
                h.pos += 1;
                if h.pos == h.period {
                    h.pos = 0;
                    h.slot += 1;
                }
            }
        }

        for o in &mut self.ook {
            for s in out.iter_mut() {
                if o.pos == 0 {
                    let bit = mix(o.hash_seed ^ o.slot) & 1;
                    o.current_level = bit as f64;
                }
                *s += polar(o.magnitude * o.current_level, o.phase);
                o.phase = wrap_phase(o.phase + o.inc);
                o.pos += 1;
                if o.pos == o.samples_per_symbol {
                    o.pos = 0;
                    o.slot += 1;
                }
            }
        }

        for f in &mut self.fsk {
            for s in out.iter_mut() {
                if f.pos == 0 {
                    let idx = mix(f.hash_seed ^ f.slot) % f.levels as u64;
                    f.current_inc = f.incs[idx as usize];
                }
                *s += polar(f.magnitude, f.phase);
                f.phase = wrap_phase(f.phase + f.current_inc);
                f.pos += 1;
                if f.pos == f.samples_per_symbol {
                    f.pos = 0;
                    f.slot += 1;
                }
            }
        }

        if let Some(n) = &mut self.noise {
            for s in out.iter_mut() {
                let (i, q) = n.rng.next_gaussian_pair();
                *s += Complex::new((n.sigma * i) as f32, (n.sigma * q) as f32);
            }
        }
    }
}

impl SampleSource for SigGen {
    fn open(desc: &SourceDesc) -> Result<Self, SourceError> {
        match desc {
            SourceDesc::SigGen(config) => SigGen::new(config.clone()),
            #[allow(unreachable_patterns)]
            _ => Err(SourceError::WrongBackend { backend: "siggen" }),
        }
    }

    fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
        sink.meta_changed(&self.meta());
        let mut chunk = vec![Complex::new(0.0, 0.0); STREAM_CHUNK];
        loop {
            self.fill(&mut chunk);
            if sink.push(&chunk) == SinkFlow::Stop {
                return Ok(());
            }
        }
    }

    fn caps(&self) -> ControlCaps {
        ControlCaps::NONE
    }

    fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
        Err(SourceError::UnsupportedControl {
            backend: "siggen",
            control: ctl.name(),
        })
    }

    fn meta(&self) -> SourceMeta {
        SourceMeta {
            sample_rate_hz: Some(self.config.sample_rate_hz),
            center_freq_hz: self.config.center_freq_hz,
            label: "siggen".to_string(),
            provenance: "synthetic cf32 (signal generator)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_sample_rates() {
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                SigGen::new(SigGenConfig::new(rate)),
                Err(SourceError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn rejects_offsets_beyond_nyquist() {
        let mut config = SigGenConfig::new(1_000_000.0);
        config.tones.push(ToneConfig {
            offset_hz: 500_001.0,
            level_dbfs: 0.0,
        });
        let err = SigGen::new(config).unwrap_err();
        let SourceError::InvalidConfig(msg) = err else {
            panic!("wrong error kind: {err:?}");
        };
        assert!(msg.contains("tone 0"), "unhelpful message: {msg}");
        assert!(msg.contains("Nyquist"), "unhelpful message: {msg}");
    }

    #[test]
    fn rejects_hopper_with_period_shorter_than_dwell() {
        let mut config = SigGenConfig::new(1_000_000.0);
        config.hoppers.push(HopperConfig {
            offsets_hz: vec![100_000.0],
            dwell_s: 0.010,
            period_s: 0.005,
            level_dbfs: -10.0,
            order: HopOrder::Cycle,
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_empty_hopper_and_bad_chirp_time() {
        let mut config = SigGenConfig::new(1_000_000.0);
        config.hoppers.push(HopperConfig {
            offsets_hz: vec![],
            dwell_s: 0.001,
            period_s: 0.002,
            level_dbfs: -10.0,
            order: HopOrder::Cycle,
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));

        let mut config = SigGenConfig::new(1_000_000.0);
        config.chirps.push(ChirpConfig {
            start_offset_hz: 0.0,
            stop_offset_hz: 100_000.0,
            sweep_time_s: 0.0,
            level_dbfs: -10.0,
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));
    }

    #[test]
    fn demo_scene_is_valid() {
        SigGen::new(SigGenConfig::demo()).expect("demo scene must always construct");
    }

    #[test]
    fn siggen_advertises_no_controls_and_rejects_set() {
        let mut gen = SigGen::new(SigGenConfig::new(1_000_000.0)).unwrap();
        assert_eq!(gen.caps(), ControlCaps::NONE);
        assert!(matches!(
            gen.set(Control::CenterFreqHz(1e6)),
            Err(SourceError::UnsupportedControl {
                backend: "siggen",
                ..
            })
        ));
    }

    #[test]
    fn rejects_ook_symbol_rate_too_fast_or_non_positive() {
        for bad in [0.0, -1.0, f64::NAN, 3_000_000.0] {
            let mut config = SigGenConfig::new(1_000_000.0);
            config.ook.push(OokConfig {
                offset_hz: 0.0,
                symbol_rate_bd: bad,
                level_dbfs: 0.0,
            });
            assert!(
                matches!(SigGen::new(config), Err(SourceError::InvalidConfig(_))),
                "OOK symbol rate {bad} Bd must be rejected"
            );
        }
    }

    #[test]
    fn rejects_fsk_bad_levels_deviation_and_rate() {
        let base = FskConfig {
            offset_hz: 0.0,
            levels: 4,
            deviation_hz: 10_000.0,
            symbol_rate_bd: 1_000.0,
            level_dbfs: 0.0,
        };

        let mut config = SigGenConfig::new(1_000_000.0);
        config.fsk.push(FskConfig {
            levels: 1,
            ..base.clone()
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));

        let mut config = SigGenConfig::new(1_000_000.0);
        config.fsk.push(FskConfig {
            deviation_hz: 0.0,
            ..base.clone()
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));

        let mut config = SigGenConfig::new(1_000_000.0);
        config.fsk.push(FskConfig {
            symbol_rate_bd: -1.0,
            ..base.clone()
        });
        assert!(matches!(
            SigGen::new(config),
            Err(SourceError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_fsk_ladder_beyond_nyquist() {
        // 8 levels * 200 kHz spacing spans ±700 kHz around 0 — the top
        // level's tone lands beyond Nyquist at 500 kHz (1 MHz sample rate).
        let mut config = SigGenConfig::new(1_000_000.0);
        config.fsk.push(FskConfig {
            offset_hz: 0.0,
            levels: 8,
            deviation_hz: 200_000.0,
            symbol_rate_bd: 1_000.0,
            level_dbfs: 0.0,
        });
        let err = SigGen::new(config).unwrap_err();
        let SourceError::InvalidConfig(msg) = err else {
            panic!("wrong error kind: {err:?}");
        };
        assert!(msg.contains("fsk 0 level"), "unhelpful message: {msg}");
    }

    /// Measured instantaneous frequency (Hz) between two consecutive
    /// non-zero-magnitude complex samples, from their phase difference.
    fn measured_freq_hz(a: Complex<f32>, b: Complex<f32>, rate: f64) -> f64 {
        let delta = (b * a.conj()).arg() as f64;
        delta * rate / std::f64::consts::TAU
    }

    #[test]
    fn ook_keys_amplitude_on_symbol_boundaries_at_the_configured_frequency() {
        const RATE: f64 = 1_000_000.0;
        const OFFSET_HZ: f64 = 50_000.0;
        const SYMBOL_RATE_BD: f64 = 10_000.0; // 100 samples/symbol
        const SPS: usize = 100;

        let mut config = SigGenConfig::new(RATE);
        config.seed = 123;
        config.ook.push(OokConfig {
            offset_hz: OFFSET_HZ,
            symbol_rate_bd: SYMBOL_RATE_BD,
            level_dbfs: 0.0,
        });
        let mut gen = SigGen::new(config).unwrap();
        let n_symbols = 50;
        let mut out = vec![Complex::new(0.0f32, 0.0); n_symbols * SPS];
        gen.fill(&mut out);

        let mut saw_on = false;
        let mut saw_off = false;
        for (sym, block) in out.chunks(SPS).enumerate() {
            let on = block[0].norm() > 0.5;
            for &s in block {
                if on {
                    assert!(
                        (s.norm() - 1.0).abs() < 1e-5,
                        "symbol {sym}: expected full scale"
                    );
                } else {
                    assert_eq!(s.norm(), 0.0, "symbol {sym}: expected exact silence");
                }
            }
            if on {
                saw_on = true;
            } else {
                saw_off = true;
            }
        }
        assert!(saw_on && saw_off, "the bit stream must key both states");

        // Frequency check across every "on" run in the trace.
        let mut checked = 0;
        for w in out.windows(2) {
            if w[0].norm() > 0.5 && w[1].norm() > 0.5 {
                let f = measured_freq_hz(w[0], w[1], RATE);
                assert!(
                    (f - OFFSET_HZ).abs() < 1.0,
                    "measured {f} Hz, expected {OFFSET_HZ} Hz"
                );
                checked += 1;
            }
        }
        assert!(checked > 1000, "too few on-samples measured: {checked}");
    }

    #[test]
    fn fsk_is_constant_envelope_phase_continuous_and_on_the_tone_ladder() {
        const RATE: f64 = 1_000_000.0;
        const LEVELS: u32 = 4;
        const DEVIATION_HZ: f64 = 20_000.0;
        const SYMBOL_RATE_BD: f64 = 10_000.0; // 100 samples/symbol
        const SPS: usize = 100;
        let expected_tones: Vec<f64> = (0..LEVELS)
            .map(|l| DEVIATION_HZ * (l as f64 - (LEVELS - 1) as f64 / 2.0))
            .collect();

        let mut config = SigGenConfig::new(RATE);
        config.seed = 999;
        config.fsk.push(FskConfig {
            offset_hz: 0.0,
            levels: LEVELS,
            deviation_hz: DEVIATION_HZ,
            symbol_rate_bd: SYMBOL_RATE_BD,
            level_dbfs: 0.0,
        });
        let mut gen = SigGen::new(config).unwrap();
        let n_symbols = 200;
        let mut out = vec![Complex::new(0.0f32, 0.0); n_symbols * SPS];
        gen.fill(&mut out);

        // Constant envelope throughout (no OOK-style keying in FSK).
        for s in &out {
            assert!((s.norm() - 1.0).abs() < 1e-5);
        }

        // Every consecutive-sample frequency estimate — transitions included
        // — lands exactly on one of the M configured tones (phase-continuous
        // CPFSK has no glitch at a symbol boundary).
        let mut seen = vec![false; LEVELS as usize];
        for w in out.windows(2) {
            let f = measured_freq_hz(w[0], w[1], RATE);
            let (idx, closest) = expected_tones
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (**a - f).abs().total_cmp(&(**b - f).abs()))
                .unwrap();
            assert!(
                (closest - f).abs() < 1.0,
                "measured {f} Hz off the ladder {expected_tones:?}"
            );
            seen[idx] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "every configured tone must appear: {seen:?}"
        );
    }

    #[test]
    fn ook_and_fsk_streams_are_chunk_invariant() {
        let mut config = SigGenConfig::new(1_000_000.0);
        config.seed = 42;
        config.ook.push(OokConfig {
            offset_hz: 100_000.0,
            symbol_rate_bd: 5_000.0,
            level_dbfs: -6.0,
        });
        config.fsk.push(FskConfig {
            offset_hz: -100_000.0,
            levels: 4,
            deviation_hz: 15_000.0,
            symbol_rate_bd: 5_000.0,
            level_dbfs: -3.0,
        });

        let total = 10_000;
        let mut whole = vec![Complex::new(0.0f32, 0.0); total];
        SigGen::new(config.clone()).unwrap().fill(&mut whole);

        let mut gen = SigGen::new(config).unwrap();
        let mut chunked = Vec::with_capacity(total);
        for size in [1usize, 3, 7, 250, 4096].into_iter().cycle() {
            if chunked.len() >= total {
                break;
            }
            let n = size.min(total - chunked.len());
            let mut buf = vec![Complex::new(0.0f32, 0.0); n];
            gen.fill(&mut buf);
            chunked.extend(buf);
        }

        assert_eq!(whole, chunked);
    }
}
