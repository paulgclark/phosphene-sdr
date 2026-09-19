// SPDX-License-Identifier: MIT

//! Track lifecycle and hopper clustering (nextgen spec §7.4, FR-AD4/FR-AD5).
//!
//! This module owns [`Track`] — the single data model that measurement,
//! classification, signature fusion, and annotation all read (spec §8.3,
//! single-writer per stage) — and the two stages that produce and organise
//! tracks:
//!
//! * [`tracker`] — the FR-AD4 [`Tracker`]: gated nearest-neighbour
//!   association of per-frame detections, alpha-beta smoothing of centre and
//!   bandwidth, M-of-N birth and miss-timeout death, stable per-session ids.
//! * [`hopper`] — FR-AD5 [`cluster_hoppers`]: folds many short
//!   co-parametric bursts into one logical [`Emitter`] (FHSS/TDMA),
//!   recovering the hop set and rate — hopping is a reportable behaviour,
//!   not annotation spam.
//!
//! [`Classification`] and [`SignatureCandidate`] are defined here because
//! they are fields of `Track` and thus part of the frozen exchange contract;
//! the *algorithms* that produce them belong to later phases (A1–A3, spec §9)
//! in their own `classify`/`signatures` modules.

pub mod hopper;
pub mod tracker;

pub use hopper::{cluster_hoppers, Emitter, HopperClusterConfig};
pub use tracker::{Tracker, TrackerConfig, TrackerError};

use serde::{Deserialize, Serialize};

use crate::measure::{Measured, MeasurementSet};

/// Stable per-session identity of a track (FR-AD4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrackId(pub u64);

/// Lifecycle state of a track (FR-AD4: M-of-N birth, miss-timeout death).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackState {
    /// Seen, but not yet confirmed by M-of-N association.
    Tentative,
    /// Confirmed live track.
    Confirmed,
    /// Recently missed; held pending the death timeout.
    Coasting,
    /// Timed out; retained only for final reporting.
    Dead,
}

/// A modulation classification with its evidence (FR-AC3: always with the
/// evidence that drove the decision; FR-AC6/NFR-A3: confidence-gated, never
/// asserted beyond its supporting evidence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    /// The decided class.
    pub class: ModulationClass,
    /// Symbol rate in baud, when a symbol-rate line was found (FR-AC2;
    /// applies to OOK/ASK, FSK and "linear" alike).
    pub symbol_rate_bd: Option<Measured<f64>>,
    /// Confidence in the class decision, `0.0..=1.0`.
    pub confidence: f32,
    /// Human-readable evidence: which §7.9 tests fired and what they saw
    /// (e.g. "envelope variance high; two instantaneous-frequency modes").
    pub evidence: Vec<String>,
}

/// The modulation classes of the §7.9 decision tree (FR-AC3), with the
/// per-class blind parameters of FR-AC3–FR-AC5.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModulationClass {
    /// On-off / amplitude keying.
    OokAsk,
    /// Frequency-shift keying.
    Fsk {
        /// Number of levels M (mode count of the IF histogram).
        levels: u32,
        /// Frequency deviation (mode spacing), Hz.
        deviation_hz: Measured<f64>,
    },
    /// OFDM, with the blind parameters of §7.11 (FR-AC4). Subcarrier spacing
    /// follows as `Δf = 1 / useful_symbol_s`.
    Ofdm {
        /// Useful symbol time T_u, seconds.
        useful_symbol_s: Measured<f64>,
        /// Cyclic-prefix fraction `N_cp / N_u`.
        cp_fraction: Measured<f64>,
    },
    /// Chirp / chirp-spread-spectrum (FR-AC5).
    ChirpCss {
        /// Chirp rate, Hz per second.
        chirp_rate_hz_per_s: Measured<f64>,
    },
    /// Linear modulation (PSK/QAM-like) with undetermined order — the tree
    /// stops here rather than guessing (§7.10).
    Linear,
    /// No class met its evidence threshold; measurements still stand.
    Unknown,
}

/// One ranked signature-fusion candidate (FR-AS2/FR-AS3: "consistent with",
/// never "is").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignatureCandidate {
    /// Human label of the matched signature (e.g. "POCSAG paging").
    pub label: String,
    /// Match score, `0.0..=1.0`.
    pub score: f32,
    /// The DB entry's mandatory primary-source citation (LC-3).
    pub source: String,
}

/// A persistent detected signal — the central data model of the analysis
/// pipeline (spec §8.3). One writer per stage: the tracker owns identity,
/// state and the smoothed kinematics; the measure stage fills
/// [`measurements`](Self::measurements); classification fills
/// [`class`](Self::class) (A1+); signature fusion fills
/// [`signatures`](Self::signatures) (A3). Serializable for golden tests and
/// the copy-as-text / headless-report features (FR-AU4, spec §8.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Stable per-session id (FR-AD4).
    pub id: TrackId,
    /// Lifecycle state.
    pub state: TrackState,
    /// Smoothed center as an offset from band center, Hz — the tracker's
    /// kinematic state used for gating/association (§7.4). The *reported*
    /// center lives in [`measurements`](Self::measurements) (FR-AM1).
    pub center_offset_hz: f64,
    /// Smoothed occupied bandwidth, Hz (kinematic state, as above).
    pub bandwidth_hz: f64,
    /// Smoothed in-band power, dBFS (kinematic state).
    pub power_dbfs: f32,
    /// Latest SNR versus the noise floor, dB.
    pub snr_db: f32,
    /// Frame sequence number the track was born at.
    pub born_frame: u64,
    /// Frame sequence number of the most recent associated detection.
    pub last_seen_frame: u64,
    /// The FR-AM measurement bundle; `None` until the measure stage has run.
    pub measurements: Option<MeasurementSet>,
    /// Modulation classification; `None` until phase A1+ classifies (or the
    /// evidence never suffices — NFR-A3).
    pub class: Option<Classification>,
    /// Ranked signature-fusion candidates, best first; empty until phase A3.
    pub signatures: Vec<SignatureCandidate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_ids_are_ordered_and_hashable() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        assert!(seen.insert(TrackId(1)));
        assert!(!seen.insert(TrackId(1)));
        assert!(TrackId(1) < TrackId(2));
    }
}

/// The lane seal, end to end on real M0-generator IQ (D-017's known ground
/// truth): siggen scene → offline harness frames → a minimal test-local
/// threshold detector → [`Tracker`] → [`cluster_hoppers`]. The steady tone
/// must come out as exactly one stable track, and the known hopper as ONE
/// emitter with its configured hop set and rate recovered. The pipeline must
/// be bit-deterministic for a fixed seed (NFR-A4).
///
/// The threshold detector below is deliberately naive scaffolding — the real
/// detector (noise floor, CFAR, blobs) is the A0-1 lane's, in `detect/`.
#[cfg(test)]
mod pipeline_tests {
    use phosphene_core::Complex;
    use phosphene_sources::{
        HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig,
    };

    use crate::detect::Detection;
    use crate::harness::{OfflineTap, OfflineTapConfig};
    use crate::tap::{FrameView, SpectrumTap};
    use crate::track::{
        cluster_hoppers, Emitter, HopperClusterConfig, Track, Tracker, TrackerConfig,
    };

    const RATE: f64 = 1_024_000.0;
    const FFT: usize = 1024;
    const FRAMES: usize = 410; // 410 ms → 28 half-aligned 15 ms hop periods
    const HOP_OFFSETS_HZ: [f64; 4] = [-375_000.0, -125_000.0, 125_000.0, 375_000.0];
    const HOP_DWELL_S: f64 = 0.005; // 5 frames at 1 ms/frame
    const HOP_PERIOD_S: f64 = 0.015; // 15 frames
    const TONE_OFFSET_HZ: f64 = 450_000.0;
    const THRESHOLD_DBFS: f32 = -45.0;

    /// Contiguous bins above a fixed threshold → one per-frame Detection.
    fn naive_detect(frame: &[f32], seq: u64) -> Vec<Detection> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < frame.len() {
            if frame[i] < THRESHOLD_DBFS {
                i += 1;
                continue;
            }
            let lo = i;
            while i < frame.len() && frame[i] >= THRESHOLD_DBFS {
                i += 1;
            }
            let hi = i - 1;
            let mut power_sum = 0.0f64;
            let mut weighted = 0.0f64;
            let mut peak = f32::NEG_INFINITY;
            for (b, &db) in frame.iter().enumerate().take(hi + 1).skip(lo) {
                let p = 10f64.powf(f64::from(db) / 10.0);
                power_sum += p;
                weighted += p * b as f64;
                peak = peak.max(db);
            }
            out.push(Detection {
                bin_lo: lo,
                bin_hi: hi,
                frame_lo: seq,
                frame_hi: seq,
                centroid_bin: weighted / power_sum,
                peak_dbfs: peak,
                power_dbfs: (10.0 * power_sum.log10()) as f32,
                // vs the scene's ≈ −108 dBFS/bin noise floor; exactness is
                // irrelevant to tracking.
                snr_db: peak + 108.0,
            });
        }
        out
    }

    /// Run the whole pipeline once; returns (live, retired, emitters).
    fn run_pipeline() -> (Vec<Track>, Vec<Track>, Vec<Emitter>) {
        let mut scene = SigGenConfig::new(RATE);
        scene.seed = 7;
        scene.noise = Some(NoiseConfig { level_dbfs: -80.0 });
        scene.tones.push(ToneConfig {
            offset_hz: TONE_OFFSET_HZ,
            level_dbfs: -20.0,
        });
        scene.hoppers.push(HopperConfig {
            offsets_hz: HOP_OFFSETS_HZ.to_vec(),
            dwell_s: HOP_DWELL_S,
            period_s: HOP_PERIOD_S,
            level_dbfs: -25.0,
            order: HopOrder::Cycle,
        });
        let mut gen = SigGen::new(scene).unwrap();

        let mut tap_cfg = OfflineTapConfig::new(RATE);
        tap_cfg.fft_size = FFT;
        tap_cfg.spectrum_history_frames = 2;
        let mut tap = OfflineTap::new(tap_cfg).unwrap();
        let meta = tap.meta();

        // D-125 item 3: `TrackerConfig::default()`'s birth/coast durations
        // are calibrated at the M0 generator's own 2.048 MS/s / FFT 512
        // (250us/frame) — at this test's coarser 1ms/frame, that duration is
        // under two frames, too short for `birth_m`'s 3 hits ever to land.
        // This scene's own comments ("5 frames at 1 ms/frame") show what it
        // actually wants: the *original* frame-counted 3-of-5 birth and
        // 2-frame coast, restated in this scene's own frame time, not the
        // M0 generator's.
        let frame_dt_s = FFT as f64 / RATE;
        let tracker_cfg = TrackerConfig {
            birth_window_s: 5.0 * frame_dt_s,
            max_coast_s: 2.0 * frame_dt_s,
            ..TrackerConfig::default()
        };
        let mut tracker = Tracker::new(tracker_cfg).unwrap();
        let mut view = FrameView::new(FFT, 1);
        let mut chunk = vec![Complex::new(0.0f32, 0.0); FFT];
        for _ in 0..FRAMES {
            gen.fill(&mut chunk);
            tap.feed(&chunk);
            tap.latest_frames(&mut view);
            let seq = view.seq(0);
            let dets = naive_detect(view.frame(0), seq);
            tracker.observe(seq, &dets, &meta);
        }

        let live: Vec<Track> = tracker.tracks().cloned().collect();
        let retired = tracker.retired().to_vec();
        let emitters = cluster_hoppers(
            retired.iter().chain(live.iter()),
            &meta,
            &HopperClusterConfig::default(),
        );
        (live, retired, emitters)
    }

    #[test]
    fn m0_scene_yields_one_tone_track_and_one_hopper_emitter() {
        let (live, retired, emitters) = run_pipeline();

        // The steady tone: exactly one long-lived track, alive since the
        // first frame, still confirmed at the end — one id for its life.
        let long_lived: Vec<&Track> = live
            .iter()
            .filter(|t| t.last_seen_frame - t.born_frame > 100)
            .collect();
        assert_eq!(long_lived.len(), 1, "the tone is exactly one track");
        let tone = long_lived[0];
        assert_eq!(tone.born_frame, 0);
        assert_eq!(tone.last_seen_frame, FRAMES as u64 - 1);
        assert!(
            (tone.center_offset_hz - TONE_OFFSET_HZ).abs() < 500.0,
            "tone at {} Hz",
            tone.center_offset_hz
        );
        // No retired track is tone-like: nothing near it in frequency, and
        // every retired track is a short burst (no id churn on the tone).
        for t in &retired {
            assert!((t.center_offset_hz - TONE_OFFSET_HZ).abs() > 50_000.0);
            assert!(t.last_seen_frame - t.born_frame < 10);
        }

        // The known hopper: ONE emitter, hop set and rate recovered.
        assert_eq!(emitters.len(), 1, "one hopper, one emitter: {emitters:#?}");
        let e = &emitters[0];
        assert!(e.hopping);
        assert_eq!(e.hop_set_hz.len(), 4, "hop set: {:?}", e.hop_set_hz);
        for (got, want) in e.hop_set_hz.iter().zip(HOP_OFFSETS_HZ) {
            assert!((got - want).abs() < 1_500.0, "hop {got} Hz != {want} Hz");
        }
        let rate = e.hop_rate_hz.expect("regular hopper has a rate");
        assert!(
            (rate.value - 1.0 / HOP_PERIOD_S).abs() < 1.0,
            "hop rate {} Hz",
            rate.value
        );
        assert!((e.mean_dwell_s.value - HOP_DWELL_S).abs() < 0.002);
        let spacing = e
            .carrier_spacing_hz
            .expect("hop set sits on a 250 kHz grid");
        assert!((spacing - 250_000.0).abs() < 3_000.0);

        // Every burst the tracker confirmed belongs to the emitter — the
        // overlay shows one emitter, not dozens of ephemeral tracks.
        assert_eq!(e.member_ids.len(), retired.len() + live.len() - 1);
        assert!(!e.member_ids.contains(&tone.id));
        assert!(
            e.member_ids.len() >= 20,
            "expected the scene's ~28 bursts, got {}",
            e.member_ids.len()
        );
    }

    #[test]
    fn pipeline_is_deterministic_for_a_fixed_seed() {
        let a = run_pipeline();
        let b = run_pipeline();
        let json = |r: &(Vec<Track>, Vec<Track>, Vec<Emitter>)| {
            serde_json::to_string(&(&r.0, &r.1, &r.2)).unwrap()
        };
        assert_eq!(json(&a), json(&b));
    }

    /// Ids must be unique across the whole session — live and dead alike
    /// (FR-AD4: stable id for the session).
    #[test]
    fn session_ids_never_repeat() {
        let (live, retired, _) = run_pipeline();
        let mut seen = std::collections::HashSet::new();
        for t in retired.iter().chain(live.iter()) {
            assert!(seen.insert(t.id), "duplicate id {:?}", t.id);
        }
    }
}
