// SPDX-License-Identifier: MIT

//! The FR-AD4 tracker: gated nearest-neighbour association of per-frame
//! detections into persistent [`Track`]s, alpha-beta smoothing of centre and
//! bandwidth, M-of-N birth and miss-timeout death (nextgen spec §7.4).
//!
//! ## Contract
//!
//! Call [`Tracker::observe`] once per spectrum frame, in strictly increasing
//! frame-sequence order, with the [`Detection`]s active in that frame (an
//! empty slice when the frame is clear — misses drive coasting and death, so
//! quiet frames must still be observed). Detections normally cover just the
//! observed frame (`frame_lo == frame_hi`); a detector that delivers a
//! completed multi-frame blob in one piece is also supported — its frame
//! span is credited toward M-of-N birth, so a blob that already persisted
//! `M` frames confirms immediately.
//!
//! ## Association (spec §7.4)
//!
//! Each live track predicts its centre one step ahead with its alpha-beta
//! velocity, then candidate (track, detection) pairs inside the
//! centre/bandwidth gates are ranked by normalised distance and assigned
//! greedily from the globally best pair down — the standard greedy
//! nearest-neighbour assignment, which resolves contested detections in
//! cost order rather than track order. Velocity in the prediction is what
//! carries identity through a crossing: each track follows its own
//! trajectory instead of grabbing the nearest detection at the crossing
//! point.
//!
//! ## Determinism (NFR-A4)
//!
//! The tracker holds no randomness and iterates only ordered storage; ties
//! in association cost break by (track, detection) index. The same
//! detection stream always yields the same tracks and ids.

use std::fmt;

use crate::detect::Detection;
use crate::tap::BandMeta;
use crate::track::{Track, TrackId, TrackState};

/// Tuning for a [`Tracker`]. All constants are spec §7 *tune* values,
/// calibrated against the M0 generator (D-017's known-ground-truth source).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackerConfig {
    /// M of the M-of-N birth rule: detections required inside the last
    /// [`birth_window_s`](Self::birth_window_s) before a tentative track
    /// confirms. A count, not a duration — unlike the window it counts hits
    /// against, "how many hits" has no frame-rate dependence to correct for
    /// (D-125 item 3).
    pub birth_m: u32,
    /// N of the M-of-N birth rule, **seconds** (D-125 item 3, not frames): a
    /// tentative track that reaches this age without M hits is dropped. At
    /// [`Tracker::observe`]'s first call this is converted to an equivalent
    /// frame count against that call's `meta.frame_dt_s` (assumed fixed for
    /// the tracker's life, per this type's own contract) and cached; the
    /// derived frame count is clamped to 64 (the hit-history bitmask's
    /// width) — see [`Tracker::effective_allowances`].
    pub birth_window_s: f64,
    /// Consecutive missed time a confirmed track survives (coasting),
    /// **seconds** (D-125 item 3, not frames), before it is retired as
    /// [`TrackState::Dead`]. Converted to an equivalent frame count the same
    /// way as [`birth_window_s`](Self::birth_window_s).
    pub max_coast_s: f64,
    /// Alpha-beta position gain, `0.0 < alpha <= 1.0` — fraction of the
    /// residual applied to the smoothed centre/bandwidth each hit.
    pub alpha: f64,
    /// Alpha-beta velocity gain, `>= 0` — fraction of the residual applied
    /// to the per-frame velocity each hit.
    pub beta: f64,
    /// Exponential smoothing gain for in-band power, `0.0 < g <= 1.0`.
    pub power_alpha: f32,
    /// Centre-frequency gate floor in bins: a detection associates only
    /// within `max(gate_min_bins × bin_hz, gate_bw_factor × bandwidth)` of
    /// the predicted centre.
    pub gate_min_bins: f64,
    /// Bandwidth-proportional part of the centre gate (see
    /// [`gate_min_bins`](Self::gate_min_bins)).
    pub gate_bw_factor: f64,
    /// Bandwidth gate: the larger of (detection, predicted) bandwidth may
    /// exceed the smaller by at most this ratio (`> 1`).
    pub bw_gate_ratio: f64,
}

impl Default for TrackerConfig {
    /// `birth_m: 1` (AN-1/AN-3, D-126): a track confirms on its very first
    /// detection, rather than waiting to reappear across several frames.
    /// This replaces the original M0-calibrated 3-of-5 birth — measured
    /// (D-126) to never confirm a burst shorter than about three analysis
    /// frames, no matter the frame rate, since M-of-N counts *hits*, not
    /// duration — with a structural gate applied further upstream instead:
    /// [`crate::detect::DetectorConfig::min_blob_cells`] and
    /// [`crate::detect::CfarConfig::pfa`] (also retuned by this lane) decide
    /// which single detections are trustworthy enough to publish
    /// immediately, so a short, strong burst is no longer asked to persist
    /// long enough to out-run its own duration. The M-of-N mechanism itself
    /// is unchanged and still available at any `birth_m > 1` for a caller
    /// that wants time-persistence gating instead of (or alongside) a
    /// stricter blob-size/Pfa gate.
    ///
    /// [`birth_window_s`](Self::birth_window_s) and
    /// [`max_coast_s`](Self::max_coast_s) keep their original M0-calibrated
    /// values (D-125 item 3: restated in seconds, not frames, so the same
    /// real-world timing holds at any frame rate) — the 2.048 MS/s /
    /// 512-point FFT generator's own 5-frame birth window and 2-frame coast
    /// (`frame_dt_s` = 250 µs) convert to 1.25 ms and 500 µs. The birth
    /// window is now moot at the default `birth_m: 1` (one hit always
    /// confirms inside any positive window) but stays meaningful the moment
    /// a caller raises `birth_m`. A snappy alpha-beta (0.5 / 0.15), and
    /// gates of 3 bins or 3/4 of the track bandwidth.
    ///
    /// Measured together on the real 913 MHz capture (D-126's coupled seal,
    /// `crates/phosphene-app/src/analyze.rs`'s
    /// `an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture`, its
    /// ground truth associated to tracks in time **and** frequency —
    /// Amendment 2): recall rose from 51/52 to 52/52 ground-truth bursts,
    /// **SHORT-burst** recall specifically (AN-1's own claim, bursts of 2
    /// frames or fewer — the ones the old M-of-N rule could never confirm)
    /// rose from 10/11 to 11/11, **and** the false-alarm rate fell from
    /// 59.1/s to 35.6/s against the pre-fix defaults — all three numbers
    /// moving the right way together, not traded.
    fn default() -> Self {
        TrackerConfig {
            birth_m: 1,
            birth_window_s: 5.0 * 250.0e-6,
            max_coast_s: 2.0 * 250.0e-6,
            alpha: 0.5,
            beta: 0.15,
            power_alpha: 0.4,
            gate_min_bins: 3.0,
            gate_bw_factor: 0.75,
            bw_gate_ratio: 4.0,
        }
    }
}

/// Errors from building a [`Tracker`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TrackerError {
    /// The configuration was rejected; the message names the field and why.
    InvalidConfig(String),
}

impl fmt::Display for TrackerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackerError::InvalidConfig(msg) => {
                write!(f, "invalid tracker configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for TrackerError {}

/// One live track plus the filter state the public [`Track`] does not carry.
#[derive(Debug, Clone)]
struct Slot {
    track: Track,
    /// Centre velocity, Hz per frame.
    vel_hz: f64,
    /// Bandwidth velocity, Hz per frame.
    bw_vel_hz: f64,
    /// Hit history bitmask, bit 0 = the most recent observed frame.
    hit_mask: u64,
    /// Consecutive missed frames since the last associated detection.
    misses: u32,
}

/// The FR-AD4 tracker. See the [module docs](self) for the contract.
#[derive(Debug, Clone)]
pub struct Tracker {
    cfg: TrackerConfig,
    /// [`TrackerConfig::birth_window_s`] and
    /// [`TrackerConfig::max_coast_s`], converted once to frame counts
    /// against the first [`Self::observe`] call's `meta.frame_dt_s` and
    /// cached (D-125 item 3) — see [`Self::effective_allowances`].
    derived: Option<(u32, u32)>,
    /// Next id to assign; ids are unique for the tracker's lifetime (FR-AD4:
    /// stable for the session).
    next_id: u64,
    /// Tentative + confirmed + coasting tracks, in birth order.
    live: Vec<Slot>,
    /// Confirmed tracks that timed out, in death order (never tentatives:
    /// M-of-N suppressed those before they became visible).
    retired: Vec<Track>,
    last_frame: Option<u64>,
}

/// A mask with the low `n` bits set (saturating at 64).
fn low_bits(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

impl Tracker {
    /// Build a tracker; validates the configuration up front.
    pub fn new(cfg: TrackerConfig) -> Result<Self, TrackerError> {
        let bad = |msg: String| Err(TrackerError::InvalidConfig(msg));
        if cfg.birth_m == 0 {
            return bad("birth_m must be at least 1".to_string());
        }
        if !(cfg.birth_window_s > 0.0 && cfg.birth_window_s.is_finite()) {
            return bad(format!(
                "birth_window_s must be finite and positive, got {}",
                cfg.birth_window_s
            ));
        }
        if !(cfg.max_coast_s >= 0.0 && cfg.max_coast_s.is_finite()) {
            return bad(format!(
                "max_coast_s must be finite and >= 0, got {}",
                cfg.max_coast_s
            ));
        }
        if !(cfg.alpha > 0.0 && cfg.alpha <= 1.0) {
            return bad(format!("alpha must be in (0, 1], got {}", cfg.alpha));
        }
        if !(cfg.beta >= 0.0 && cfg.beta.is_finite()) {
            return bad(format!("beta must be finite and >= 0, got {}", cfg.beta));
        }
        if !(cfg.power_alpha > 0.0 && cfg.power_alpha <= 1.0) {
            return bad(format!(
                "power_alpha must be in (0, 1], got {}",
                cfg.power_alpha
            ));
        }
        if !(cfg.gate_min_bins > 0.0 && cfg.gate_min_bins.is_finite()) {
            return bad(format!(
                "gate_min_bins must be finite and positive, got {}",
                cfg.gate_min_bins
            ));
        }
        if !(cfg.gate_bw_factor >= 0.0 && cfg.gate_bw_factor.is_finite()) {
            return bad(format!(
                "gate_bw_factor must be finite and >= 0, got {}",
                cfg.gate_bw_factor
            ));
        }
        if !(cfg.bw_gate_ratio > 1.0 && cfg.bw_gate_ratio.is_finite()) {
            return bad(format!(
                "bw_gate_ratio must be finite and > 1, got {}",
                cfg.bw_gate_ratio
            ));
        }
        Ok(Tracker {
            cfg,
            derived: None,
            next_id: 1,
            live: Vec::new(),
            retired: Vec::new(),
            last_frame: None,
        })
    }

    /// The configuration this tracker runs with.
    pub fn config(&self) -> &TrackerConfig {
        &self.cfg
    }

    /// `(effective_birth_n, effective_max_coast_frames)` — [`TrackerConfig`]'s
    /// second-denominated allowances, converted to frame counts against
    /// `frame_dt_s` (D-125 item 3) and cached from the first call: `meta`
    /// "must describe the same grid for the tracker's whole life"
    /// ([`Self::observe`]'s own contract), so `frame_dt_s` is assumed fixed
    /// once seen. `effective_birth_n` is clamped to `1..=64` — the
    /// hit-history bitmask's width — so a birth window configured far wider
    /// than 64 frames at a given rate degrades to "the last 64 frames",
    /// documented here rather than silently: a caller specifying seconds
    /// this large at a frame rate this high is asking for a birth window no
    /// `u64` bitmask can represent exactly.
    fn effective_allowances(&mut self, frame_dt_s: f64) -> (u32, u32) {
        *self.derived.get_or_insert_with(|| {
            let birth_n = (self.cfg.birth_window_s / frame_dt_s)
                .round()
                .clamp(1.0, 64.0) as u32;
            let max_coast = (self.cfg.max_coast_s / frame_dt_s)
                .round()
                .clamp(0.0, f64::from(u32::MAX)) as u32;
            (birth_n, max_coast)
        })
    }

    /// Observe one frame's detections. `meta` converts detection grid
    /// coordinates to Hz and must describe the same grid for the tracker's
    /// whole life.
    ///
    /// # Panics
    ///
    /// Panics if `frame` is not strictly greater than the previously
    /// observed frame — the frame-to-frame contract of spec §7.4.
    pub fn observe(&mut self, frame: u64, detections: &[Detection], meta: &BandMeta) {
        if let Some(last) = self.last_frame {
            assert!(
                frame > last,
                "observe() frames must be strictly increasing (got {frame} after {last})"
            );
        }
        let dt = self.last_frame.map_or(1, |last| frame - last);
        self.last_frame = Some(frame);
        let dtf = dt as f64;
        let bin_hz = meta.bin_hz;
        let (birth_n, max_coast_frames) = self.effective_allowances(meta.frame_dt_s);

        // Predict every live track one interval ahead (alpha-beta dead
        // reckoning: this is what carries identity through a crossing).
        let predicted: Vec<(f64, f64)> = self
            .live
            .iter()
            .map(|s| {
                (
                    s.track.center_offset_hz + s.vel_hz * dtf,
                    (s.track.bandwidth_hz + s.bw_vel_hz * dtf).max(bin_hz),
                )
            })
            .collect();

        // Gated candidate pairs, ranked by normalised distance; ties break
        // by (track, detection) index for determinism.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (si, &(pred_c, pred_bw)) in predicted.iter().enumerate() {
            let gate_hz = (self.cfg.gate_min_bins * bin_hz).max(self.cfg.gate_bw_factor * pred_bw);
            for (di, d) in detections.iter().enumerate() {
                let dc = (d.centroid_offset_hz(meta) - pred_c).abs();
                if dc > gate_hz {
                    continue;
                }
                let det_bw = (d.bin_hi - d.bin_lo + 1) as f64 * bin_hz;
                let ratio = det_bw.max(pred_bw) / det_bw.min(pred_bw).max(bin_hz);
                if ratio > self.cfg.bw_gate_ratio {
                    continue;
                }
                let cost = dc / gate_hz + 0.5 * (ratio - 1.0) / (self.cfg.bw_gate_ratio - 1.0);
                pairs.push((cost, si, di));
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut assigned: Vec<Option<usize>> = vec![None; self.live.len()];
        let mut det_used = vec![false; detections.len()];
        for &(_, si, di) in &pairs {
            if assigned[si].is_none() && !det_used[di] {
                assigned[si] = Some(di);
                det_used[di] = true;
            }
        }

        // Update, coast, retire.
        let window = low_bits(birth_n);
        let mut retained = Vec::with_capacity(self.live.len() + detections.len());
        for (si, mut s) in std::mem::take(&mut self.live).into_iter().enumerate() {
            let (pred_c, pred_bw) = predicted[si];
            s.hit_mask = if dt >= 64 { 0 } else { s.hit_mask << dt };
            match assigned[si] {
                Some(di) => {
                    let d = &detections[di];
                    let residual_c = d.centroid_offset_hz(meta) - pred_c;
                    s.track.center_offset_hz = pred_c + self.cfg.alpha * residual_c;
                    s.vel_hz += self.cfg.beta * residual_c / dtf;
                    let det_bw = (d.bin_hi - d.bin_lo + 1) as f64 * bin_hz;
                    let residual_bw = det_bw - pred_bw;
                    s.track.bandwidth_hz = (pred_bw + self.cfg.alpha * residual_bw).max(bin_hz);
                    s.bw_vel_hz += self.cfg.beta * residual_bw / dtf;
                    s.track.power_dbfs +=
                        self.cfg.power_alpha * (d.power_dbfs - s.track.power_dbfs);
                    s.track.snr_db = d.snr_db;
                    s.track.last_seen_frame = d.frame_hi.max(frame);
                    s.misses = 0;
                    let span = (d.frame_hi - d.frame_lo + 1).min(u64::from(birth_n));
                    s.hit_mask |= low_bits(span as u32);
                    match s.track.state {
                        TrackState::Tentative => {
                            if (s.hit_mask & window).count_ones() >= self.cfg.birth_m {
                                s.track.state = TrackState::Confirmed;
                            }
                        }
                        TrackState::Coasting => s.track.state = TrackState::Confirmed,
                        TrackState::Confirmed => {}
                        TrackState::Dead => unreachable!("live list never holds Dead tracks"),
                    }
                    retained.push(s);
                }
                None => {
                    s.misses = s.misses.saturating_add(dt.min(u64::from(u32::MAX)) as u32);
                    match s.track.state {
                        TrackState::Tentative => {
                            // M-of-N verdict: drop only once confirmation is
                            // provably unsatisfiable — the hits so far plus
                            // every frame left in the birth window cannot
                            // reach M (at age >= N this reduces to the plain
                            // "window passed without M hits" expiry).
                            // `max_coast_frames` is a death allowance for
                            // confirmed tracks and never governs birth.
                            let age = frame - s.track.born_frame + 1;
                            let hits = u64::from((s.hit_mask & window).count_ones());
                            let remaining = u64::from(birth_n).saturating_sub(age);
                            if hits + remaining >= u64::from(self.cfg.birth_m) {
                                retained.push(s);
                            }
                        }
                        TrackState::Confirmed | TrackState::Coasting => {
                            // Dead-reckon through the coast so re-acquisition
                            // gates around where the signal would be by now.
                            s.track.center_offset_hz = pred_c;
                            s.track.bandwidth_hz = pred_bw;
                            if s.misses > max_coast_frames {
                                s.track.state = TrackState::Dead;
                                self.retired.push(s.track);
                            } else {
                                s.track.state = TrackState::Coasting;
                                retained.push(s);
                            }
                        }
                        TrackState::Dead => unreachable!("live list never holds Dead tracks"),
                    }
                }
            }
        }

        // Unassociated detections found new tentative tracks. A delivered
        // multi-frame blob is credited its whole span, so an already-proven
        // blob can confirm at birth.
        for (di, d) in detections.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            let span = (d.frame_hi - d.frame_lo + 1).min(u64::from(birth_n));
            let hit_mask = low_bits(span as u32);
            let state = if hit_mask.count_ones() >= self.cfg.birth_m {
                TrackState::Confirmed
            } else {
                TrackState::Tentative
            };
            let track = Track {
                id: TrackId(self.next_id),
                state,
                center_offset_hz: d.centroid_offset_hz(meta),
                bandwidth_hz: (d.bin_hi - d.bin_lo + 1) as f64 * bin_hz,
                power_dbfs: d.power_dbfs,
                snr_db: d.snr_db,
                born_frame: d.frame_lo,
                last_seen_frame: d.frame_hi.max(frame),
                measurements: None,
                class: None,
                signatures: Vec::new(),
            };
            self.next_id += 1;
            retained.push(Slot {
                track,
                vel_hz: 0.0,
                bw_vel_hz: 0.0,
                hit_mask,
                misses: 0,
            });
        }
        self.live = retained;
    }

    /// The live confirmed and coasting tracks, in birth order. Tentative
    /// tracks are internal — M-of-N birth means a not-yet-confirmed blip is
    /// never visible downstream.
    pub fn tracks(&self) -> impl Iterator<Item = &Track> {
        self.live
            .iter()
            .map(|s| &s.track)
            .filter(|t| !matches!(t.state, TrackState::Tentative))
    }

    /// Confirmed tracks that have died (miss timeout), in death order —
    /// the completed bursts hopper clustering consumes. Grows for the
    /// tracker's life unless drained with
    /// [`drain_retired`](Self::drain_retired).
    pub fn retired(&self) -> &[Track] {
        &self.retired
    }

    /// Take ownership of the retired tracks, leaving the list empty.
    pub fn drain_retired(&mut self) -> Vec<Track> {
        std::mem::take(&mut self.retired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_at(frame_dt_s: f64) -> BandMeta {
        BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s,
            cal_offset_db: 0.0,
        }
    }

    /// The M0 calibration rate `TrackerConfig::default()`'s comment states
    /// (2.048 MS/s, FFT 512): at this rate the default config's
    /// `birth_window_s`/`max_coast_s` convert to exactly 5 and 2 frames, so
    /// tests written against the pre-D-125 frame-counted API still read the
    /// same at this one frame rate.
    fn meta() -> BandMeta {
        meta_at(250.0e-6)
    }

    /// A one-frame detection centred on `bin` with a ±1-bin extent.
    fn det(frame: u64, bin: f64) -> Detection {
        Detection {
            bin_lo: (bin - 1.0).round() as usize,
            bin_hi: (bin + 1.0).round() as usize,
            frame_lo: frame,
            frame_hi: frame,
            centroid_bin: bin,
            peak_dbfs: -20.0,
            power_dbfs: -19.0,
            snr_db: 40.0,
        }
    }

    fn tracker() -> Tracker {
        Tracker::new(TrackerConfig::default()).unwrap()
    }

    #[test]
    fn rejects_bad_configs_with_named_fields() {
        let mut c = TrackerConfig {
            birth_m: 0,
            ..TrackerConfig::default()
        };
        let TrackerError::InvalidConfig(msg) = Tracker::new(c.clone()).unwrap_err();
        assert!(msg.contains("birth_m"), "unhelpful: {msg}");

        c.birth_m = 3;
        c.birth_window_s = 0.0;
        let TrackerError::InvalidConfig(msg) = Tracker::new(c.clone()).unwrap_err();
        assert!(msg.contains("birth_window_s"), "unhelpful: {msg}");

        c.birth_window_s = 5.0 * 250.0e-6;
        c.max_coast_s = -1.0;
        let TrackerError::InvalidConfig(msg) = Tracker::new(c.clone()).unwrap_err();
        assert!(msg.contains("max_coast_s"), "unhelpful: {msg}");

        c.max_coast_s = 2.0 * 250.0e-6;
        c.alpha = 0.0;
        let TrackerError::InvalidConfig(msg) = Tracker::new(c.clone()).unwrap_err();
        assert!(msg.contains("alpha"), "unhelpful: {msg}");

        c.alpha = 0.5;
        c.bw_gate_ratio = 1.0;
        let TrackerError::InvalidConfig(msg) = Tracker::new(c).unwrap_err();
        assert!(msg.contains("bw_gate_ratio"), "unhelpful: {msg}");
    }

    #[test]
    fn steady_tone_yields_exactly_one_track_with_a_stable_id() {
        let m = meta();
        let mut tr = tracker();
        let mut seen_id = None;
        for frame in 0..50 {
            tr.observe(frame, &[det(frame, 812.0)], &m);
            if let Some(t) = tr.tracks().next() {
                let id = seen_id.get_or_insert(t.id);
                assert_eq!(*id, t.id, "id changed mid-life at frame {frame}");
            }
        }
        let tracks: Vec<&Track> = tr.tracks().collect();
        assert_eq!(tracks.len(), 1, "a steady tone is exactly one track");
        let t = tracks[0];
        assert_eq!(t.state, TrackState::Confirmed);
        assert_eq!(t.born_frame, 0);
        assert_eq!(t.last_seen_frame, 49);
        assert!((t.center_offset_hz - 300_000.0).abs() < 1.0);
        assert!(tr.retired().is_empty());
    }

    #[test]
    fn m_of_n_birth_suppresses_a_single_frame_blip() {
        // The default `birth_m: 1` (AN-1/AN-3, D-126) confirms on the first
        // hit by design — suppressing a lone blip is now the upstream blob-
        // size/Pfa gate's job (`DetectorConfig::min_blob_cells`,
        // `CfarConfig::pfa`), not the tracker's. This test exercises the
        // M-of-N mechanism itself, still fully present at `birth_m > 1`, on
        // an explicit config rather than the shipped default.
        let m = meta();
        let cfg = TrackerConfig {
            birth_m: 3,
            ..TrackerConfig::default()
        };
        let mut tr = Tracker::new(cfg).unwrap();
        for frame in 0..20 {
            let dets = if frame == 5 {
                vec![det(frame, 200.0)]
            } else {
                Vec::new()
            };
            tr.observe(frame, &dets, &m);
            assert_eq!(tr.tracks().count(), 0, "blip leaked at frame {frame}");
        }
        assert!(
            tr.retired().is_empty(),
            "blips never reach the retired list"
        );
    }

    #[test]
    fn death_retires_a_departed_signal_after_the_coast() {
        let m = meta();
        let cfg = TrackerConfig::default();
        // `meta()` runs at the default config's own calibration frame rate
        // (250us), so this converts to exactly 2 frames, matching the
        // pre-D-125 frame-counted value.
        let (last_hit, max_coast) = (29u64, (cfg.max_coast_s / m.frame_dt_s).round() as u64);
        let mut tr = Tracker::new(cfg).unwrap();
        for frame in 0..=last_hit {
            tr.observe(frame, &[det(frame, 812.0)], &m);
        }
        // Misses within the allowance: the track coasts, still visible.
        for frame in (last_hit + 1)..=(last_hit + max_coast) {
            tr.observe(frame, &[], &m);
            let t = tr.tracks().next().expect("coasting track stays visible");
            assert_eq!(t.state, TrackState::Coasting);
        }
        // One miss past the allowance: retired as Dead.
        tr.observe(last_hit + max_coast + 1, &[], &m);
        assert_eq!(tr.tracks().count(), 0);
        assert_eq!(tr.retired().len(), 1);
        let dead = &tr.retired()[0];
        assert_eq!(dead.state, TrackState::Dead);
        assert_eq!(dead.last_seen_frame, last_hit);
    }

    #[test]
    fn a_reappearing_signal_within_the_coast_keeps_its_track() {
        let m = meta();
        let mut tr = tracker();
        for frame in 0..10 {
            tr.observe(frame, &[det(frame, 812.0)], &m);
        }
        let id = tr.tracks().next().unwrap().id;
        tr.observe(10, &[], &m); // one dropout frame
        tr.observe(11, &[det(11, 812.0)], &m);
        let t = tr.tracks().next().unwrap();
        assert_eq!(t.id, id);
        assert_eq!(t.state, TrackState::Confirmed);
        assert_eq!(tr.tracks().count(), 1);
    }

    #[test]
    fn crossing_pair_does_not_swap_ids() {
        let m = meta();
        let mut tr = tracker();
        // A rises from bin 412 at +2 bins/frame; B falls from bin 612 at −2
        // bins/frame; they meet at bin 512 on frame 50 and part again.
        let pos_a = |frame: u64| 412.0 + 2.0 * frame as f64;
        let pos_b = |frame: u64| 612.0 - 2.0 * frame as f64;
        let mut id_a = None;
        let mut id_b = None;
        for frame in 0..100 {
            tr.observe(
                frame,
                &[det(frame, pos_a(frame)), det(frame, pos_b(frame))],
                &m,
            );
            if frame == 10 {
                // Both confirmed and moving; capture who is who by position.
                let mut tracks: Vec<&Track> = tr.tracks().collect();
                assert_eq!(tracks.len(), 2);
                tracks.sort_by(|x, y| x.center_offset_hz.total_cmp(&y.center_offset_hz));
                id_a = Some(tracks[0].id); // A is still the lower one
                id_b = Some(tracks[1].id);
            }
        }
        let tracks: Vec<&Track> = tr.tracks().collect();
        assert_eq!(tracks.len(), 2, "the pair stays two tracks throughout");
        for t in tracks {
            let truth = if Some(t.id) == id_a {
                m.bin_offset_hz(pos_a(99))
            } else {
                assert_eq!(Some(t.id), id_b, "unexpected third id");
                m.bin_offset_hz(pos_b(99))
            };
            assert!(
                (t.center_offset_hz - truth).abs() < 2.0 * m.bin_hz,
                "id {:?} ended at {} Hz, its trajectory says {} Hz — ids swapped",
                t.id,
                t.center_offset_hz,
                truth
            );
        }
    }

    #[test]
    fn tentative_birth_is_not_governed_by_the_coast_allowance() {
        // Permissive M-of-N (2-of-5) with a tight death coast (1 frame): a
        // signal seen at frames 0 and 3 can still reach 2-of-5, so the two
        // missed frames in between must not cull it — max_coast_frames is a
        // death allowance for confirmed tracks, never a birth rule.
        let m = meta();
        // birth_window_s = 5 frames, max_coast_s = 1 frame, both at meta()'s
        // 250us frame rate — same numbers the pre-D-125 test used.
        let cfg = TrackerConfig {
            birth_m: 2,
            birth_window_s: 5.0 * 250.0e-6,
            max_coast_s: 1.0 * 250.0e-6,
            ..TrackerConfig::default()
        };
        let mut tr = Tracker::new(cfg).unwrap();
        tr.observe(0, &[det(0, 812.0)], &m);
        tr.observe(1, &[], &m);
        tr.observe(2, &[], &m); // 2 consecutive misses > max_coast_s's 1 frame
        tr.observe(3, &[det(3, 812.0)], &m);
        let t = tr.tracks().next().expect("track must survive to confirm");
        assert_eq!(t.state, TrackState::Confirmed);
        assert_eq!(t.born_frame, 0);

        // The complement still holds: once 2-of-5 is provably unsatisfiable
        // (a lone hit, then enough misses that hits + frames left < M), the
        // tentative is dropped without ever becoming visible.
        let mut tr = Tracker::new(TrackerConfig {
            birth_m: 2,
            birth_window_s: 5.0 * 250.0e-6,
            max_coast_s: 1.0 * 250.0e-6,
            ..TrackerConfig::default()
        })
        .unwrap();
        tr.observe(0, &[det(0, 812.0)], &m);
        for frame in 1..8 {
            tr.observe(frame, &[], &m);
        }
        assert_eq!(tr.tracks().count(), 0);
        assert!(tr.retired().is_empty());
    }

    #[test]
    fn a_delivered_multiframe_blob_confirms_at_birth() {
        let m = meta();
        let mut tr = tracker();
        // One completed 4-frame blob delivered at its final frame.
        let blob = Detection {
            frame_lo: 3,
            frame_hi: 6,
            ..det(6, 300.0)
        };
        tr.observe(6, &[blob], &m);
        let t = tr.tracks().next().expect("blob span covers M frames");
        assert_eq!(t.state, TrackState::Confirmed);
        assert_eq!(t.born_frame, 3);
        assert_eq!(t.last_seen_frame, 6);
    }

    /// D-125 item 3: `max_coast_s` is a duration. The same coast allowance
    /// must survive the same real time regardless of frame rate — at 4x the
    /// frame rate, 4x as many missed frames are tolerated before death, not
    /// the same frame count.
    #[test]
    fn coast_allowance_is_the_same_wall_time_at_two_frame_rates() {
        for &frame_dt_s in &[250.0e-6, 62.5e-6] {
            let m = meta_at(frame_dt_s);
            let cfg = TrackerConfig {
                max_coast_s: 500.0e-6, // 2 frames at 250us, 8 frames at 62.5us
                ..TrackerConfig::default()
            };
            let expected_coast_frames = (cfg.max_coast_s / frame_dt_s).round() as u64;
            let mut tr = Tracker::new(cfg).unwrap();
            let last_hit = 29u64;
            for frame in 0..=last_hit {
                tr.observe(frame, &[det(frame, 812.0)], &m);
            }
            // Misses within the allowance: the track coasts, still visible,
            // for `expected_coast_frames` frames — the same real time at
            // every rate.
            for frame in (last_hit + 1)..=(last_hit + expected_coast_frames) {
                tr.observe(frame, &[], &m);
                let t = tr.tracks().next().unwrap_or_else(|| {
                    panic!("frame_dt_s={frame_dt_s}: coasting track stays visible at frame {frame}")
                });
                assert_eq!(t.state, TrackState::Coasting);
            }
            // One miss past the allowance: retired as Dead, at the same
            // real time (last_hit's instant plus max_coast_s) either way.
            tr.observe(last_hit + expected_coast_frames + 1, &[], &m);
            assert_eq!(
                tr.tracks().count(),
                0,
                "frame_dt_s={frame_dt_s}: should have died by now"
            );
            assert_eq!(tr.retired().len(), 1, "frame_dt_s={frame_dt_s}");
        }
    }

    /// D-125 item 3: `birth_window_s` is a duration too. A signal seen
    /// twice within the same real-time window confirms at either frame
    /// rate, even though the frame *count* spanning the two hits differs
    /// 4x between them.
    #[test]
    fn birth_window_is_the_same_wall_time_at_two_frame_rates() {
        let birth_window_s = 1.25e-3; // the M0 calibration: 5 frames @ 250us
        for &frame_dt_s in &[250.0e-6, 62.5e-6] {
            let m = meta_at(frame_dt_s);
            let cfg = TrackerConfig {
                birth_m: 2,
                birth_window_s,
                ..TrackerConfig::default()
            };
            let window_frames = (birth_window_s / frame_dt_s).round() as u64;
            let mut tr = Tracker::new(cfg).unwrap();
            tr.observe(0, &[det(0, 812.0)], &m);
            for frame in 1..(window_frames - 1) {
                tr.observe(frame, &[], &m);
            }
            // The second hit lands one frame before the window's own age
            // limit (`window_frames`), the last frame at which a second hit
            // can still complete 2-of-N — same real-time boundary at every
            // rate.
            tr.observe(window_frames - 1, &[det(window_frames - 1, 812.0)], &m);
            let t = tr.tracks().next().unwrap_or_else(|| {
                panic!("frame_dt_s={frame_dt_s}: track must confirm within its birth window")
            });
            assert_eq!(t.state, TrackState::Confirmed);
        }
    }

    #[test]
    fn identical_inputs_yield_identical_output() {
        let m = meta();
        let run = || {
            let mut tr = tracker();
            for frame in 0..60 {
                let mut dets = vec![det(frame, 812.0)];
                if frame % 15 < 5 {
                    dets.push(det(frame, 137.0 + 250.0 * f64::from(frame as u32 / 15 % 4)));
                }
                tr.observe(frame, &dets, &m);
            }
            let live: Vec<Track> = tr.tracks().cloned().collect();
            (
                serde_json::to_string(&live).unwrap(),
                serde_json::to_string(tr.retired()).unwrap(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn observe_rejects_non_monotonic_frames() {
        let m = meta();
        let mut tr = tracker();
        tr.observe(5, &[], &m);
        tr.observe(5, &[], &m);
    }
}
