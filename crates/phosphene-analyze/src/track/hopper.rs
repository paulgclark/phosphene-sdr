// SPDX-License-Identifier: MIT

//! FR-AD5 hopper / burst clustering: fold many short co-parametric burst
//! tracks into one logical [`Emitter`] (FHSS/TDMA), recovering the hop set
//! and rate (nextgen spec §7.4).
//!
//! Frequency hopping is itself a **reportable behaviour**, not noise to be
//! suppressed: without this stage a hopper produces hundreds of ephemeral
//! tracks and the annotation overlay becomes unreadable. With it, the
//! overlay shows one emitter whose hop set and rate are stated.
//!
//! ## Method (spec §7.4, *tune* thresholds calibrated on the M0 generator)
//!
//! Candidate bursts are tracks whose whole life fits inside
//! [`HopperClusterConfig::max_burst_s`] — a steady carrier never qualifies.
//! Candidates are grouped by co-parametric bandwidth (modulation class would
//! join the criterion in phase A1+, when classification exists); inside a
//! group, burst centres collapse into distinct hop frequencies. Shared
//! bandwidth alone is not evidence of one emitter — two independent radios
//! of the same type are co-parametric too — so frequency clusters then split
//! by **temporal exclusivity**: a single FHSS/TDMA emitter occupies one
//! frequency at a time, so bursts that overlap in time can never share an
//! emitter, and concurrent same-bandwidth hoppers come out as separate
//! emitters. Each exclusive partition becomes an emitter only if its bursts
//! are *regular* in at least one of the two spec §7.4 senses:
//!
//! * **time-slot regularity** — burst start times recur at a stable period
//!   (low relative median-absolute-deviation), which also yields the hop
//!   rate; or
//! * **carrier-grid regularity** — three or more distinct hop frequencies
//!   sit on a common spacing grid (two points always fit a grid, so a
//!   two-frequency set is evidence only through its timing).
//!
//! Irregular co-parametric bursts stay individual tracks: clustering must
//! never invent an emitter the evidence does not support (NFR-A3).
//!
//! Deterministic by construction (NFR-A4): no randomness, ordered storage,
//! `total_cmp` sorts with index tie-breaks.

use serde::{Deserialize, Serialize};

use crate::measure::Measured;
use crate::tap::BandMeta;
use crate::track::{Track, TrackId};

/// Tuning for [`cluster_hoppers`]. All values are spec §7 *tune* constants.
#[derive(Debug, Clone, PartialEq)]
pub struct HopperClusterConfig {
    /// Longest track life (seconds) still considered a burst; anything
    /// longer — a steady tone, a long transmission — is never clustered.
    pub max_burst_s: f64,
    /// Minimum bursts before a group may become an emitter.
    pub min_bursts: usize,
    /// Bandwidth grouping: adjacent (by bandwidth) bursts stay in one
    /// co-parametric group while each bandwidth is within this ratio of its
    /// neighbour (`> 1`).
    pub bw_group_ratio: f64,
    /// Centres closer than this collapse into one hop frequency. `0.0`
    /// selects an automatic tolerance of `max(2 bins, half the group's
    /// median bandwidth)`.
    pub freq_tol_hz: f64,
    /// Time-slot regularity bound: relative median-absolute-deviation of the
    /// burst-start intervals must not exceed this (`> 0`).
    pub timing_tol: f64,
    /// Carrier-grid regularity bound: each hop frequency must sit within
    /// this fraction of the grid spacing of an exact grid point (`> 0`).
    pub grid_tol: f64,
}

impl Default for HopperClusterConfig {
    /// M0-calibrated defaults: bursts under 100 ms, at least 4 of them, 25%
    /// timing jitter, 8% grid tolerance.
    fn default() -> Self {
        HopperClusterConfig {
            max_burst_s: 0.1,
            min_bursts: 4,
            bw_group_ratio: 2.0,
            freq_tol_hz: 0.0,
            timing_tol: 0.25,
            grid_tol: 0.08,
        }
    }
}

/// One logical emitter recovered from many burst tracks (FR-AD5). Frequencies
/// are offsets from band centre, like [`Track::center_offset_hz`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    /// The member burst tracks, in burst-start order.
    pub member_ids: Vec<TrackId>,
    /// Distinct hop frequencies (offset Hz, ascending). One entry for a
    /// single-frequency (TDMA-like) bursty emitter.
    pub hop_set_hz: Vec<f64>,
    /// Grid spacing in Hz when the hop set sits on a regular carrier grid —
    /// asserted only for three or more distinct frequencies, since two
    /// points always fit a grid.
    pub carrier_spacing_hz: Option<f64>,
    /// Hops (bursts) per second, when the time-slot pattern is regular; the
    /// confidence reflects the timing jitter (NFR-A3).
    pub hop_rate_hz: Option<Measured<f64>>,
    /// Mean burst (dwell) length, seconds, with a confidence from the
    /// dwell spread.
    pub mean_dwell_s: Measured<f64>,
    /// Median burst bandwidth, Hz — the co-parametric signature the members
    /// share.
    pub bandwidth_hz: f64,
    /// True when the emitter visits two or more frequencies — the reportable
    /// hopping behaviour of FR-AD5.
    pub hopping: bool,
}

/// One candidate burst distilled from a track.
struct Burst {
    id: TrackId,
    center_hz: f64,
    bandwidth_hz: f64,
    born_frame: u64,
    last_frame: u64,
    duration_s: f64,
}

impl Burst {
    /// True when the two bursts are on the air at the same time — which a
    /// single FHSS/TDMA emitter, transmitting on one frequency at a time,
    /// can never be.
    fn overlaps(&self, other: &Burst) -> bool {
        self.born_frame <= other.last_frame && other.born_frame <= self.last_frame
    }
}

/// Bursts sharing one hop frequency (centres within the merge tolerance).
type FreqCluster<'a> = Vec<&'a Burst>;

/// Median of a non-empty, ascending-sorted slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// (median, relative median-absolute-deviation) of a non-empty set.
fn median_and_rel_mad(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let med = median_sorted(&sorted);
    let mut dev: Vec<f64> = sorted.iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let mad = median_sorted(&dev);
    if med.abs() > 0.0 {
        (med, mad / med.abs())
    } else {
        (med, f64::INFINITY)
    }
}

/// Cluster short co-parametric burst tracks into logical emitters (FR-AD5).
///
/// Pass every track worth considering — typically the tracker's
/// [`retired`](crate::track::Tracker::retired) list plus its live
/// [`tracks`](crate::track::Tracker::tracks); long-lived tracks filter
/// themselves out. Tracks that join no emitter simply remain individual
/// tracks. Emitters are returned ordered by their lowest hop frequency.
///
/// # Panics
///
/// Panics if the configuration is degenerate (non-positive tolerances or
/// durations, `min_bursts` of zero, or a bandwidth ratio not above 1).
pub fn cluster_hoppers<'a, I>(tracks: I, meta: &BandMeta, cfg: &HopperClusterConfig) -> Vec<Emitter>
where
    I: IntoIterator<Item = &'a Track>,
{
    assert!(
        cfg.max_burst_s > 0.0 && cfg.max_burst_s.is_finite(),
        "max_burst_s must be finite and positive"
    );
    assert!(cfg.min_bursts >= 1, "min_bursts must be at least 1");
    assert!(
        cfg.bw_group_ratio > 1.0 && cfg.bw_group_ratio.is_finite(),
        "bw_group_ratio must be finite and > 1"
    );
    assert!(
        cfg.freq_tol_hz >= 0.0 && cfg.freq_tol_hz.is_finite(),
        "freq_tol_hz must be finite and >= 0"
    );
    assert!(
        cfg.timing_tol > 0.0 && cfg.timing_tol.is_finite(),
        "timing_tol must be finite and positive"
    );
    assert!(
        cfg.grid_tol > 0.0 && cfg.grid_tol.is_finite(),
        "grid_tol must be finite and positive"
    );

    let mut bursts: Vec<Burst> = tracks
        .into_iter()
        .filter_map(|t| {
            let duration_s = (t.last_seen_frame - t.born_frame + 1) as f64 * meta.frame_dt_s;
            (duration_s <= cfg.max_burst_s).then_some(Burst {
                id: t.id,
                center_hz: t.center_offset_hz,
                bandwidth_hz: t.bandwidth_hz,
                born_frame: t.born_frame,
                last_frame: t.last_seen_frame,
                duration_s,
            })
        })
        .collect();

    // Co-parametric grouping: ascending by bandwidth, split where the ratio
    // step to the neighbour exceeds the group ratio.
    bursts.sort_by(|a, b| {
        a.bandwidth_hz
            .total_cmp(&b.bandwidth_hz)
            .then(a.born_frame.cmp(&b.born_frame))
            .then(a.id.cmp(&b.id))
    });
    let mut emitters = Vec::new();
    let mut start = 0;
    for i in 1..=bursts.len() {
        let split = i == bursts.len()
            || bursts[i].bandwidth_hz
                > bursts[i - 1].bandwidth_hz.max(meta.bin_hz) * cfg.bw_group_ratio;
        if split {
            emitters.extend(emit_group(&bursts[start..i], meta, cfg));
            start = i;
        }
    }
    emitters.sort_by(|a, b| a.hop_set_hz[0].total_cmp(&b.hop_set_hz[0]));
    emitters
}

/// Fold one same-bandwidth group into zero or more emitters.
///
/// Shared bandwidth alone is not sufficient evidence of one emitter: two
/// independent radios of the same type are co-parametric too. So the group
/// is first split by **temporal exclusivity** — a single FHSS/TDMA emitter
/// occupies one frequency at a time, so frequency clusters whose bursts
/// overlap in time cannot share an emitter — and each exclusive partition
/// must then pass the size and regularity evidence on its own.
fn emit_group(group: &[Burst], meta: &BandMeta, cfg: &HopperClusterConfig) -> Vec<Emitter> {
    if group.len() < cfg.min_bursts {
        return Vec::new();
    }
    let mut bws: Vec<f64> = group.iter().map(|b| b.bandwidth_hz).collect();
    bws.sort_by(f64::total_cmp);
    let group_median_bw = median_sorted(&bws);

    // Collapse centres into frequency clusters (single-linkage chaining on
    // the sorted centres, same tolerance as before).
    let freq_tol = if cfg.freq_tol_hz > 0.0 {
        cfg.freq_tol_hz
    } else {
        (2.0 * meta.bin_hz).max(0.5 * group_median_bw)
    };
    let mut by_center: Vec<&Burst> = group.iter().collect();
    by_center.sort_by(|a, b| {
        a.center_hz
            .total_cmp(&b.center_hz)
            .then(a.born_frame.cmp(&b.born_frame))
            .then(a.id.cmp(&b.id))
    });
    let mut clusters: Vec<FreqCluster<'_>> = Vec::new();
    for b in by_center {
        match clusters.last_mut() {
            Some(c)
                if b.center_hz - c.last().expect("clusters are non-empty").center_hz
                    <= freq_tol =>
            {
                c.push(b);
            }
            _ => clusters.push(vec![b]),
        }
    }

    // Temporal exclusivity: greedily place each frequency cluster (ascending
    // frequency) into the first partition none of whose bursts it overlaps
    // in time; concurrent transmissions force a new partition. Deterministic:
    // cluster order and partition scan order are both fixed.
    let mut partitions: Vec<Vec<FreqCluster<'_>>> = Vec::new();
    'clusters: for cluster in clusters {
        for partition in &mut partitions {
            let concurrent = partition
                .iter()
                .flatten()
                .any(|a| cluster.iter().any(|b| a.overlaps(b)));
            if !concurrent {
                partition.push(cluster);
                continue 'clusters;
            }
        }
        partitions.push(vec![cluster]);
    }

    partitions
        .into_iter()
        .filter_map(|p| emit_partition(&p, meta, cfg))
        .collect()
}

/// Try to fold one temporally-exclusive partition into an emitter; `None`
/// when it fails the size or regularity evidence.
fn emit_partition(
    partition: &[FreqCluster<'_>],
    meta: &BandMeta,
    cfg: &HopperClusterConfig,
) -> Option<Emitter> {
    let bursts: Vec<&Burst> = partition.iter().flatten().copied().collect();
    if bursts.len() < cfg.min_bursts {
        return None;
    }
    let mut bws: Vec<f64> = bursts.iter().map(|b| b.bandwidth_hz).collect();
    bws.sort_by(f64::total_cmp);
    let median_bw = median_sorted(&bws);
    let freq_tol = if cfg.freq_tol_hz > 0.0 {
        cfg.freq_tol_hz
    } else {
        (2.0 * meta.bin_hz).max(0.5 * median_bw)
    };

    // One hop frequency per cluster; ascending for a stable hop set.
    let mut hop_set_hz: Vec<f64> = partition
        .iter()
        .map(|c| c.iter().map(|b| b.center_hz).sum::<f64>() / c.len() as f64)
        .collect();
    hop_set_hz.sort_by(f64::total_cmp);

    // Time-slot regularity: intervals between burst starts recur.
    let mut starts: Vec<f64> = bursts
        .iter()
        .map(|b| b.born_frame as f64 * meta.frame_dt_s)
        .collect();
    starts.sort_by(f64::total_cmp);
    let intervals: Vec<f64> = starts.windows(2).map(|w| w[1] - w[0]).collect();
    let mut hop_rate_hz = None;
    let mut timing_regular = false;
    if intervals.len() >= 2 {
        let (period_s, rel_mad) = median_and_rel_mad(&intervals);
        if period_s > 0.0 && rel_mad <= cfg.timing_tol {
            timing_regular = true;
            let confidence = (1.0 - rel_mad / cfg.timing_tol).clamp(0.0, 1.0) as f32;
            hop_rate_hz = Some(Measured::new(1.0 / period_s, confidence));
        }
    }

    // Carrier-grid regularity: distinct frequencies on a common spacing.
    // Two points always define a spacing — round((f1 − f0)/spacing) hits an
    // integer with zero residual by construction — so a grid is only
    // EVIDENCE at three or more distinct frequencies; with exactly two,
    // only timing regularity may support the hop claim.
    let mut carrier_spacing_hz = None;
    if hop_set_hz.len() >= 3 {
        let spacing = hop_set_hz
            .windows(2)
            .map(|w| w[1] - w[0])
            .min_by(f64::total_cmp)
            .expect("three or more hop frequencies have adjacent gaps");
        let tol = (cfg.grid_tol * spacing).max(freq_tol);
        let on_grid = spacing > 0.0
            && hop_set_hz.iter().all(|f| {
                let steps = ((f - hop_set_hz[0]) / spacing).round();
                (f - hop_set_hz[0] - steps * spacing).abs() <= tol
            });
        if on_grid {
            carrier_spacing_hz = Some(spacing);
        }
    }

    // Spec §7.4: a regular carrier grid OR a regular time-slot pattern.
    let grid_regular = carrier_spacing_hz.is_some();
    if !(timing_regular || grid_regular) {
        return None;
    }

    // FR-AM4 asks for the MEAN burst duration; the confidence comes from
    // the spread (relative MAD) of the durations around their median.
    let durations: Vec<f64> = bursts.iter().map(|b| b.duration_s).collect();
    let mean_dwell = durations.iter().sum::<f64>() / durations.len() as f64;
    let (_, dwell_rel_mad) = median_and_rel_mad(&durations);
    let dwell_confidence = (1.0 - dwell_rel_mad).clamp(0.0, 1.0) as f32;

    let mut ordered: Vec<&Burst> = bursts.clone();
    ordered.sort_by(|a, b| a.born_frame.cmp(&b.born_frame).then(a.id.cmp(&b.id)));
    Some(Emitter {
        member_ids: ordered.iter().map(|b| b.id).collect(),
        hopping: hop_set_hz.len() >= 2,
        hop_set_hz,
        carrier_spacing_hz,
        hop_rate_hz,
        mean_dwell_s: Measured::new(mean_dwell, dwell_confidence),
        bandwidth_hz: median_bw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::TrackState;

    fn meta() -> BandMeta {
        BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        }
    }

    /// A completed burst track: born at `born`, alive `frames` frames, at
    /// `center_hz` offset with a 3 kHz bandwidth.
    fn burst(id: u64, born: u64, frames: u64, center_hz: f64) -> Track {
        Track {
            id: TrackId(id),
            state: TrackState::Dead,
            center_offset_hz: center_hz,
            bandwidth_hz: 3_000.0,
            power_dbfs: -25.0,
            snr_db: 40.0,
            born_frame: born,
            last_seen_frame: born + frames - 1,
            measurements: None,
            class: None,
            signatures: Vec::new(),
        }
    }

    /// The M0 hopper's shape: 4-frequency 250 kHz grid, 5-frame dwell,
    /// 15-frame period, cycling — known hop set and rate by construction.
    fn m0_hopper_bursts(count: u64) -> Vec<Track> {
        const HOPS: [f64; 4] = [-375_000.0, -125_000.0, 125_000.0, 375_000.0];
        (0..count)
            .map(|k| burst(k + 1, k * 15, 5, HOPS[(k % 4) as usize]))
            .collect()
    }

    #[test]
    fn known_hopper_clusters_to_one_emitter_with_hop_set_and_rate() {
        let m = meta();
        let mut tracks = m0_hopper_bursts(24);
        // A steady tone spanning the whole scene must not be swallowed.
        let mut tone = burst(100, 0, 360, 450_000.0);
        tone.state = TrackState::Confirmed;
        tracks.push(tone);

        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 1, "one hopper, one emitter");
        let e = &emitters[0];
        assert!(e.hopping);
        assert_eq!(e.member_ids.len(), 24);
        assert!(!e.member_ids.contains(&TrackId(100)), "tone clustered");
        assert_eq!(e.hop_set_hz.len(), 4, "hop set: {:?}", e.hop_set_hz);
        for (got, want) in e.hop_set_hz.iter().zip([-375e3, -125e3, 125e3, 375e3]) {
            assert!((got - want).abs() < 1.0, "hop {got} != {want}");
        }
        let rate = e.hop_rate_hz.expect("regular timing yields a rate");
        assert!(
            (rate.value - 1.0 / 0.015).abs() < 0.5,
            "rate {}",
            rate.value
        );
        assert!(rate.confidence > 0.9);
        assert!((e.mean_dwell_s.value - 0.005).abs() < 1e-9);
        assert_eq!(e.carrier_spacing_hz, Some(250_000.0));
    }

    #[test]
    fn two_concurrent_same_bandwidth_hoppers_stay_two_emitters() {
        let m = meta();
        // Two independent FHSS radios of the same type: identical bandwidth,
        // identical 15-frame period and 5-frame dwell, transmitting
        // CONCURRENTLY (same burst times) on disjoint hop sets. Bandwidth
        // alone would fold them into one false emitter; temporal exclusivity
        // must keep them apart.
        const HOPS_A: [f64; 2] = [-375_000.0, -125_000.0];
        const HOPS_B: [f64; 2] = [125_000.0, 375_000.0];
        let mut tracks = Vec::new();
        for k in 0..12u64 {
            tracks.push(burst(k + 1, k * 15, 5, HOPS_A[(k % 2) as usize]));
            tracks.push(burst(k + 101, k * 15, 5, HOPS_B[(k % 2) as usize]));
        }

        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 2, "two radios, two emitters: {emitters:#?}");
        let (a, b) = (&emitters[0], &emitters[1]);
        for (e, hops, id_base) in [(a, HOPS_A, 1u64), (b, HOPS_B, 101)] {
            assert!(e.hopping);
            assert_eq!(e.member_ids.len(), 12);
            assert!(e
                .member_ids
                .iter()
                .all(|id| (id_base..id_base + 12).contains(&id.0)));
            assert_eq!(e.hop_set_hz.len(), 2, "hop set: {:?}", e.hop_set_hz);
            for (got, want) in e.hop_set_hz.iter().zip(hops) {
                assert!((got - want).abs() < 1.0, "hop {got} != {want}");
            }
            let rate = e.hop_rate_hz.expect("regular timing yields a rate");
            assert!((rate.value - 1.0 / 0.015).abs() < 0.5);
        }
    }

    #[test]
    fn interleaved_nonoverlapping_hop_sets_still_split_when_concurrent() {
        let m = meta();
        // Same as above but with the two radios' hop sets interleaved in
        // frequency — the split must come from time overlap, not from the
        // sets being band-separated.
        const HOPS_A: [f64; 2] = [-375_000.0, 125_000.0];
        const HOPS_B: [f64; 2] = [-125_000.0, 375_000.0];
        let mut tracks = Vec::new();
        for k in 0..12u64 {
            tracks.push(burst(k + 1, k * 15, 5, HOPS_A[(k % 2) as usize]));
            tracks.push(burst(k + 101, k * 15, 5, HOPS_B[(k % 2) as usize]));
        }
        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 2, "{emitters:#?}");
        for e in &emitters {
            assert_eq!(e.member_ids.len(), 12);
            assert_eq!(e.hop_set_hz.len(), 2);
        }
    }

    #[test]
    fn two_frequencies_need_regular_timing_because_two_points_always_fit_a_grid() {
        let m = meta();
        // The same two frequencies both times; only the timing differs. If
        // the irregular case produced an emitter, the two-point "grid"
        // would be the culprit — two points define a spacing with zero
        // residual by construction, so a pair of frequencies is never grid
        // evidence on its own.
        const F_LO: f64 = -200_000.0;
        const F_HI: f64 = 150_000.0;

        let irregular: Vec<Track> = [0u64, 9, 52, 61, 140, 310]
            .iter()
            .enumerate()
            .map(|(k, &b)| burst(k as u64 + 1, b, 5, if k % 2 == 0 { F_LO } else { F_HI }))
            .collect();
        assert!(
            cluster_hoppers(irregular.iter(), &m, &HopperClusterConfig::default()).is_empty(),
            "two frequencies at irregular timing must not be asserted as a hopper"
        );

        let regular: Vec<Track> = (0..6u64)
            .map(|k| burst(k + 1, k * 15, 5, if k % 2 == 0 { F_LO } else { F_HI }))
            .collect();
        let emitters = cluster_hoppers(regular.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 1, "regular timing still supports the claim");
        let e = &emitters[0];
        assert!(e.hopping);
        assert_eq!(e.hop_set_hz.len(), 2);
        assert_eq!(
            e.carrier_spacing_hz, None,
            "no grid asserted from only two frequencies"
        );
        assert!(e.hop_rate_hz.is_some());
    }

    #[test]
    fn mean_dwell_is_the_mean_not_the_median() {
        let m = meta();
        // 8 bursts on a strict 20-frame cadence and a 3-frequency grid:
        // durations of 4,4,4,4,4,4,4,12 frames have median 4 ms but mean
        // 5 ms — FR-AM4 asks for the mean.
        const HOPS: [f64; 3] = [-300_000.0, -100_000.0, 100_000.0];
        let tracks: Vec<Track> = (0..8u64)
            .map(|k| {
                let frames = if k == 7 { 12 } else { 4 };
                burst(k + 1, k * 20, frames, HOPS[(k % 3) as usize])
            })
            .collect();
        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 1);
        assert!(
            (emitters[0].mean_dwell_s.value - 0.005).abs() < 1e-12,
            "mean of (7×4 + 12) frames is 5 ms, got {} s",
            emitters[0].mean_dwell_s.value
        );
    }

    #[test]
    fn irregular_bursts_stay_unclustered() {
        let m = meta();
        // Co-parametric bandwidths, but frequencies off any grid and timing
        // with no stable period.
        let tracks = [
            burst(1, 0, 5, -310_000.0),
            burst(2, 7, 5, 143_000.0),
            burst(3, 40, 5, -51_000.0),
            burst(4, 45, 5, 402_000.0),
            burst(5, 120, 5, 87_000.0),
            burst(6, 300, 5, -222_000.0),
        ];
        assert!(
            cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default()).is_empty(),
            "no regularity, no emitter"
        );
    }

    #[test]
    fn single_frequency_regular_bursts_cluster_as_a_nonhopping_emitter() {
        let m = meta();
        // TDMA-like: one frequency, strict 20-frame slot cadence.
        let tracks: Vec<Track> = (0..8).map(|k| burst(k + 1, k * 20, 4, 100_000.0)).collect();
        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 1);
        let e = &emitters[0];
        assert!(!e.hopping, "one frequency is not hopping");
        assert_eq!(e.hop_set_hz.len(), 1);
        assert_eq!(e.carrier_spacing_hz, None);
        let rate = e.hop_rate_hz.expect("regular slots yield a rate");
        assert!((rate.value - 50.0).abs() < 0.5);
    }

    #[test]
    fn grid_regularity_alone_clusters_when_timing_is_jittered() {
        let m = meta();
        // On-grid frequencies but timing jitter well past timing_tol.
        const HOPS: [f64; 4] = [-300_000.0, -100_000.0, 100_000.0, 300_000.0];
        let borns = [0u64, 9, 45, 60, 130, 150, 300, 420];
        let tracks: Vec<Track> = borns
            .iter()
            .enumerate()
            .map(|(k, &b)| burst(k as u64 + 1, b, 5, HOPS[k % 4]))
            .collect();
        let emitters = cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default());
        assert_eq!(emitters.len(), 1, "the carrier grid is evidence enough");
        let e = &emitters[0];
        assert!(e.hopping);
        assert_eq!(e.carrier_spacing_hz, Some(200_000.0));
        assert!(
            e.hop_rate_hz.is_none(),
            "no rate asserted without regular timing"
        );
    }

    #[test]
    fn too_few_bursts_never_cluster() {
        let m = meta();
        let tracks = m0_hopper_bursts(3); // below min_bursts = 4
        assert!(cluster_hoppers(tracks.iter(), &m, &HopperClusterConfig::default()).is_empty());
    }
}
