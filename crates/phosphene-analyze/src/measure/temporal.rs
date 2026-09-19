// SPDX-License-Identifier: MIT

//! Temporal descriptors (FR-AM4, spec §7.5): duty cycle, mean burst
//! duration, and repetition interval — all from the track's on/off
//! transitions over the observation window. The dominant period comes from
//! the autocorrelation of the on/off time series, exactly as §7.5
//! prescribes; when no period clears the evidence bar the signal is reported
//! aperiodic (`period_s: None`), and a full-duty signal is reported
//! continuous — never a fabricated PRI.

use super::psd::count_factor;
use super::{Measured, Temporal};

/// One run of consecutive on-frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Run {
    /// Index of the first on-frame.
    pub start: usize,
    /// Number of consecutive on-frames.
    pub len: usize,
}

/// The temporal analysis handed back to the engine: the FR-AM4 bundle plus
/// the run structure the behavior flags reuse.
pub(super) struct TemporalAnalysis {
    /// The FR-AM4 descriptors.
    pub temporal: Temporal,
    /// On-runs, in time order.
    pub runs: Vec<Run>,
    /// True when the signal was present for (effectively) the whole window.
    pub continuous: bool,
}

/// Analyze the on/off series. `frame_dt_s` is the spacing of the series.
pub(super) fn analyze(
    on: &[bool],
    frame_dt_s: f64,
    continuous_duty: f32,
    min_pri_correlation: f32,
) -> TemporalAnalysis {
    let n = on.len();
    let n_on = on.iter().filter(|&&b| b).count();
    let runs = collect_runs(on);

    let duty = n_on as f32 / n.max(1) as f32;
    let duty_conf = count_factor(n, 16.0) as f32;
    let continuous = duty >= continuous_duty;

    let mean_burst_s = if continuous {
        None
    } else {
        mean_burst(&runs, n, frame_dt_s)
    };

    let period_s = if continuous {
        None
    } else {
        dominant_period(on, &runs, frame_dt_s, min_pri_correlation)
    };

    TemporalAnalysis {
        temporal: Temporal {
            duty: Measured::new(duty, duty_conf),
            mean_burst_s,
            period_s,
        },
        runs,
        continuous,
    }
}

fn collect_runs(on: &[bool]) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut current: Option<Run> = None;
    for (i, &b) in on.iter().enumerate() {
        match (&mut current, b) {
            (None, true) => current = Some(Run { start: i, len: 1 }),
            (Some(r), true) => r.len += 1,
            (Some(r), false) => {
                runs.push(*r);
                current = None;
            }
            (None, false) => {}
        }
    }
    if let Some(r) = current {
        runs.push(r);
    }
    runs
}

/// Mean on-duration. Runs truncated by the window edges bias the mean low,
/// so interior runs (fully inside the window) are preferred; when only
/// edge-touching runs exist they are used at half confidence.
fn mean_burst(runs: &[Run], n: usize, frame_dt_s: f64) -> Option<Measured<f64>> {
    if runs.is_empty() {
        return None;
    }
    let interior: Vec<&Run> = runs
        .iter()
        .filter(|r| r.start > 0 && r.start + r.len < n)
        .collect();
    let (used, edge_penalty): (Vec<&Run>, f32) = if interior.is_empty() {
        (runs.iter().collect(), 0.5)
    } else {
        (interior, 1.0)
    };
    let mean_frames = used.iter().map(|r| r.len as f64).sum::<f64>() / used.len() as f64;
    let conf = count_factor(used.len(), 2.0) as f32 * edge_penalty;
    Some(Measured::new(mean_frames * frame_dt_s, conf))
}

/// Dominant repetition interval via autocorrelation of the zero-mean on/off
/// series (§7.5). To avoid reporting a harmonic, the smallest local maximum
/// within 80% of the global maximum is taken as the fundamental. Asserted
/// only when at least two full periods fit the window, the series actually
/// transitions (≥2 runs and ≥1 off-frame), and the normalised correlation
/// clears `min_correlation`.
fn dominant_period(
    on: &[bool],
    runs: &[Run],
    frame_dt_s: f64,
    min_correlation: f32,
) -> Option<Measured<f64>> {
    let n = on.len();
    let n_on = on.iter().filter(|&&b| b).count();
    if runs.len() < 2 || n_on == 0 || n_on == n {
        return None;
    }

    let mean = n_on as f64 / n as f64;
    let x: Vec<f64> = on.iter().map(|&b| f64::from(u8::from(b)) - mean).collect();
    let denom: f64 = x.iter().map(|v| v * v).sum();
    if denom <= 0.0 {
        return None;
    }

    // Normalised autocorrelation, compensated for the shrinking overlap so a
    // perfectly periodic series scores ~1 at its period regardless of lag.
    // Lags are capped at n/2, keeping the compensation factor ≤ 2 and
    // guaranteeing at least two full periods of evidence.
    let max_lag = n / 2;
    if max_lag < 1 {
        return None;
    }
    let r: Vec<f64> = (1..=max_lag)
        .map(|lag| {
            let raw: f64 = (0..n - lag).map(|t| x[t] * x[t + lag]).sum();
            (raw / denom) * (n as f64 / (n - lag) as f64)
        })
        .collect();

    // Small lags are trivially correlated — consecutive frames of the same
    // burst — and are not a repetition interval. Search only past the
    // autocorrelation's first non-positive dip (a period must span at least
    // one whole on/off transition); a series that never dips has no
    // repetition structure to report.
    let search_from = r.iter().position(|&v| v <= 0.0)?;
    let global_max = r[search_from..].iter().copied().fold(f64::MIN, f64::max);
    if global_max < f64::from(min_correlation) {
        return None;
    }
    // Smallest local maximum within 80% of the global maximum — the
    // fundamental rather than one of its multiples.
    let lag = (search_from..r.len())
        .find(|&i| {
            let left = if i == 0 { f64::MIN } else { r[i - 1] };
            let right = if i + 1 < r.len() { r[i + 1] } else { f64::MIN };
            r[i] >= left && r[i] >= right && r[i] >= 0.8 * global_max
        })
        .map(|i| i + 1)?;

    let periods = n as f64 / lag as f64;
    let periods_factor = ((periods - 1.0) / (periods + 1.0)).clamp(0.0, 1.0);
    let conf = (r[lag - 1].clamp(0.0, 1.0) * periods_factor) as f32;
    Some(Measured::new(lag as f64 * frame_dt_s, conf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(pattern: &[(bool, usize)]) -> Vec<bool> {
        pattern
            .iter()
            .flat_map(|&(b, n)| std::iter::repeat_n(b, n))
            .collect()
    }

    #[test]
    fn continuous_signal_reports_continuous() {
        let on = vec![true; 64];
        let a = analyze(&on, 0.001, 0.98, 0.5);
        assert!(a.continuous);
        assert!((a.temporal.duty.value - 1.0).abs() < 1e-6);
        assert!(a.temporal.mean_burst_s.is_none());
        assert!(a.temporal.period_s.is_none(), "no fabricated PRI");
    }

    #[test]
    fn periodic_burst_train_recovers_duty_burst_and_pri() {
        // 5 on / 10 off, repeating: duty 1/3, burst 5 ms, PRI 15 ms.
        let on = series(&[
            (true, 5),
            (false, 10),
            (true, 5),
            (false, 10),
            (true, 5),
            (false, 10),
            (true, 5),
            (false, 10),
        ]);
        let a = analyze(&on, 0.001, 0.98, 0.5);
        assert!(!a.continuous);
        assert!((a.temporal.duty.value - 1.0 / 3.0).abs() < 0.02);
        let burst = a.temporal.mean_burst_s.unwrap();
        assert!((burst.value - 0.005).abs() < 1e-9, "burst {}", burst.value);
        assert!(burst.confidence > 0.0);
        let pri = a.temporal.period_s.unwrap();
        assert!((pri.value - 0.015).abs() < 1e-9, "pri {}", pri.value);
        assert!(pri.confidence > 0.0);
    }

    #[test]
    fn aperiodic_bursts_get_no_pri_but_a_burst_length() {
        // Irregular gaps (all pairwise burst-start spacings distinct):
        // bursty, but no repetition interval to assert.
        let on = series(&[
            (false, 2),
            (true, 4),
            (false, 3),
            (true, 4),
            (false, 12),
            (true, 4),
            (false, 9),
            (true, 4),
            (false, 18),
        ]);
        let a = analyze(&on, 0.001, 0.98, 0.5);
        assert!(a.temporal.mean_burst_s.is_some());
        assert!(
            a.temporal.period_s.is_none(),
            "aperiodic series must not be given a PRI"
        );
    }

    #[test]
    fn single_burst_has_no_period() {
        let on = series(&[(false, 10), (true, 6), (false, 48)]);
        let a = analyze(&on, 0.001, 0.98, 0.5);
        assert!(a.temporal.period_s.is_none());
        let burst = a.temporal.mean_burst_s.unwrap();
        assert!((burst.value - 0.006).abs() < 1e-9);
    }

    #[test]
    fn edge_truncated_runs_fall_back_at_reduced_confidence() {
        // Both runs touch a window edge — usable, but flagged by confidence.
        let on = series(&[(true, 4), (false, 8), (true, 4)]);
        let a = analyze(&on, 0.001, 0.98, 0.5);
        let burst = a.temporal.mean_burst_s.unwrap();
        assert!((burst.value - 0.004).abs() < 1e-9);
        assert!(burst.confidence <= 0.5);
    }
}
