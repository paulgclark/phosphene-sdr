// SPDX-License-Identifier: MIT

//! The degradation contract and its accounting (NFR-P3, FR-D11, D-013).
//!
//! When input outpaces compute, phosphene sheds **whole batches** — a batch
//! being the block of input samples feeding one FFT at the current size
//! (D-013) — and it counts what it shed. Nothing is ever partially processed
//! and nothing is ever decimated within a batch; the display then reports the
//! honest numbers instead of quietly looking smooth (FR-D11). This module owns
//! both halves of that contract as pure, clock-free logic:
//!
//! * [`Batcher`] — assembles an arbitrary stream of sample blocks into whole
//!   N-sample batches and offers each **complete** batch to the caller's
//!   queue. Partial batches are staged, never delivered, so whole-batch
//!   granularity holds by construction. A refused batch is a dropped batch;
//!   the caller records it via [`PipelineHealth::record_dropped`].
//! * [`PipelineHealth`] — the D-013 accounting. Batches are recorded at
//!   **resolution time**: [`record_completed`](PipelineHealth::record_completed)
//!   when a batch's FFT has actually completed,
//!   [`record_dropped`](PipelineHealth::record_dropped) when a whole batch was
//!   shed at the producer. `% of samples processed` is
//!   `samples_in_completed_FFTs / samples_received` over a rolling window,
//!   where a sample counts as received once its batch resolves; the drop
//!   counter counts **batches**, cumulatively. Input rate and FFTs/s come from
//!   the same records, so the three figures can never disagree with each
//!   other. The `t_s` on each record is the instant the batch is
//!   **attributed** to — and D-028 requires one time base for the whole
//!   window: pass the batch's *receipt* time for completions as well as for
//!   drops (a completion may be *recorded* later, once the FFT has actually
//!   run, while still being attributed to its receipt instant). A record
//!   whose receipt has already **aged out of the retained window** — routine
//!   under backlog, since the ≥ 250 ms ring delays completions — increments
//!   the lifetime totals only and never a bucket (D-030): the rolling
//!   figures describe the window, and a batch received before the window
//!   began is not in it, while the D-013 cumulative counters still see it. Samples still in flight (staged in a [`Batcher`], queued in the
//!   ring) are not yet resolved and are not yet counted — which is what makes
//!   the percentage **exactly** 100 whenever nothing was dropped, and exactly
//!   the completed/offered ratio under overload.
//!
//! Like the rest of this crate, nothing here reads a clock: every record and
//! snapshot takes the caller's time in seconds (any monotonic origin). Fixed
//! inputs at fixed times give bit-identical snapshots, which is what lets the
//! D-013 seal drive an over-rate stream deterministically instead of racing
//! wall-clock on a CI runner. All state is preallocated at construction; the
//! per-batch paths allocate nothing (§7.7).

use std::fmt;

use rustfft::num_complex::Complex;

/// NFR-P3: the ring buffer between source and compute holds at least this
/// much stream time at the configured rate.
pub const RING_HEADROOM_S: f64 = 0.25;

/// Minimum ring-buffer capacity in samples for a source at `rate_hz`
/// (NFR-P3: ≥ [`RING_HEADROOM_S`] of stream time at the configured rate).
pub fn min_ring_samples(rate_hz: f64) -> Result<usize, HealthError> {
    if !rate_hz.is_finite() || rate_hz <= 0.0 {
        return Err(HealthError::InvalidRate(rate_hz));
    }
    Ok(((rate_hz * RING_HEADROOM_S).ceil() as usize).max(1))
}

/// Errors from configuring the health accounting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthError {
    /// The batch length (samples per FFT) must be non-zero.
    InvalidBatchLen(usize),
    /// The rolling-window length must be finite and positive seconds.
    InvalidWindow(f64),
    /// A sample rate must be finite and positive Hz.
    InvalidRate(f64),
}

impl fmt::Display for HealthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthError::InvalidBatchLen(n) => {
                write!(f, "invalid batch length {n}: must be non-zero")
            }
            HealthError::InvalidWindow(s) => {
                write!(f, "invalid rolling window {s} s: must be finite and > 0")
            }
            HealthError::InvalidRate(hz) => {
                write!(f, "invalid sample rate {hz} Hz: must be finite and > 0")
            }
        }
    }
}

impl std::error::Error for HealthError {}

/// What one call to [`Batcher::push`] did with the complete batches it
/// assembled. `accepted` batches went to the caller's queue and will be
/// recorded when their FFT completes; `dropped` batches were refused by the
/// queue and must be recorded via [`PipelineHealth::record_dropped`] — the
/// producer-side half of the NFR-P3 accounting (§8.2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PushOutcome {
    /// Whole batches the acceptor took.
    pub accepted: u64,
    /// Whole batches the acceptor refused — shed under the D-013 policy.
    pub dropped: u64,
}

/// Assembles sample blocks of arbitrary size into whole N-sample batches.
///
/// Sources deliver blocks whose sizes carry no meaning; FFTs consume exactly
/// N samples. This is the seam where the two meet, and where the NFR-P3
/// granularity is enforced: only **complete** batches are ever offered
/// onward, so a downstream refusal always sheds a whole batch and a partial
/// batch is never processed, dropped, or counted — it simply waits for the
/// rest of its samples. The staging buffer is preallocated; `push` allocates
/// nothing.
#[derive(Debug, Clone)]
pub struct Batcher {
    /// Staging for a partial batch; `fill` samples are valid.
    batch: Vec<Complex<f32>>,
    fill: usize,
}

impl Batcher {
    /// Build a batcher for `batch_len`-sample batches (the FFT size, D-013).
    pub fn new(batch_len: usize) -> Result<Self, HealthError> {
        if batch_len == 0 {
            return Err(HealthError::InvalidBatchLen(batch_len));
        }
        Ok(Batcher {
            batch: vec![Complex::new(0.0, 0.0); batch_len],
            fill: 0,
        })
    }

    /// Samples per batch (the FFT size N).
    pub fn batch_len(&self) -> usize {
        self.batch.len()
    }

    /// Samples currently staged toward the next batch (always `< batch_len`).
    /// At end of stream these never resolved into a batch and are reported,
    /// not processed.
    pub fn pending(&self) -> usize {
        self.fill
    }

    /// Feed one block of samples, offering each completed batch to
    /// `try_accept` (`true` = queued for compute, `false` = refused). A
    /// refusal drops that whole batch — it is gone, and the *next* batch
    /// starts from the following sample; record the returned
    /// [`PushOutcome::dropped`] with [`PipelineHealth::record_dropped`].
    ///
    /// `try_accept` always sees exactly `batch_len` samples. Batching is
    /// invariant to how the stream is chunked: any sequence of block sizes
    /// yields the same batches.
    pub fn push(
        &mut self,
        mut samples: &[Complex<f32>],
        mut try_accept: impl FnMut(&[Complex<f32>]) -> bool,
    ) -> PushOutcome {
        let n = self.batch.len();
        let mut outcome = PushOutcome::default();

        // Complete a staged partial batch first.
        if self.fill > 0 {
            let take = (n - self.fill).min(samples.len());
            self.batch[self.fill..self.fill + take].copy_from_slice(&samples[..take]);
            self.fill += take;
            samples = &samples[take..];
            if self.fill < n {
                return outcome;
            }
            self.fill = 0;
            offer(&self.batch, &mut try_accept, &mut outcome);
        }

        // Whole batches straight from the input, copy-free.
        while samples.len() >= n {
            offer(&samples[..n], &mut try_accept, &mut outcome);
            samples = &samples[n..];
        }

        // Stage the remainder for the next call.
        self.batch[..samples.len()].copy_from_slice(samples);
        self.fill = samples.len();
        outcome
    }
}

fn offer(
    batch: &[Complex<f32>],
    try_accept: &mut impl FnMut(&[Complex<f32>]) -> bool,
    outcome: &mut PushOutcome,
) {
    if try_accept(batch) {
        outcome.accepted += 1;
    } else {
        outcome.dropped += 1;
    }
}

/// One snapshot of the FR-D11 figures, all derived from the same records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthSnapshot {
    /// Measured input rate in samples/s over the rolling window: every sample
    /// whose batch resolved (completed **or** dropped), divided by the window
    /// span actually covered.
    pub input_rate_sps: f64,
    /// D-013: `samples_in_completed_FFTs / samples_received` over the rolling
    /// window, as a percentage. **Exactly** 100.0 whenever the window holds no
    /// drops (the NFR-P3 negative half); exactly the completed/offered batch
    /// ratio otherwise (the batch length cancels).
    pub processed_pct: f32,
    /// Completed FFTs per second over the same window and span.
    pub ffts_per_s: f64,
    /// Cumulative dropped **batches** (D-013: the drop counter counts
    /// batches, not samples). Never resets while the pipeline runs.
    pub dropped_batches: u64,
    /// Cumulative completed batches (one FFT each).
    pub batches_completed: u64,
}

/// Number of sub-buckets the rolling window is quantised into. Rates and the
/// percentage age out with a granularity of `window / BUCKET_COUNT`.
const BUCKET_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, Default)]
struct Bucket {
    /// Absolute index `floor(t / bucket_width)` this slot currently holds.
    index: u64,
    completed: u64,
    dropped: u64,
}

/// THE retention predicate (D-030): is the bucket stamped `index` inside the
/// window whose newest bucket is `newest`? [`PipelineHealth::snapshot`] sums
/// with it and [`PipelineHealth::slot_for`] admits records with it — one
/// predicate, shared, so the two sides can never drift apart.
fn in_window(index: u64, newest: u64) -> bool {
    index + BUCKET_COUNT as u64 > newest
}

/// The D-013 accounting: batch-resolution records in, FR-D11 figures out.
///
/// Batches are recorded when they **resolve** — completed (FFT done) or
/// dropped (shed whole at the producer). Each record accounts `batch_len`
/// received samples; completions also account them as processed. The rolling
/// window is a ring of [`BUCKET_COUNT`] time buckets, preallocated, so
/// records and snapshots are O(1)/O([`BUCKET_COUNT`]) with no allocation.
///
/// Time is caller-supplied seconds from any fixed origin, monotone
/// non-decreasing in intent; a slightly stale `t` (thread-interleaving jitter)
/// is clamped into the retained window rather than misfiled.
#[derive(Debug, Clone)]
pub struct PipelineHealth {
    batch_len: usize,
    window_s: f64,
    bucket_width_s: f64,
    buckets: [Bucket; BUCKET_COUNT],
    /// Newest absolute bucket index recorded; `None` before the first record.
    head: Option<u64>,
    /// Time of the first record ever, for early-life rate spans.
    first_t: f64,
    total_completed: u64,
    total_dropped: u64,
}

impl PipelineHealth {
    /// Build the accounting for `batch_len`-sample batches (the FFT size N)
    /// with a rolling window of `window_s` seconds. The window length only
    /// smooths the *displayed* rates and percentage; the D-013 definitions of
    /// the numbers do not depend on it.
    pub fn new(batch_len: usize, window_s: f64) -> Result<Self, HealthError> {
        if batch_len == 0 {
            return Err(HealthError::InvalidBatchLen(batch_len));
        }
        if !window_s.is_finite() || window_s <= 0.0 {
            return Err(HealthError::InvalidWindow(window_s));
        }
        Ok(PipelineHealth {
            batch_len,
            window_s,
            bucket_width_s: window_s / BUCKET_COUNT as f64,
            buckets: [Bucket::default(); BUCKET_COUNT],
            head: None,
            first_t: 0.0,
            total_completed: 0,
            total_dropped: 0,
        })
    }

    /// Record `batches` whose FFTs have completed, attributed to `t_s` —
    /// which per D-028 must be their **receipt** time, the same instant
    /// their shed siblings enter the denominator, never the (later)
    /// FFT-completion time. A receipt that has already aged out of the
    /// retained window updates the lifetime total only (D-030).
    ///
    /// # Panics
    ///
    /// Panics if `t_s` is not finite and ≥ 0 — a programmer error, like every
    /// Δt precondition in this crate.
    pub fn record_completed(&mut self, t_s: f64, batches: u64) {
        if let Some(slot) = self.slot_for(t_s) {
            self.buckets[slot].completed += batches;
        }
        self.total_completed += batches;
    }

    /// Record `batches` shed whole at the producer at time `t_s` (NFR-P3) —
    /// a dropped batch is received and shed in the same moment, so this is
    /// its receipt instant, the D-028 time base. An aged-out instant updates
    /// the lifetime total only (D-030, symmetric with completions).
    ///
    /// # Panics
    ///
    /// Panics if `t_s` is not finite and ≥ 0.
    pub fn record_dropped(&mut self, t_s: f64, batches: u64) {
        if let Some(slot) = self.slot_for(t_s) {
            self.buckets[slot].dropped += batches;
        }
        self.total_dropped += batches;
    }

    /// The bucket slot a record at `t_s` belongs to, or `None` when that
    /// instant has already aged out of the retained window (D-030): such a
    /// record updates the lifetime totals only and touches no bucket — a
    /// timestamp is never clamped into a bucket it does not belong to, which
    /// would count an aged-out batch *inside* the window it left.
    ///
    /// Aged-out is decided by [`in_window`], the same predicate `snapshot`
    /// sums buckets with, so the two can never disagree. This also removes
    /// the wrap-corruption hazard the old clamp existed for: an in-window
    /// index maps to a unique slot whose occupant is either itself or
    /// something strictly older (reset below), never newer.
    fn slot_for(&mut self, t_s: f64) -> Option<usize> {
        check_t(t_s);
        let idx = (t_s / self.bucket_width_s) as u64;
        let head = match self.head {
            None => {
                self.head = Some(idx);
                self.first_t = t_s;
                idx
            }
            Some(head) if idx > head => {
                self.head = Some(idx);
                idx
            }
            Some(head) => head,
        };
        if !in_window(idx, head) {
            return None;
        }
        let slot = (idx % BUCKET_COUNT as u64) as usize;
        if self.buckets[slot].index != idx {
            self.buckets[slot] = Bucket {
                index: idx,
                completed: 0,
                dropped: 0,
            };
        }
        Some(slot)
    }

    /// The FR-D11 figures as of time `t_s`.
    ///
    /// # Panics
    ///
    /// Panics if `t_s` is not finite and ≥ 0.
    pub fn snapshot(&self, t_s: f64) -> HealthSnapshot {
        check_t(t_s);
        if self.head.is_none() {
            return HealthSnapshot {
                input_rate_sps: 0.0,
                processed_pct: 100.0,
                ffts_per_s: 0.0,
                dropped_batches: 0,
                batches_completed: 0,
            };
        }
        let t_idx = (t_s / self.bucket_width_s) as u64;
        let (mut completed_w, mut dropped_w) = (0u64, 0u64);
        for b in &self.buckets {
            if in_window(b.index, t_idx) {
                completed_w += b.completed;
                dropped_w += b.dropped;
            }
        }
        // Rate span: what the window actually covers — from the first record
        // if it is younger than the window, floored at one bucket so a
        // snapshot taken an instant after the first record stays finite.
        let span = (t_s - self.first_t).clamp(self.bucket_width_s, self.window_s);
        let resolved = completed_w + dropped_w;
        let processed_pct = if dropped_w == 0 {
            // Exact by construction (the NFR-P3 negative half): no drops in
            // the window means 100%, not 99.999…%.
            100.0
        } else {
            (100.0 * completed_w as f64 / resolved as f64) as f32
        };
        HealthSnapshot {
            input_rate_sps: (resolved * self.batch_len as u64) as f64 / span,
            processed_pct,
            ffts_per_s: completed_w as f64 / span,
            dropped_batches: self.total_dropped,
            batches_completed: self.total_completed,
        }
    }
}

fn check_t(t_s: f64) {
    assert!(
        t_s.is_finite() && t_s >= 0.0,
        "time must be finite and ≥ 0 seconds, got {t_s}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(v: f32) -> Complex<f32> {
        Complex::new(v, -v)
    }

    fn stream(len: usize) -> Vec<Complex<f32>> {
        (0..len).map(|i| sample(i as f32)).collect()
    }

    #[test]
    fn config_is_validated() {
        assert_eq!(Batcher::new(0).err(), Some(HealthError::InvalidBatchLen(0)));
        assert_eq!(
            PipelineHealth::new(0, 1.0).err(),
            Some(HealthError::InvalidBatchLen(0))
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                PipelineHealth::new(64, bad),
                Err(HealthError::InvalidWindow(_))
            ));
            assert!(matches!(
                min_ring_samples(bad),
                Err(HealthError::InvalidRate(_))
            ));
        }
    }

    #[test]
    fn errors_name_the_offending_value() {
        assert!(HealthError::InvalidBatchLen(0).to_string().contains('0'));
        assert!(HealthError::InvalidWindow(-3.0).to_string().contains("-3"));
        assert!(HealthError::InvalidRate(f64::NAN)
            .to_string()
            .contains("NaN"));
    }

    #[test]
    fn ring_capacity_holds_250ms_at_the_configured_rate() {
        // NFR-P3 arithmetic, from the configured rate — not from a constant
        // that happens to match a default.
        for rate in [8_000.0, 2_048_000.0, 20e6, 61.44e6] {
            let cap = min_ring_samples(rate).unwrap();
            assert!(
                cap as f64 / rate >= RING_HEADROOM_S,
                "capacity {cap} holds only {} s at {rate} Hz",
                cap as f64 / rate
            );
        }
        assert_eq!(min_ring_samples(61.44e6).unwrap(), 15_360_000);
    }

    #[test]
    fn batcher_delivers_whole_batches_only_and_is_chunking_invariant() {
        let n = 64;
        // Push the same stream in pathological block sizes and in one call;
        // the delivered batches must be identical, all exactly n long.
        let data = stream(10 * n + 17);
        let mut split_batches: Vec<Vec<Complex<f32>>> = Vec::new();
        let mut b = Batcher::new(n).unwrap();
        let mut i = 0;
        for size in [1, 7, 13, n - 1, n, n + 1, 5 * n, 1000].iter().cycle() {
            if i >= data.len() {
                break;
            }
            let end = (i + size).min(data.len());
            b.push(&data[i..end], |batch| {
                assert_eq!(batch.len(), n, "a partial batch was delivered");
                split_batches.push(batch.to_vec());
                true
            });
            i = end;
        }
        assert_eq!(split_batches.len(), 10);
        assert_eq!(b.pending(), 17);

        let mut whole_batches: Vec<Vec<Complex<f32>>> = Vec::new();
        let mut b = Batcher::new(n).unwrap();
        let outcome = b.push(&data, |batch| {
            whole_batches.push(batch.to_vec());
            true
        });
        assert_eq!(
            outcome,
            PushOutcome {
                accepted: 10,
                dropped: 0
            }
        );
        assert_eq!(split_batches, whole_batches);
    }

    #[test]
    fn batcher_partial_tail_is_staged_not_dropped() {
        let n = 32;
        let mut b = Batcher::new(n).unwrap();
        let outcome = b.push(&stream(n / 2), |_| true);
        assert_eq!(outcome, PushOutcome::default());
        assert_eq!(b.pending(), n / 2);
    }

    /// The D-013 seal: a known over-rate stream against an injected slow
    /// consumer, driven deterministically — fixed sample count, fixed batch
    /// size, synthetic time — and the reported percentage must match the
    /// batches actually completed, exactly.
    #[test]
    fn d013_reported_percentage_matches_batches_actually_completed() {
        let n = 64;
        let capacity = 4; // queue slots, in batches — the injected bottleneck
        let mut queue: Vec<Vec<Complex<f32>>> = Vec::new();
        let mut batcher = Batcher::new(n).unwrap();
        let mut health = PipelineHealth::new(n, 1.0).unwrap();

        let total_batches = 100u64;
        let data = stream(total_batches as usize * n);
        let mut offered_drops = 0u64;
        let mut completed = 0u64;
        let mut t = 0.0f64;

        // Producer runs 5 batches ahead of the consumer per turn: the queue
        // (capacity 4) overflows and whole batches are shed.
        for chunk in data.chunks(5 * n) {
            let outcome = batcher.push(chunk, |batch| {
                if queue.len() < capacity {
                    queue.push(batch.to_vec());
                    true
                } else {
                    false
                }
            });
            offered_drops += outcome.dropped;
            if outcome.dropped > 0 {
                health.record_dropped(t, outcome.dropped);
            }
            // The slow consumer: one FFT's worth of work per producer turn.
            if let Some(_batch) = queue.pop() {
                completed += 1;
                health.record_completed(t, 1);
            }
            t += 0.001;
        }
        // Drain what the queue still holds.
        while queue.pop().is_some() {
            completed += 1;
            health.record_completed(t, 1);
        }

        assert!(
            offered_drops > 0,
            "the injected bottleneck never overflowed"
        );
        assert_eq!(completed + offered_drops, total_batches);

        let snap = health.snapshot(t);
        // The percentage is the completed/offered ratio, exactly (all
        // resolutions are inside the window here).
        let expected = 100.0 * completed as f64 / total_batches as f64;
        assert_eq!(snap.processed_pct, expected as f32);
        // The drop counter counts batches, not samples.
        assert_eq!(snap.dropped_batches, offered_drops);
        assert_eq!(snap.batches_completed, completed);
    }

    /// The contract's negative half (NFR-P3): no overload → the count is
    /// exactly zero and the percentage exactly 100.
    #[test]
    fn no_overload_is_exactly_zero_drops_and_exactly_100_percent() {
        let n = 128;
        let mut batcher = Batcher::new(n).unwrap();
        let mut health = PipelineHealth::new(n, 1.0).unwrap();
        let data = stream(50 * n);
        let mut t = 0.0;
        for chunk in data.chunks(3 * n + 11) {
            let outcome = batcher.push(chunk, |_| true);
            if outcome.dropped > 0 {
                health.record_dropped(t, outcome.dropped);
            }
            health.record_completed(t, outcome.accepted);
            t += 0.01;
        }
        let snap = health.snapshot(t);
        assert_eq!(snap.dropped_batches, 0);
        assert_eq!(snap.processed_pct, 100.0);
        assert_eq!(snap.batches_completed, 50);
    }

    /// Overload that ends leaves the window: the percentage recovers to
    /// exactly 100 while the cumulative drop counter honestly keeps its total.
    #[test]
    fn percentage_recovers_after_overload_but_the_counter_does_not_forget() {
        let mut health = PipelineHealth::new(64, 1.0).unwrap();
        health.record_dropped(0.1, 30);
        health.record_completed(0.1, 10);
        let during = health.snapshot(0.2);
        assert_eq!(during.processed_pct, 25.0);

        // Ten seconds later the drops have aged out of the rolling window.
        health.record_completed(10.2, 20);
        let after = health.snapshot(10.5);
        assert_eq!(after.processed_pct, 100.0);
        assert_eq!(after.dropped_batches, 30);
        assert_eq!(after.batches_completed, 30);
    }

    /// Input rate and FFTs/s come from the same records as the percentage —
    /// one accounting, not two estimates that can disagree.
    #[test]
    fn rates_derive_from_the_same_records() {
        let n = 100;
        let mut health = PipelineHealth::new(n, 1.0).unwrap();
        // 50 completed + 50 dropped, all at t = 0.5, snapshot at t = 1.0:
        // the observed span is the 0.5 s since the first record.
        health.record_completed(0.5, 50);
        health.record_dropped(0.5, 50);
        let snap = health.snapshot(1.0);
        assert_eq!(snap.input_rate_sps, 100.0 * n as f64 / 0.5);
        assert_eq!(snap.ffts_per_s, 50.0 / 0.5);
        assert_eq!(snap.processed_pct, 50.0);
    }

    #[test]
    fn snapshot_before_any_record_is_idle_and_honest() {
        let health = PipelineHealth::new(1024, 1.0).unwrap();
        let snap = health.snapshot(5.0);
        assert_eq!(snap.input_rate_sps, 0.0);
        assert_eq!(snap.ffts_per_s, 0.0);
        assert_eq!(snap.processed_pct, 100.0);
        assert_eq!(snap.dropped_batches, 0);
        assert_eq!(snap.batches_completed, 0);
    }

    #[test]
    fn slightly_stale_time_stays_in_its_own_bucket() {
        let mut health = PipelineHealth::new(64, 1.0).unwrap();
        health.record_completed(5.0, 1);
        // A record whose timestamp lost a thread race stays in the window.
        health.record_completed(4.999, 1);
        let snap = health.snapshot(5.0);
        assert_eq!(snap.batches_completed, 2);
        assert_eq!(snap.processed_pct, 100.0);
    }

    /// D-030: a record whose instant has aged out of the retained window
    /// increments the lifetime totals ONLY — it never lands in a bucket it
    /// does not belong to, where it would inflate the window figures. The
    /// old clamp made this exact shape read 50% where 0% was correct.
    #[test]
    fn aged_out_record_updates_totals_only_never_the_window() {
        let n = 64;
        let mut health = PipelineHealth::new(n, 1.0).unwrap();
        // Fresh evidence: 20 drops received now, at t = 10.
        health.record_dropped(10.0, 20);
        // A backlogged completion whose RECEIPT was at t = 1 — nine windows
        // ago. Cumulative counters must see it; the window must not.
        health.record_completed(1.0, 8);
        let snap = health.snapshot(10.01);
        assert_eq!(snap.batches_completed, 8, "lifetime total must keep it");
        assert_eq!(snap.dropped_batches, 20);
        // Window: 20 fresh drops, zero fresh completions → exactly 0%.
        assert_eq!(snap.processed_pct, 0.0);
        assert_eq!(snap.ffts_per_s, 0.0);
        // Symmetric for drops: an aged drop is not window evidence either.
        health.record_completed(10.02, 5);
        health.record_dropped(2.0, 30);
        let snap = health.snapshot(10.03);
        assert_eq!(snap.dropped_batches, 50, "lifetime total must keep it");
        // Window now: 20 drops + 5 completions → 20%, the aged 30 excluded.
        assert_eq!(snap.processed_pct, 20.0);
    }
}
