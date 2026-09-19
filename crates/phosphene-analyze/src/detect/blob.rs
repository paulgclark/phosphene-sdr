// SPDX-License-Identifier: MIT

//! Blob formation: connected components over the (freq, time) occupancy
//! mask (nextgen spec §7.3, FR-AD3).
//!
//! [`BlobBuilder`] labels the mask **streamingly**, one frame at a time, so
//! detection regions are emitted as soon as they end rather than after the
//! whole capture — the classic two-pass/union-find labeler reorganised
//! around a frame-by-frame scan. Occupied bins in each frame are grouped
//! into runs; runs are connected to the previous frame's runs under
//! **8-connectivity** (bin ranges overlapping or diagonally touching), so a
//! sweeping chirp that advances one bin per frame stays a single component.
//! Runs in the same frame separated by any unoccupied bin are distinct
//! (they may still join later through a shared successor, handled by
//! union-find with cross-frame merging).
//!
//! A component with no run in the newly pushed frame has ended: it is
//! finalized into a [`Detection`] and returned from that
//! [`push_frame`](BlobBuilder::push_frame) call. [`flush`](BlobBuilder::flush)
//! finalizes everything still open (end of capture, or a sequence gap — the
//! caller decides what a discontinuity means; [`crate::detect::Detector`]
//! flushes on one).
//!
//! Per component the builder accumulates the [`Detection`] statistics:
//! grid extents, the power-weighted centroid, peak per-bin power, peak SNR
//! versus the supplied §7.1 floor, and integrated power. *Integrated power*
//! is reported as the **mean per-frame in-component power** — the sum of all
//! occupied cells' linear power divided by the component's frame count — so
//! a long blob reads the same as a short one at equal strength.
//!
//! The union-find arena is **bounded by peak concurrency, not by
//! runtime** — phosphene is a long-running tool. A component's slot is
//! recycled the moment it dies: absorbed roots when their union happens,
//! finalized components when they are emitted (or dropped by the size
//! filter), and everything on [`flush`](BlobBuilder::flush), which resets
//! the arena entirely. This is safe because the only references into the
//! arena are the previous frame's run labels, and those are canonicalized
//! to live roots at the end of every push — dead slots are unreachable by
//! then. [`slots`](BlobBuilder::slots) exposes the arena size so tests can
//! hold the bound.

use crate::detect::{DetectError, Detection};

/// One maximal run of occupied bins in a frame, tagged with its component.
#[derive(Debug, Clone, Copy)]
struct Run {
    lo: usize,
    hi: usize,
    comp: usize,
}

/// Accumulated statistics for one in-progress component; lives at the
/// union-find root.
#[derive(Debug, Clone)]
struct CompStats {
    bin_lo: usize,
    bin_hi: usize,
    frame_lo: u64,
    frame_hi: u64,
    /// Σ linear power over all occupied cells.
    sum_lin: f64,
    /// Σ linear power × bin index (centroid numerator).
    sum_lin_bin: f64,
    peak_db: f32,
    peak_snr: f32,
    cells: usize,
}

/// Streaming connected-components labeler over per-frame occupancy masks.
/// See the [module docs](self).
#[derive(Debug, Clone)]
pub struct BlobBuilder {
    bins: usize,
    min_cells: usize,
    /// Union-find forest over the live components (recycled slots).
    parent: Vec<usize>,
    /// Statistics, authoritative at roots.
    stats: Vec<CompStats>,
    /// Recyclable arena slots (dead components).
    free: Vec<usize>,
    /// Slots that died during the current push (absorbed by a union or
    /// finalized); moved to `free` once the frame's run labels are
    /// canonicalized. Persistent only to recycle its allocation.
    dying: Vec<usize>,
    /// Runs of the previously pushed frame.
    prev: Vec<Run>,
    /// Scratch for the current frame's runs (recycled allocation).
    cur: Vec<Run>,
}

impl BlobBuilder {
    /// Build a labeler for frames of `bins` bins. Components smaller than
    /// `min_cells` occupied cells are dropped at finalization (a clutter
    /// filter; `1` keeps everything).
    pub fn new(bins: usize, min_cells: usize) -> Result<Self, DetectError> {
        if bins == 0 {
            return Err(DetectError::InvalidConfig(
                "blob formation needs at least one bin".to_string(),
            ));
        }
        if min_cells == 0 {
            return Err(DetectError::InvalidConfig(
                "minimum blob size must be at least one cell \
                 (use 1 to keep every component)"
                    .to_string(),
            ));
        }
        Ok(BlobBuilder {
            bins,
            min_cells,
            parent: Vec::new(),
            stats: Vec::new(),
            free: Vec::new(),
            dying: Vec::new(),
            prev: Vec::new(),
            cur: Vec::new(),
        })
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    /// Merge root `b` into root `a` (both must be roots, distinct). `b`'s
    /// slot is recycled once this frame's run labels are canonicalized.
    fn union(&mut self, a: usize, b: usize) {
        debug_assert!(self.parent[a] == a && self.parent[b] == b && a != b);
        self.parent[b] = a;
        self.dying.push(b);
        let sb = self.stats[b].clone();
        let sa = &mut self.stats[a];
        sa.bin_lo = sa.bin_lo.min(sb.bin_lo);
        sa.bin_hi = sa.bin_hi.max(sb.bin_hi);
        sa.frame_lo = sa.frame_lo.min(sb.frame_lo);
        sa.frame_hi = sa.frame_hi.max(sb.frame_hi);
        sa.sum_lin += sb.sum_lin;
        sa.sum_lin_bin += sb.sum_lin_bin;
        sa.peak_db = sa.peak_db.max(sb.peak_db);
        sa.peak_snr = sa.peak_snr.max(sb.peak_snr);
        sa.cells += sb.cells;
    }

    fn new_component(&mut self, seq: u64) -> usize {
        let fresh = CompStats {
            bin_lo: usize::MAX,
            bin_hi: 0,
            frame_lo: seq,
            frame_hi: seq,
            sum_lin: 0.0,
            sum_lin_bin: 0.0,
            peak_db: f32::NEG_INFINITY,
            peak_snr: f32::NEG_INFINITY,
            cells: 0,
        };
        match self.free.pop() {
            Some(id) => {
                self.parent[id] = id;
                self.stats[id] = fresh;
                id
            }
            None => {
                let id = self.parent.len();
                self.parent.push(id);
                self.stats.push(fresh);
                id
            }
        }
    }

    fn finalize(&mut self, root: usize) -> Option<Detection> {
        let s = &self.stats[root];
        if s.cells < self.min_cells {
            return None;
        }
        Some(Detection {
            bin_lo: s.bin_lo,
            bin_hi: s.bin_hi,
            frame_lo: s.frame_lo,
            frame_hi: s.frame_hi,
            centroid_bin: s.sum_lin_bin / s.sum_lin,
            peak_dbfs: s.peak_db,
            power_dbfs: (10.0 * (s.sum_lin / (s.frame_hi - s.frame_lo + 1) as f64).log10()) as f32,
            snr_db: s.peak_snr,
        })
    }

    /// Absorb one frame's occupancy mask (with the dB frame it came from and
    /// the current per-bin noise floor, both needed for the component
    /// statistics), returning every detection region that **ended** at this
    /// frame — oldest first, then lowest-frequency first.
    ///
    /// `seq` stamps the component's frame extent; frames are assumed
    /// consecutive. On a sequence discontinuity call
    /// [`flush`](Self::flush) first, or components will bridge the gap.
    ///
    /// # Panics
    ///
    /// Panics if `frame_db`, `mask` or `floor_db` differ in length from the
    /// constructed bin count.
    pub fn push_frame(
        &mut self,
        frame_db: &[f32],
        mask: &[bool],
        floor_db: &[f32],
        seq: u64,
    ) -> Vec<Detection> {
        assert_eq!(frame_db.len(), self.bins, "frame length must equal bins");
        assert_eq!(mask.len(), self.bins, "mask length must equal bins");
        assert_eq!(floor_db.len(), self.bins, "floor length must equal bins");

        let mut cur = std::mem::take(&mut self.cur);
        cur.clear();
        let mut k = 0;
        while k < self.bins {
            if mask[k] {
                let lo = k;
                while k < self.bins && mask[k] {
                    k += 1;
                }
                cur.push(Run {
                    lo,
                    hi: k - 1,
                    comp: usize::MAX,
                });
            } else {
                k += 1;
            }
        }

        let prev = std::mem::take(&mut self.prev);

        // Connect each current run to every 8-connected previous run, in a
        // single merged sweep (both run lists are sorted by construction).
        let mut p = 0;
        for run in cur.iter_mut() {
            while p < prev.len() && prev[p].hi + 1 < run.lo {
                p += 1; // strictly left of this and every later run
            }
            let mut q = p;
            while q < prev.len() && prev[q].lo <= run.hi + 1 {
                let root = self.find(prev[q].comp);
                if run.comp == usize::MAX {
                    run.comp = root;
                } else {
                    let mine = self.find(run.comp);
                    if mine != root {
                        self.union(mine, root);
                    }
                }
                q += 1;
            }
            if run.comp == usize::MAX {
                run.comp = self.new_component(seq);
            }
        }

        // Accumulate this frame's cells (after all of the frame's unions, so
        // every run resolves to its final root).
        for run in cur.iter() {
            let root = self.find(run.comp);
            let s = &mut self.stats[root];
            s.bin_lo = s.bin_lo.min(run.lo);
            s.bin_hi = s.bin_hi.max(run.hi);
            s.frame_hi = seq;
            for bin in run.lo..=run.hi {
                let db = frame_db[bin];
                let lin = 10f64.powf(db as f64 / 10.0);
                s.sum_lin += lin;
                s.sum_lin_bin += lin * bin as f64;
                s.cells += 1;
                s.peak_db = s.peak_db.max(db);
                s.peak_snr = s.peak_snr.max(db - floor_db[bin]);
            }
        }

        // Canonicalize this frame's run labels to their final roots — after
        // this, dead (absorbed) slots are unreachable and safe to recycle.
        for run in cur.iter_mut() {
            run.comp = self.find(run.comp);
        }

        // Components with a run in the previous frame but none now are done;
        // their slots are recycled along with the frame's absorbed ones.
        let mut out = Vec::new();
        let mut done: Vec<usize> = Vec::new();
        for run in prev.iter() {
            let root = self.find(run.comp);
            if !cur.iter().any(|r| r.comp == root) && !done.contains(&root) {
                done.push(root);
            }
        }
        for root in done {
            out.extend(self.finalize(root));
            self.free.push(root);
        }
        self.free.append(&mut self.dying);
        Self::sort_detections(&mut out);

        self.prev = cur;
        self.cur = prev; // recycle the allocation
        out
    }

    /// Finalize and return every still-open component — end of capture or a
    /// sequence discontinuity. Resets the component arena entirely (nothing
    /// is live afterwards); the builder is ready for a fresh frame.
    pub fn flush(&mut self) -> Vec<Detection> {
        let prev = std::mem::take(&mut self.prev);
        let mut out = Vec::new();
        let mut done: Vec<usize> = Vec::new();
        for run in prev.iter() {
            let root = self.find(run.comp);
            if !done.contains(&root) {
                done.push(root);
            }
        }
        for root in done {
            out.extend(self.finalize(root));
        }
        self.parent.clear();
        self.stats.clear();
        self.free.clear();
        self.dying.clear();
        Self::sort_detections(&mut out);
        self.cur = prev;
        out
    }

    fn sort_detections(out: &mut [Detection]) {
        out.sort_by(|a, b| {
            (a.frame_lo, a.bin_lo, a.frame_hi, a.bin_hi)
                .cmp(&(b.frame_lo, b.bin_lo, b.frame_hi, b.bin_hi))
        });
    }

    /// Bins per frame this labeler was built for.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Component slots currently allocated. Bounded by the peak number of
    /// simultaneously open regions (plus the handful dying within a frame),
    /// never by how long the builder has been running — the invariant the
    /// long-run growth test holds.
    pub fn slots(&self) -> usize {
        self.parent.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: [f32; 16] = [-40.0; 16];

    fn mask_of(bins: &[usize]) -> Vec<bool> {
        let mut m = vec![false; 16];
        for &b in bins {
            m[b] = true;
        }
        m
    }

    /// Frame with −10 dB in the given bins, −100 elsewhere.
    fn frame_of(bins: &[usize]) -> Vec<f32> {
        let mut f = vec![-100.0f32; 16];
        for &b in bins {
            f[b] = -10.0;
        }
        f
    }

    fn push(bb: &mut BlobBuilder, bins: &[usize], seq: u64) -> Vec<Detection> {
        bb.push_frame(&frame_of(bins), &mask_of(bins), &FLOOR, seq)
    }

    #[test]
    fn rejects_zero_bins_and_zero_min_cells() {
        assert!(matches!(
            BlobBuilder::new(0, 1),
            Err(DetectError::InvalidConfig(_))
        ));
        assert!(matches!(
            BlobBuilder::new(16, 0),
            Err(DetectError::InvalidConfig(_))
        ));
    }

    #[test]
    fn one_run_over_three_frames_makes_one_detection_with_exact_stats() {
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        for seq in 10..13 {
            assert!(push(&mut bb, &[4, 5], seq).is_empty());
        }
        let out = bb.flush();
        assert_eq!(out.len(), 1);
        let d = &out[0];
        assert_eq!((d.bin_lo, d.bin_hi), (4, 5));
        assert_eq!((d.frame_lo, d.frame_hi), (10, 12));
        // Six cells of linear 0.1 ⇒ mean per-frame power 0.2 ⇒ −6.99 dB;
        // centroid midway between the equal-power bins.
        assert!((d.centroid_bin - 4.5).abs() < 1e-9);
        assert!((d.power_dbfs - (10.0 * 0.2f32.log10())).abs() < 1e-4);
        assert_eq!(d.peak_dbfs, -10.0);
        assert_eq!(d.snr_db, 30.0);
    }

    #[test]
    fn a_component_is_emitted_the_frame_after_it_ends() {
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        assert!(push(&mut bb, &[8], 0).is_empty());
        let out = push(&mut bb, &[], 1);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].frame_lo, out[0].frame_hi), (0, 0));
        assert!(bb.flush().is_empty(), "already emitted — must not repeat");
    }

    #[test]
    fn disjoint_runs_are_separate_detections() {
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        // [2,3] and [7] are separated by unoccupied bins in every frame.
        push(&mut bb, &[2, 3, 7], 0);
        push(&mut bb, &[2, 3, 7], 1);
        let out = bb.flush();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].bin_lo, out[0].bin_hi), (2, 3));
        assert_eq!((out[1].bin_lo, out[1].bin_hi), (7, 7));
    }

    #[test]
    fn diagonal_touch_connects_but_a_two_bin_jump_does_not() {
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        let mut all = Vec::new();
        all.extend(push(&mut bb, &[3], 0));
        all.extend(push(&mut bb, &[4], 1)); // diagonal: connected
        all.extend(push(&mut bb, &[6], 2)); // two-bin jump: a new component
        all.extend(bb.flush());
        let mut spans: Vec<(usize, usize, u64, u64)> = all
            .iter()
            .map(|d| (d.bin_lo, d.bin_hi, d.frame_lo, d.frame_hi))
            .collect();
        spans.sort();
        assert_eq!(spans, vec![(3, 4, 0, 1), (6, 6, 2, 2)]);
    }

    #[test]
    fn a_bridging_run_unions_two_earlier_components() {
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        push(&mut bb, &[2, 3, 7, 8], 0); // two separate runs
        push(&mut bb, &[3, 4, 5, 6, 7], 1); // touches both ⇒ union
        let out = bb.flush();
        assert_eq!(out.len(), 1, "bridged components must merge: {out:?}");
        let d = &out[0];
        assert_eq!((d.bin_lo, d.bin_hi), (2, 8));
        assert_eq!((d.frame_lo, d.frame_hi), (0, 1));
        assert_eq!(d.snr_db, 30.0);
    }

    #[test]
    fn arena_is_bounded_over_a_long_run_and_reset_by_flush() {
        // Ten thousand frames, each closing the previous component and
        // opening a new one (alternating, unconnected bins) with a second
        // pair bridging into unions — the worst churn for slot recycling.
        // phosphene runs for hours: slots must track peak concurrency, not
        // component count.
        let mut bb = BlobBuilder::new(16, 1).unwrap();
        let mut emitted = 0usize;
        for seq in 0..10_000u64 {
            let bins: &[usize] = if seq % 2 == 0 { &[3, 12] } else { &[9] };
            emitted += push(&mut bb, bins, seq).len();
        }
        assert!(emitted > 9_000, "churn scene should emit constantly");
        assert!(
            bb.slots() <= 6,
            "arena grew with runtime: {} slots after 10k frames",
            bb.slots()
        );
        bb.flush();
        assert_eq!(bb.slots(), 0, "flush must reset the arena");
    }

    #[test]
    fn min_cells_filters_specks() {
        let mut bb = BlobBuilder::new(16, 3).unwrap();
        push(&mut bb, &[2], 0); // 1 cell — dropped
        push(&mut bb, &[9, 10], 1);
        push(&mut bb, &[9], 2); // 3 cells — kept
        let mut all = bb.flush();
        all.extend(push(&mut bb, &[], 3));
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].bin_lo, 9);
    }
}
