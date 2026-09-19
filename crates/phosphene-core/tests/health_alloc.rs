// SPDX-License-Identifier: MIT

//! §7.7 seal: the per-batch health hot path allocates nothing.
//!
//! A counting global allocator wraps the system allocator; the test warms the
//! [`Batcher`] / [`PipelineHealth`] paths once, then asserts that a sustained
//! run of pushes, drop/completion records and snapshots performs **zero**
//! allocations. This is an integration test on purpose: the library crate
//! forbids `unsafe`, but a `GlobalAlloc` impl needs it, and here it wraps the
//! system allocator only.
//!
//! D-121 (design B, structural isolation): this file holds exactly one
//! test, so it was never exposed to the sibling-test interference that hit
//! `phosphene-render`'s allocation seal (a harness thread reporting one
//! test's result allocating inside another counted window) — there is no
//! sibling here to race against. It still shares the same process-wide
//! counter, so the mandatory negative control lives beside it in
//! `health_alloc_negative_control.rs`.
//!
//! ## D-122 after-arm evidence
//! No base arm applies here (no sibling test, so no interference to
//! measure). Loaded anyway for parity at
//! `fab1f2dd57e670512600309bd2ba75e62d79c375` (the committed tree):
//! N = 1000 with 8 instances at once, 0 failures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use phosphene_core::{Batcher, Complex, PipelineHealth};

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: defers entirely to the system allocator; the counter is a relaxed
// atomic with no bearing on allocation correctness.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn health_hot_path_allocates_nothing_per_batch() {
    let n = 1024;
    // Everything preallocated up front, exactly as the app pipeline does.
    let mut batcher = Batcher::new(n).unwrap();
    let mut health = PipelineHealth::new(n, 1.0).unwrap();
    let mut queue = vec![Complex::new(0.0f32, 0.0); 4 * n];
    let mut queued = 0usize;
    let block: Vec<Complex<f32>> = (0..3 * n + 7)
        .map(|i| Complex::new(i as f32, -(i as f32)))
        .collect();

    let mut run = |rounds: usize, t0: f64| {
        let mut t = t0;
        for _ in 0..rounds {
            // A slow consumer: room for 4 batches, drained one per round.
            let outcome = batcher.push(&block, |batch| {
                if queued + n <= queue.len() {
                    queue[queued..queued + n].copy_from_slice(batch);
                    queued += n;
                    true
                } else {
                    false
                }
            });
            if outcome.dropped > 0 {
                health.record_dropped(t, outcome.dropped);
            }
            if queued >= n {
                queued -= n;
                health.record_completed(t, 1);
            }
            let snap = health.snapshot(t);
            assert!(snap.processed_pct <= 100.0);
            t += 0.0005;
        }
    };

    // Warm-up: first touches (lazy queue growth would show up here).
    run(8, 0.0);

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    run(512, 1.0);
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "the per-batch hot path allocated {} times (§7.7 forbids any)",
        after - before
    );
}
