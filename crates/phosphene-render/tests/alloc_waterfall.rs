// SPDX-License-Identifier: MIT

//! M2-C extension of the §7.7 seal: the full §7.6 waterfall feed path as
//! D-051 rebuilt it — the per-spectrum `RowAggregator::push_spectrum` fold,
//! the counted-drop `advance` gap, the pending [`RowRing`] the compute
//! thread pushes into and the display drains per frame, plus the
//! keypress-rate `set_mode` / `set_cadence` reconfigurations (both speed
//! modes, both time bases) — allocates **nothing** once warm. The batched
//! GPU row upload that follows in production is wgpu's, outside §7.7's
//! scope as this repo applies it.
//!
//! This test used to share a binary and a `SERIAL` lock with the trace-path
//! test in `alloc.rs`. Under load (8 instances at once), the test harness's
//! own thread reporting one test's result could allocate inside the other
//! test's counted window — an intermittent failure with nothing wrong in
//! the code under test (D-121: 1 failure in 1000 runs at N=1000/8, matching
//! the CI signature of a few allocations over 232 frames). This file now
//! holds only this one test, so there is no sibling result to race against:
//! design B (structural isolation) keeps the process-wide counter — every
//! thread is still counted — and removes the interference by giving each
//! counted test its own test binary.
//!
//! ## D-121 / D-122 arm evidence (this binary, `alloc` before the split)
//! * **Base arm**, at `e173757a13dd6e30c12fd666b4ab2fed5da9581c` (the shared
//!   `alloc.rs`, two tests behind one `SERIAL` lock): built once with
//!   `cargo test -p phosphene-render --test alloc --no-run`, then run
//!   N = 1000 with 8 instances at once. 1 failure in 1000, in this test:
//!   `assertion left == right failed: the waterfall feed path allocated 1
//!   times over 232 frames (§7.7)`. Reverting the split later and rerunning
//!   the same N/load reproduced the same failure (1 in 1000, identical
//!   signature), confirming the harness-interference mechanism.
//! * **After arm**, at `fab1f2dd57e670512600309bd2ba75e62d79c375` (the
//!   committed tree — this file's `#[should_panic]` sibling change is
//!   test-only and does not touch this binary): N = 1000 with 8 instances
//!   at once, 0 failures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use phosphene_render::{derive_cadence, RowAggregation, RowAggregator, RowRing, WaterfallMode};

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
fn waterfall_feed_path_allocates_nothing_per_frame_once_warm() {
    let n = 1024usize;
    let si = 5e-4f32;
    let cadence = |mode, interval| derive_cadence(mode, Some(si), 30.0, interval, 1024.0, 640);
    let mut agg = RowAggregator::new(n, RowAggregation::Max, cadence(WaterfallMode::Fast, 1e-3));
    let mut ring = RowRing::new(n, 256);
    // The display-side drain buffer, pre-reserved once like the frame loops
    // hold it.
    let mut drained: Vec<f32> = Vec::with_capacity(n * 256);
    let mut sink = 0.0f32;
    let mut spectrum = vec![-90.0f32; n];

    // Warm-up: fold a few frames, emit rows, drain once.
    for frame in 0..8u32 {
        mutate_spectrum(&mut spectrum, frame);
        for _ in 0..8 {
            agg.push_spectrum(&spectrum, si, |row| ring.push_row(row));
        }
        drained.clear();
        ring.drain_into(&mut drained);
    }

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for frame in 8..240u32 {
        mutate_spectrum(&mut spectrum, frame);
        // Keypress-rate reconfiguration inside the counted window: the A
        // key's toggle, the F key's mode switch across both time bases, and
        // the -/= cadence changes reset the open interval but must not
        // allocate either.
        match frame % 60 {
            13 => agg.set_mode(RowAggregation::Mean),
            21 => agg.set_cadence(cadence(WaterfallMode::Traditional, 1e-3)),
            37 => agg.set_mode(RowAggregation::Max),
            44 => agg.set_cadence(derive_cadence(
                WaterfallMode::Fast,
                None,
                30.0,
                1e-3,
                1024.0,
                640,
            )),
            51 => agg.set_cadence(cadence(WaterfallMode::Fast, (frame % 7) as f32 * 1e-4)),
            _ => {}
        }
        // A frame's worth of compute-side folds, a counted-drop gap now and
        // then, then the display drain — the production shape.
        for _ in 0..8 {
            agg.push_spectrum(&spectrum, si, |row| ring.push_row(row));
        }
        if frame % 30 == 5 {
            agg.advance(4.0 * si, |row| ring.push_row(row));
        }
        drained.clear();
        ring.drain_into(&mut drained);
        if let Some(&v) = drained.first() {
            sink += v;
        }
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "the waterfall feed path allocated {} times over 232 frames (§7.7)",
        after - before
    );
    assert!(sink.is_finite(), "rows were emitted and read");
}

/// Vary the waterfall input in place each frame — no allocation.
fn mutate_spectrum(spectrum: &mut [f32], frame: u32) {
    for (i, v) in spectrum.iter_mut().enumerate() {
        *v = -90.0 + 15.0 * (((i as u32 + 5 * frame) % 89) as f32 / 89.0);
    }
    let n = spectrum.len();
    spectrum[(41 * frame as usize + 17) % n] = -25.0;
}
