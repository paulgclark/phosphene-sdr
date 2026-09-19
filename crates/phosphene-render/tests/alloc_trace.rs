// SPDX-License-Identifier: MIT

//! §7.7 seal for the per-frame trace path (D-035): once warm, building the
//! CPU geometry for a frame — grid, waterfall frame, max-hold trace, live
//! trace — allocates **nothing**, at full span and zoomed. This is the
//! counting-allocator proof D-035 requires for repaying M1-C's per-frame
//! `resample_view` allocation, now covering both traces.
//!
//! Scope: the CPU geometry this crate owns. The GPU submission that follows
//! (encoder creation, queue writes) allocates inside wgpu and is outside
//! §7.7's "no allocation on the hot path" contract as this repo applies it —
//! the preallocated-buffers discipline governs our own code.
//!
//! A counting global allocator wraps the system allocator, as in
//! `phosphene-core`'s `health_alloc` seal; an integration test on purpose,
//! since a `GlobalAlloc` impl needs `unsafe`.
//!
//! This test used to share a binary and a `SERIAL` lock with the waterfall
//! feed-path test in `alloc.rs`. Under load, the test harness's own thread
//! reporting one test's result could allocate inside the other test's
//! counted window, an intermittent failure with nothing wrong in the code
//! under test (D-121). This file now holds only this one test, so there is
//! no sibling result to race against: design B (structural isolation) keeps
//! the process-wide counter — every thread is still counted — and removes
//! the interference by giving each counted test its own test binary.
//!
//! ## D-121 / D-122 arm evidence (this binary, `alloc` before the split)
//! * **Base arm**, at `e173757a13dd6e30c12fd666b4ab2fed5da9581c` (the shared
//!   `alloc.rs`, two tests behind one `SERIAL` lock): built once with
//!   `cargo test -p phosphene-render --test alloc --no-run`, then run
//!   N = 1000 with 8 instances at once. 1 failure in 1000, in
//!   `waterfall_feed_path_allocates_nothing_per_frame_once_warm`:
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

use phosphene_render::scene::{
    build_frame, build_grid, build_trace, build_trace_styled, GridPx, TraceScratch, TraceStyle,
    Vertex,
};
use phosphene_render::{DisplayParams, Theme, ZoomSpan};

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

/// One frame's CPU geometry, exactly as `FrameComposer::render_rtsa_frame`
/// builds it: grid + waterfall frame, then the max-hold overlay, then the
/// live trace — through one reused vertex buffer and one reused scratch.
#[allow(clippy::too_many_arguments)]
fn build_frame_geometry(
    verts: &mut Vec<Vertex>,
    scratch: &mut TraceScratch,
    live: &[f32],
    max_hold: &[f32],
    hist: GridPx,
    wf: GridPx,
    size: [u32; 2],
    params: &DisplayParams,
    theme: &Theme,
) {
    verts.clear();
    build_grid(verts, hist, size, params, theme);
    build_frame(verts, wf, size, theme);
    build_trace_styled(
        verts,
        max_hold,
        hist,
        size,
        params,
        &TraceStyle::max_hold(theme),
        scratch,
    );
    build_trace(verts, live, hist, size, params, theme, scratch);
}

#[test]
fn trace_path_allocates_nothing_per_frame_once_warm() {
    let n = 1024usize;
    let size = [1280u32, 720u32];
    let hist = GridPx {
        min_x: 56.0,
        min_y: 40.0,
        max_x: 1264.0,
        max_y: 500.0,
    };
    let wf = GridPx {
        min_x: 56.0,
        min_y: 506.0,
        max_x: 1264.0,
        max_y: 676.0,
    };
    let theme = Theme::default();
    let mut params = DisplayParams::default();

    // Preallocated once, as the composer holds them across frames.
    let mut verts: Vec<Vertex> = Vec::new();
    let mut scratch = TraceScratch::default();
    let mut live = vec![-95.0f32; n];
    let mut max_hold = vec![-80.0f32; n];

    // Both resample branches (max-over-bins downsample at full span,
    // interpolation when zoomed past bin resolution) count as hot paths:
    // warm each, then assert.
    for zoom in [ZoomSpan::FULL, ZoomSpan::new(0.4, 0.6)] {
        params.zoom = zoom;
        // Warm-up: buffers grow to this width once.
        for frame in 0..2u32 {
            mutate_bins(&mut live, &mut max_hold, frame);
            build_frame_geometry(
                &mut verts,
                &mut scratch,
                &live,
                &max_hold,
                hist,
                wf,
                size,
                &params,
                &theme,
            );
        }
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for frame in 2..120u32 {
            mutate_bins(&mut live, &mut max_hold, frame);
            build_frame_geometry(
                &mut verts,
                &mut scratch,
                &live,
                &max_hold,
                hist,
                wf,
                size,
                &params,
                &theme,
            );
            assert!(!verts.is_empty(), "the frame built no geometry");
        }
        let after = ALLOCATIONS.load(Ordering::Relaxed);
        assert_eq!(
            after - before,
            0,
            "the per-frame trace path allocated {} times over 118 frames at zoom {zoom:?} (§7.7, D-035)",
            after - before
        );
    }
}

/// Change the spectra in place each frame — moving peaks, no allocation —
/// so the assertion covers real varying data, not a frozen frame.
fn mutate_bins(live: &mut [f32], max_hold: &mut [f32], frame: u32) {
    let n = live.len();
    for (i, v) in live.iter_mut().enumerate() {
        *v = -95.0 + 10.0 * (((i as u32 + 3 * frame) % 97) as f32 / 97.0);
    }
    let peak = (37 * frame as usize + 100) % n;
    live[peak] = -20.0;
    for (i, v) in max_hold.iter_mut().enumerate() {
        *v = -80.0 + 5.0 * (((i as u32 + frame) % 53) as f32 / 53.0);
    }
    max_hold[(peak + n / 3) % n] = -30.0;
}
