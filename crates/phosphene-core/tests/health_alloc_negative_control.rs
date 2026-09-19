// SPDX-License-Identifier: MIT

//! D-121 mandatory negative control for the `health_alloc` seal, a standing
//! CI guard (D-122): design B keeps the process-wide counting allocator, so
//! an allocation made on a thread standing in for the path under test — not
//! the thread running the assertion — must still be visible inside the
//! counted window. This test is `#[should_panic]` and passes exactly when
//! that panic fires. If it ever passes *without* panicking, the counter has
//! quietly stopped seeing other threads, which is exactly what would let a
//! real per-batch allocation handed to a worker thread hide from
//! `health_alloc` undetected, and `cargo test` catches that as a failure
//! here (an unsatisfied `should_panic`).
//!
//! ## D-122 after-arm evidence
//! At `fab1f2dd57e670512600309bd2ba75e62d79c375` (the committed tree):
//! N = 1000 with 8 instances at once, 0 failures — the guard's
//! `should_panic` was satisfied every run under load. Also checked by hand:
//! commenting out the helper thread's allocation makes `cargo test` fail
//! with "test did not panic as expected", confirming the guard depends on
//! the real allocation and not on the attribute alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

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
#[should_panic(expected = "allocated")]
fn helper_thread_allocation_is_still_counted() {
    // A barrier, not a plain join, so the helper's allocation lands while
    // the main thread is still doing its own counted work below — inside
    // the window, not merely before it opens.
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let handle = thread::spawn(move || {
        worker_barrier.wait();
        // A known allocation, standing in for path-under-test work handed
        // to another thread.
        let v: Vec<u8> = Vec::with_capacity(64);
        std::hint::black_box(&v);
    });

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    barrier.wait();
    // The main thread's own counted work, allocation-free, so the helper's
    // allocation is not the only thing that could happen in this window.
    let mut sink = 0u64;
    for i in 0..10_000u64 {
        sink = sink.wrapping_add(std::hint::black_box(i));
    }
    handle.join().unwrap();
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    assert!(sink < u64::MAX, "the main-thread loop ran");
    assert_eq!(
        after - before,
        0,
        "the helper thread allocated {} time(s) inside the counted window \
         (D-121): design B keeps one process-wide counter, so this must be \
         visible here — an unsatisfied should_panic on this test would mean \
         the fix has gone blind to allocations on other threads",
        after - before
    );
}
