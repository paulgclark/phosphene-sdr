// SPDX-License-Identifier: MIT

//! D-047's measurement: the per-row waterfall upload cost, batched versus
//! row-at-a-time. The number decides whether the strobed waterfall is a
//! performance feature, a legibility feature, or unnecessary — so it is
//! **measured and reported**, never asserted (runner GPUs vary wildly, and
//! a timing assertion would only measure the runner: D-021).
//!
//! Run manually: `cargo test -p phosphene-render --test wf_upload_cost --release -- --ignored --nocapture`

use std::time::Instant;

use phosphene_render::{FrameComposer, RowCadence, OFFSCREEN_FORMAT};

fn gpu() -> (wgpu::Device, wgpu::Queue, String) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no GPU adapter — the D-047 measurement cannot run");
    let info = adapter.get_info();
    let label = format!("{} ({:?})", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("phosphene-wf-upload-cost"),
        ..Default::default()
    }))
    .expect("GPU device request failed");
    (device, queue, label)
}

/// Time `frames` iterations of pushing `rows_per_frame` rows — either as
/// one batched call (the D-047 shape production uses) or as one call per
/// row (the shape it replaced) — each iteration submitted and waited, so
/// the staged copies actually execute. Returns µs per row.
const CADENCE: Option<RowCadence> = None;

fn measure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    composer: &mut FrameComposer,
    bins: usize,
    rows_per_frame: usize,
    frames: usize,
    batched: bool,
) -> f64 {
    let batch: Vec<f32> = (0..rows_per_frame * bins)
        .map(|i| -90.0 + (i % 97) as f32 * 0.1)
        .collect();
    // Warm-up: create the ring texture and settle driver state.
    composer.push_waterfall_rows(device, queue, &batch, bins, CADENCE);
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let start = Instant::now();
    for _ in 0..frames {
        if batched {
            composer.push_waterfall_rows(device, queue, &batch, bins, CADENCE);
        } else {
            for row in batch.chunks_exact(bins) {
                composer.push_waterfall_row(device, queue, row, Some(1e-3));
            }
        }
        queue.submit([]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    start.elapsed().as_secs_f64() * 1e6 / (frames * rows_per_frame) as f64
}

/// Report per-row upload cost across the shapes D-047 cares about: the fast
/// default (~17 rows/frame at 1 ms rows and 60 fps), a strobe-regime burst,
/// and single rows, at two FFT sizes.
#[test]
#[ignore = "measurement, not a gate — run with --ignored --nocapture (D-047/D-021)"]
fn measure_per_row_upload_cost() {
    let (device, queue, label) = gpu();
    println!("adapter: {label}");
    println!("bins  rows/frame  mode      us/row");
    for &bins in &[1024usize, 8192] {
        let mut composer = FrameComposer::new(&device, OFFSCREEN_FORMAT);
        for &(rows, frames) in &[(1usize, 600usize), (17, 600), (128, 200)] {
            for batched in [false, true] {
                let us = measure(&device, &queue, &mut composer, bins, rows, frames, batched);
                println!(
                    "{bins:<5} {rows:<11} {:<9} {us:.2}",
                    if batched { "batched" } else { "per-row" },
                );
            }
        }
    }
}
