// SPDX-License-Identifier: MIT

//! DSP throughput benches — the M0-D re-baseline of product spec §3.2
//! (D-005, Appendix B) plus the FFT-size curve D-012 requires.
//!
//! Three benches, mirroring the §3.2 methodology:
//!
//! * `window_fft`  — windowed complex FFT in isolation, across the full
//!   FR-D10 size range 512–32768 (D-012's throughput curve).
//! * `mag_log`     — the |·|²+log10 stage in isolation (§3.2 row 2).
//! * `end_to_end`  — `SpectrumAnalyzer::process`, the whole §7.0 CPU path
//!   (window → FFT → mag² → dBFS → fftshift), across the same size range.
//!   NFR-P1's throughput guarantee is read off this bench at the default
//!   size 1024 (D-012).
//!
//! Throughput is reported in complex samples ("elements") per second so the
//! numbers land directly in the §3.2 table's MS/s units.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use phosphene_benches::{mag_log, tone_frame};
use phosphene_core::{Complex, SpectrumAnalyzer, Window, WindowKind};
use rustfft::FftPlanner;

/// The FR-D10 user-selectable FFT size range, endpoints included (D-012).
const FFT_SIZES: &[usize] = &[512, 1024, 2048, 4096, 8192, 16384, 32768];

fn bench_window_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_fft");
    for &n in FFT_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let window = Window::new(WindowKind::Hann, n);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
        let frame = tone_frame(n, 100.0);
        let mut buf = vec![Complex::new(0.0f32, 0.0); n];
        let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                for (out, (s, w)) in buf.iter_mut().zip(frame.iter().zip(window.coefficients())) {
                    *out = s * w;
                }
                fft.process_with_scratch(&mut buf, &mut scratch);
                black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_mag_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("mag_log");
    for &n in &[1024usize, 32768] {
        group.throughput(Throughput::Elements(n as u64));
        let spectrum = tone_frame(n, 100.0);
        let mut dbfs = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                mag_log(&spectrum, &mut dbfs);
                black_box(&dbfs);
            });
        });
    }
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    for &n in FFT_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let mut analyzer =
            SpectrumAnalyzer::new(n, WindowKind::Hann).expect("size is within FR-D10 range");
        let frame = tone_frame(n, 100.0);
        let mut dbfs = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                analyzer.process(&frame, &mut dbfs);
                black_box(&dbfs);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_window_fft, bench_mag_log, bench_end_to_end);
criterion_main!(benches);
