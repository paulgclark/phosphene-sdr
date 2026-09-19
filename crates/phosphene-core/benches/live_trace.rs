// SPDX-License-Identifier: MIT

//! Live-trace cost seal (spec §7.4, lane M1-A R2).
//!
//! §7.4 requires the live EMA be computed incrementally per batch in closed
//! form so its trace-update cost is O(N) per **batch**, not per spectrum.
//! Two demonstrations:
//!
//! * `paths/*` — the closed-form batch call against the per-spectrum oracle
//!   over the same data.
//! * `batch_count/*` — a **fixed** volume of spectra partitioned into
//!   different batch counts. The per-sample pass over the data is identical
//!   in every case; what varies is only how often the trace is blended, so
//!   the runtime differences across this group are exactly the trace-update
//!   cost scaling with batch count rather than spectrum count.
//!
//! Run with `cargo bench -p phosphene-core`.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

use phosphene_core::LiveTrace;

const BINS: usize = 1024;
/// N = 1024 at 20 MS/s — the NFR-P1/D-012 operating point.
const SPECTRUM_INTERVAL: f32 = 1024.0 / 20.0e6;
const TAU_LIVE: f32 = 0.1;
/// Total spectra folded per measured iteration (~13 ms of signal at the
/// operating point).
const TOTAL_SPECTRA: usize = 256;

/// Deterministic synthetic dBFS batch, `TOTAL_SPECTRA × BINS` contiguous.
fn spectra() -> Vec<f32> {
    let mut state: u64 = 0xBEEF_CAFE;
    (0..TOTAL_SPECTRA * BINS)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (-90.0 + 80.0 * ((z >> 11) as f64 / (1u64 << 53) as f64)) as f32
        })
        .collect()
}

fn seeded_trace() -> LiveTrace {
    let mut lt = LiveTrace::new(BINS, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    lt.accumulate(&vec![-60.0f32; BINS]);
    lt
}

fn bench_paths(c: &mut Criterion) {
    let data = spectra();
    let mut group = c.benchmark_group("paths");
    group.throughput(Throughput::Elements(TOTAL_SPECTRA as u64));

    group.bench_function("closed_form_batch", |b| {
        b.iter_batched_ref(
            seeded_trace,
            |lt| lt.accumulate_batch(&data),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("per_spectrum_oracle", |b| {
        b.iter_batched_ref(
            seeded_trace,
            |lt| {
                for spectrum in data.as_chunks::<BINS>().0 {
                    lt.accumulate(spectrum);
                }
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_batch_count(c: &mut Criterion) {
    let data = spectra();
    let mut group = c.benchmark_group("batch_count");
    group.throughput(Throughput::Elements(TOTAL_SPECTRA as u64));

    // Same TOTAL_SPECTRA every time; only the partitioning differs. Batch
    // count 256 degenerates to one trace blend per spectrum (the §7.4
    // anti-pattern); batch count 1 blends once.
    for spectra_per_batch in [1usize, 4, 16, 64, 256] {
        let batches = TOTAL_SPECTRA / spectra_per_batch;
        group.bench_with_input(
            BenchmarkId::new("batches", batches),
            &spectra_per_batch,
            |b, &per_batch| {
                b.iter_batched_ref(
                    seeded_trace,
                    |lt| {
                        for batch in data.chunks_exact(per_batch * BINS) {
                            lt.accumulate_batch(batch);
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_paths, bench_batch_count);
criterion_main!(benches);
