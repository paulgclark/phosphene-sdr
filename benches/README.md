<!-- SPDX-License-Identifier: MIT -->

# phosphene benches — the §3.2 re-baseline (D-005, Appendix B)

The feasibility numbers in product spec §3.2 were measured on a weak 2-vCPU
container and are **not to be quoted**. This directory holds the criterion
benches that re-measure them on real hardware, and the stored per-runner-class
baselines the CI regression gate (spec §10: >10% median regression fails)
compares against.

## Re-baselined numbers (2026-08-24)

Measured with `cargo run -p xtask -- bench-baseline`, single thread, `f32`,
Hann window, full-scale complex tone input.

### Hardware coverage (D-022)

Appendix B's machine list is satisfied **by hardware class, on hardware the
fleet actually possesses** (D-022). Every number below names the exact CPU it
was measured on; nothing is extrapolated or scaled to stand in for a machine
we do not have.

| Appendix B class | status |
|---|---|
| x86 desktop | ✅ `x86_64-linux-i9-11900k` (Intel Core i9-11900K, 8C/16T, 3.5 GHz base) |
| Apple Silicon | ✅ `aarch64-macos-m3-ultra` (Apple M3 Ultra, 28 cores, macOS) |
| Laptop class | ⛔ **deferred by name (D-022): no laptop-class hardware suitable for a clean benchmark exists in the fleet as of 2026-08-24** |
| Threadripper class | ⛔ **deferred by name (D-022): no Threadripper-class hardware exists in the fleet as of 2026-08-24** |

Single thread, median per iteration, throughput in complex MS/s. Each column
is exactly the CPU named in its header — nothing is averaged across machines:

| bench | i9-11900K | | M3 Ultra | |
|---|---|---|---|---|
| | median | MS/s | median | MS/s |
| `window_fft/512` | 0.36 µs | 1424 | 0.63 µs | 814 |
| `window_fft/1024` | 0.87 µs | 1174 | 1.46 µs | 699 |
| `window_fft/2048` | 1.89 µs | 1084 | 2.99 µs | 685 |
| `window_fft/4096` | 4.64 µs | 884 | 7.17 µs | 571 |
| `window_fft/8192` | 9.86 µs | 831 | 15.03 µs | 545 |
| `window_fft/16384` | 21.43 µs | 765 | 33.75 µs | 486 |
| `window_fft/32768` | 50.18 µs | 653 | 70.23 µs | 467 |
| `mag_log/1024` | 3.50 µs | 293 | 1.65 µs | 620 |
| `mag_log/32768` | 161.66 µs | 203 | 53.66 µs | 611 |
| `end_to_end/512` | 3.36 µs | 152 | 1.52 µs | 337 |
| `end_to_end/1024` | 6.90 µs | **148** | 3.24 µs | **316** |
| `end_to_end/2048` | 13.95 µs | 147 | 6.54 µs | 313 |
| `end_to_end/4096` | 28.72 µs | 143 | 14.17 µs | 289 |
| `end_to_end/8192` | 58.49 µs | 140 | 29.92 µs | 274 |
| `end_to_end/16384` | 118.20 µs | 139 | 62.59 µs | 262 |
| `end_to_end/32768` | 243.51 µs | 135 | 127.25 µs | 258 |

**Stated as measured, not smoothed over:** the i9-11900K is *faster* than the
M3 Ultra on the bare windowed-FFT stage at every size — e.g.
`window_fft/8192`: **9.86 µs vs 15.03 µs** — while the M3 Ultra wins `mag_log`
and the full `end_to_end` path at every size. We have not established the
cause and do not claim one here; the numbers are what the machines measured.

CI hosted runners, recorded 2026-08-24 from the bench job's `--quick` profile —
**informational only, never a gate (D-021)**: the hosted pool is heterogeneous,
with +20–36% run-to-run variance measured on identical code:

| runner | `end_to_end/1024` |
|---|---|
| `ci-ubuntu-latest` (4-vCPU x86_64) | 129 MS/s |
| `ci-macos-14` (Apple-Silicon M-series) | 192 MS/s |

Reading per D-012: **NFR-P1's guarantee is read off `end_to_end` at the
default FFT size 1024** — 148 MS/s single-core on the i9-11900K and
315.8 MS/s on the M3 Ultra, against the 61.44 MS/s target — and the rest of
the curve (512–32768) is reported so the shape is
known rather than assumed. `end_to_end` is `SpectrumAnalyzer::process`, the
whole §7.0 CPU path (window → FFT → mag² → dBFS → fftshift); the stage
benches isolate the windowed FFT and the mag/log tail the way §3.2 measured
them. These are the numbers for the README perf table once the README lands
(docs lane) — and per D-022 that table must name the exact CPU behind every
number and must not imply coverage the fleet does not have: fewer honest rows
beat a complete-looking table.

## Baselines and the regression gate (as ruled by D-021)

Spec §10 asks that a >10% regression be caught; D-021 rules **where that
measurement is trustworthy enough to gate on**:

* **Authoritative gate — fixed hardware.** `baselines/<class>.json` stores
  medians for a **fixed, known machine class** (a hardware descriptor, never a
  host name; this repo is public). `cargo run -p xtask -- bench-check --class
  <class>` on that machine fails on a median more than 10% over the stored
  baseline. `x86_64-linux-i9-11900k.json` is the current class of record.
* **CI hosted runners — advisory only.** The CI bench job runs `bench-check
  --advisory`: it prints the comparison and uploads the measured results as an
  artifact, but never fails the build. GitHub's hosted pool is a machine
  lottery (+20–36% observed on identical code), so a wall-clock gate there
  measures the draw, not the code — and widening the threshold past that
  variance would wave real regressions through instead.
* Baselines must be recorded with the same profile (`--quick` or full) they
  are checked with.
* An intentional perf change updates the baseline in the same PR, reviewed
  like any other golden update.
