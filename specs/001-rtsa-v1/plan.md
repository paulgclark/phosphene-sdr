# 001 — RTSA v1 · implementation plan

Derived from `docs/rtsa-high-level-spec.md` and the clarifications in `spec.md`.
Crate boundaries (product spec §8.1) are the **disjoint-lane globs** — a lane owns a crate.

## Lane map (the disjointness contract)

| Lane glob | Owns | Never touches |
|---|---|---|
| `crates/phosphene-core/**` | DSP: window, FFT, mag/log, persistence histogram, live/max accumulators (§7) | GUI, I/O, any device code |
| `crates/phosphene-sources/**` | `SampleSource` trait + backends, each a cargo feature | DSP math, rendering |
| `crates/phosphene-render/**` | wgpu pipelines, egui chrome, colormaps, layout, input map | DSP math, source I/O |
| `crates/phosphene-app/**` | binary: clap CLI, config, wiring, headless | the internals of the three above |
| `xtask/**`, `.github/**` | dev automation, CI, packaging, cargo-deny | crate internals |

`phosphene-core` is the only crate with no dependency on the others, so it can be built and
sealed first and in isolation — which is why M0 starts there.

## M0 — sequencing

M0's exit is *"60 fps on both OSes, `--sdr siggen` looks alive, CI green"*. That decomposes
into four lanes, three of which are independent:

* **M0-A `core-skeleton`** — workspace + `phosphene-core`: window functions, rustfft wrapper,
  mag/log, and the **dBFS calibration of D-006** with its golden test (full-scale tone →
  0.0 dBFS ±0.1 dB, invariant across FFT size and window). No GUI. *Independent.*
* **M0-B `siggen-source`** — `phosphene-sources` + the `SampleSource` trait + the **signal
  generator** (tones, noise floor, chirps, bursty hopper). It is both demo mode and the test
  fixture for every later lane, which is why it is built before any real backend.
  *Independent* (depends only on the trait it defines).
* **M0-C `render-skeleton`** — `phosphene-render` + `phosphene-app`: winit+wgpu window on
  Linux (X11 **and** native Wayland) and macOS, egui overlay, a live trace, and the **D-010
  visual identity** from the first frame. *Independent* — renders synthetic data until M0-A
  and M0-B land.
* **M0-D `ci-and-bench`** — CI matrix (ubuntu X11 + Wayland-headless, macos-14 arm64,
  macos-13 x86_64), `cargo deny check licenses` (**LC-2 gate, from the first commit**),
  fmt/clippy gates, and the criterion bench harness that **re-baselines §3.2 on real
  hardware** (D-005) and **reports artifact sizes against NFR-P5's 15 MB** (clarification C7).
  *Independent.*

Wiring A+B+C into a running `--sdr siggen` is the M0 integration step, taken by whichever
lane lands last rather than as a fifth lane.

## Node assignment (capability routing)

* **The macOS CI runner** — macOS/ARM build lanes and the macOS CI leg.
* **The x86 build machine** — x86 Linux build/perf lanes; the bench re-baseline runs here and
  on the macOS CI runner.
* **The Linux CI runner** — CI only; never a lane target.
* **The SDR machine / the capture machine** — SDR hardware-in-loop, from M3. Not used in M0–M2.

## Review

Standard two-tier. **Assessor addendum for this repo: the independent reimplementation must
derive from spec §7 math ONLY — LC-1 binds assessors.**

## What M0 deliberately does NOT do

No persistence histogram (M1), no waterfall (M1), no real device backends (M3), no SIMD or
GPU compute (M4). M0 proves the skeleton is sound and the guarantees are measurable; it does
not chase the look beyond D-010's baseline chrome.
