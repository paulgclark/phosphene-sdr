# High-Level Specification — Standalone RTSA ("fosphor-style") Spectrum Analyzer

| | |
|---|---|
| **Name** | `phosphene` (decided 2026-08-24 — see §2) |
| **Version** | 0.1 (high-level) |
| **Date** | 2026-08-23 |
| **Owner** | Paul Clark / Factorial Labs |
| **Status** | Historical planning draft, predating the shipped v0.1.0 implementation |
| **Platforms** | Linux (x86_64, aarch64; X11 **and native Wayland**), macOS 13+ (Apple Silicon) (D-119) |
| **Language** | Rust (decided) |
| **Project license** | MIT, © Paul Clark / Factorial Labs (decided) |
| **v1 scope** | Fixed-tune RTSA display; sweep/scan is a post-v1 phase (decided) |

---

## 0. How to use this document

This is the product-level source of truth. It expands into detailed implementation specs, preserving requirement IDs (`FR-*`, `NFR-*`, `LC-*`) for traceability. Suggested decomposition into implementation specs:

1. **DSP core spec** — normative math of §7 plus exact calibration conventions and golden test vectors.
2. **Rendering & UI spec** — wgpu pipeline, egui integration, layout, keyboard/mouse map, mockups.
3. **Source backend spec** — the `SampleSource` trait, one section per backend of §5.2, each tagged with its licensing integration pattern from §4.5.
4. **Performance & acceleration spec** — SIMD and GPU compute paths, benchmark harness, regression thresholds.
5. **Build, CI, release & packaging spec** — §9/§10 milestones, artifact matrix, license gating.
6. **Docs & launch spec** — README, demo assets, distribution channels (§11).

Two constraints in this document are **non-negotiable and must be propagated verbatim into every downstream spec and every coder task prompt**:

> **LC-1 (clean-room):** No agent may clone, read, quote, or translate the source code of fosphor / gr-fosphor (GPLv3) or any other GPL project (including SDR++). §7 of this document fully defines the algorithms. Observing *behavior* (screenshots, videos, published talks, running a binary) is permitted; source is not.
>
> **LC-2 (license policy):** No GPL-licensed code may be linked, vendored, or statically/dynamically bundled into the repository or release artifacts. Permitted: MIT, BSD, Apache-2.0, Zlib, BSL-1.0, MPL-2.0, CC0, OFL (fonts). LGPL only via dynamic linking to a system-installed library. GPL functionality is reached only via the subprocess or runtime-plugin patterns of §4.5. Enforced in CI (§10).

Where this document says "default," the value must be user-configurable unless stated otherwise. Open items requiring a decision are collected in §13.

---

## 1. Product vision

**One-liner:** a single-binary, real-time spectrum analyzer for Linux and macOS that gives you the famous "phosphor persistence" RTSA display — histogram-graded spectrum, waterfall, live and max-hold traces — from any IQ source, with zero exotic dependencies: download, pipe samples in or name your SDR, and it runs.

**Why it should exist.** gr-fosphor delivers the best FOSS spectrum visualization ever made, but it is GPLv3, effectively dormant (mirror's last commit March 2024), requires a working OpenCL ICD plus OpenCL↔OpenGL interop created from a GLX context (no EGL path — broken on native Wayland), uses APIs Apple deprecated years ago, and drags in GNU Radio, Qt5, Boost, and freetype to build. In 2026, none of those constraints is necessary: modern CPUs process full SDR rates without a GPU, and portable GPU compute (wgpu → Vulkan/Metal) needs no interop layer. Nothing in the FOSS ecosystem currently offers a permissively-licensed, cross-platform, native persistence display (§3.3). This tool fills that gap.

**Target users.** SDR students and hobbyists (RTL-SDR, HackRF, Pluto), professionals and RF labs (USRP, LimeSDR, bladeRF), GNU Radio users who want a better spectrum sink without an OOT module, and instructors demoing RF concepts live.

**Non-goals for v1:** demodulation of any kind; transmit; recording pipelines beyond screenshots (IQ capture is a later nice-to-have); a full receiver UX (this is an *instrument*, not an SDR++ competitor); Windows support (do not foreclose it — the chosen stack is Windows-capable — but do not spend v1 effort on it); plugin APIs; remote/web UI.

---

## 2. Name: `phosphene` (decided 2026-08-24)

**phosphene** — the phantom light perceived without light entering the eye: fitting for the tool that shows you signals other analyzers can't. Pronounced *FOSS-feen* (yes, a FOSS tool with FOSS in the name). It tips its hat to fosphor phonetically without borrowing the identity — in prose, "fosphor-style" remains a factual comparison, never part of the brand, and the name must **not** be presented as affiliated with the osmocom project. Binary, repo, crate, config dir, and Homebrew formula all use `phosphene`.

Availability verified 2026-08-24: crates.io exact name free — publish the `phosphene` crate with the first tagged release to lock it; no dominant software collision on GitHub (a dormant music-viz library and an old GNOME theme share the word); the "phosphene sdr" search space is clear. Remaining diligence at M2: `brew search phosphene` before the tap formula lands (low risk — it's our own tap) and a casual trademark scan.

Branding note: the in-house long-persistence colormap (FR-D8) is named **p7**, after the WWII radar phosphor with the long yellow-green afterglow; the README's colormap table documents the deep cut.

---

## 3. Research findings that shape this design

Findings below were produced during the feasibility study (2026-08-23): direct inspection of the gr-fosphor tree (for spec-authoring only — implementers see §LC-1), throughput microbenchmarks, and a prior-art sweep.

### 3.1 What makes fosphor's display distinctive (behavioral description)

1. **Process every sample.** No decimation between the antenna and the display; at 20 MS/s, all ~19.5k FFTs/s contribute. This is why short transients (frequency-hoppers, bursts, radar chirps) are *visible* rather than statistically invisible as in decimating displays.
2. **Persistence histogram.** For each frequency bin, a distribution over power levels accumulated with exponential rise/decay — rendering how *often* each (freq, power) cell is hit. Common signals glow hot; rare transients flash and fade like a CRT phosphor. This is the standard display mode of hardware RTSAs (Tektronix "DPX", R&S, Signal Hound) — the technique is public, generic instrument technology and predates fosphor.
3. Simultaneous **live trace** (short-time average), **max-hold trace** (with slow decay), and **waterfall**, sharing the frequency axis, with an adjustable histogram/waterfall split.
4. Fixed 1024-point FFT (a fosphor limitation we will not inherit), 128 power levels, intensity → colormap.

### 3.2 Feasibility benchmarks (measured, conservative)

Measured on a deliberately weak cloud container (2 vCPU, 2.8 GHz Xeon). Paul's target machines (M1 Max MacBook Pro, Threadripper) are multiples faster per core; re-baseline on real hardware in M0.

| Operation | Measured throughput | Note |
|---|---|---|
| Windowed 1024-pt complex FFT (f32) | **154 MS/s single core** | scipy/pocketfft; rustfft is comparable class |
| Magnitude² + log10 | ~148 MS/s single core | numpy, unfused; fused SIMD will beat this |
| Histogram accumulation (1024×128, u32) | ~900 M updates/s single core | plain C, `-O3`; = ~900 MS/s equivalent |

**Implication (core design bet):** a CPU-only path processes 100% of samples at ≥20 MS/s on modest laptops and ≥61.44 MS/s on target-class hardware using 2–3 cores. The GPU is an *optional* accelerator (200+ MS/s, lower CPU load), not a dependency. This inverts fosphor's 2013 architecture and eliminates its entire install-pain surface.

### 3.3 Prior art / competitive landscape

| Tool | Persistence display | Platforms | License | Notes |
|---|---|---|---|---|
| gr-fosphor | yes (the benchmark) | Linux/macOS/Win, fragile | GPLv3 | OpenCL+GL interop, GLX-only on Linux; dormant; needs GNU Radio. Prebuilt via radioconda (`gnuradio-fosphor`, updated Mar 2025) |
| Khanfar Phosphor (2025) | yes | **Windows only** | GPLv3 | Repackages the fosphor engine; not a rewrite |
| SDR++ / SigDigger / GQRX | no (FFT + waterfall only) | cross-platform | GPL | Receivers, not RTSA instruments |
| Maia SDR | yes-ish (FPGA waterfall) | Pluto-only | varies | In-FPGA, not a desktop tool |
| **This project** | **yes** | **Linux + macOS native (Wayland OK)** | **MIT** | standalone, single binary, SDR-agnostic |

### 3.4 References

- gr-fosphor mirror: https://github.com/osmocom/gr-fosphor — docs: https://projects.osmocom.org/projects/sdr/wiki/Fosphor
- radioconda prebuilt package (today's stopgap): https://anaconda.org/ryanvolz/gnuradio-fosphor
- Khanfar Phosphor coverage: https://www.rtl-sdr.com/khanfar-phosphor-real-time-gpu-accelerated-spectrum-visualization-for-sdrs/
- SigDigger: https://batchdrake.github.io/SigDigger/ · SDR++: https://github.com/AlexandreRouma/SDRPlusPlus (GPL — LC-1 applies)
- wgpu: https://github.com/gfx-rs/wgpu · egui: https://github.com/emilk/egui · rustfft: https://github.com/ejmahler/RustFFT

---

## 4. Legal & licensing constraints (⚠ read before designing anything)

### 4.1 Project license

MIT, copyright "Paul Clark / Factorial Labs". Every source file carries `SPDX-License-Identifier: MIT`. Repo contains `LICENSE`, `CITATION.cff`, and a README credit block with links to Paul's site and GitHub. Attribution is achieved by the MIT notice-retention requirement plus the soft mechanisms in §11 — no license gimmicks.

### 4.2 Clean-room rule (LC-1)

Restated from §0; propagate verbatim. Rationale: fosphor is GPLv3 and its kernel source explicitly asserts that software using the kernels is a derivative work. This spec's §7 defines all needed algorithms from standard, public DSP practice so no implementer ever needs GPL source. Any PR or agent output that appears translated from fosphor must be rejected and rewritten.

### 4.3 Dependency license policy (LC-2)

Restated from §0. Additional specifics: **FFTW is GPL — prohibited**; use rustfft (MIT/Apache-2.0). Colormaps: matplotlib's viridis/magma/inferno/plasma are CC0; Google's turbo is Apache-2.0 — both fine; do **not** copy fosphor's colormap tables. Embedded UI font must be OFL or Apache (e.g., Inter, JetBrains Mono, B612). CI enforces the policy with `cargo-deny` (§10).

### 4.4 SDR library license audit (verified from upstream repos, 2026-08-23)

| Library (device) | License (verified) | Ruling for our MIT binary |
|---|---|---|
| **SoapySDR** core | Boost 1.0 | ✅ May link directly. Its hardware modules (SoapyUHD, SoapyRTLSDR, SoapyPlutoSDR…) are user-installed **runtime plugins** loaded by SoapySDR — their licenses never touch our artifacts |
| **libhackrf** (HackRF) | BSD-3-Clause in `host/libhackrf` source headers (repo-root COPYING is GPLv2, covering firmware/tools) | ✅ May link. Re-verify per-file at integration time; never vendor anything outside `host/libhackrf` |
| **libairspy** (Airspy) | Permissive per-component (`libairspy/LICENSE.md`, BSD-style) | ✅ May link; verify exact text at integration |
| **libiio** (PlutoSDR) | LGPL-2.1+ (library; some files MIT; bundled iio-utils are GPL — don't ship) | ✅ Dynamic link only |
| **LimeSuite** (LimeSDR) | Apache-2.0 | ✅ May link |
| **libbladeRF** (bladeRF) | LGPL-2.1 (repo's `bladeRF-cli` is GPLv2 — don't ship) | ✅ Dynamic link only |
| **libzmq** (ZMQ input) | MPL-2.0 (relicensed from LGPL) | ✅ May link; or prefer the pure-Rust `zeromq` crate (MIT) and skip the C dependency |
| **UHD** (USRP) | **GPL-3.0** (commercial alternative from Ettus only) | ❌ **Never link.** Reach USRPs via SoapySDR runtime module or subprocess |
| **librtlsdr** (RTL-SDR) | **GPL-2.0** | ❌ **Never link.** Reach via `rtl_sdr` subprocess, the `rtl_tcp` network protocol (a protocol is not code), or SoapySDR runtime module |
| fosphor / gr-fosphor | GPL-3.0 (+ derivative-work note on kernels) | ❌ LC-1: source is off-limits entirely |

### 4.5 The three integration patterns (the licensing firewall)

Every SDR backend in every downstream spec must be tagged with exactly one of:

- **Pattern A — direct link:** permissive libraries only (libhackrf, libairspy, LimeSuite, SoapySDR core) plus LGPL via dynamic linking (libiio, libbladeRF). Feature-gated so the base binary builds with none of them.
- **Pattern B — subprocess capture:** spawn a user-installed CLI tool and read raw IQ from its stdout (`rtl_sdr -`, `hackrf_transfer -r -`, UHD's `rx_samples_to_file` if stdout-capable — verify in detailed spec). Zero linking → zero license coupling, regardless of the tool's license. Also independently valuable: trivially debuggable, and matches the tool's pipe-first design.
- **Pattern C — runtime plugin host:** link SoapySDR core (Boost 1.0); it discovers and loads whatever vendor modules the *user* installed, GPL ones included, at runtime on the user's machine. Our repo and release artifacts contain no GPL bits. One pattern covers uhd/lime/rtl/hackrf/airspy/pluto at once.

GPL-licensed devices (USRP, RTL-SDR) are reachable **only** via B or C. This is settled architecture, not an implementation choice.

---

## 5. Functional requirements

### 5.1 Display (FR-D)

- **FR-D1 — Persistence histogram spectrum** (the centerpiece). Frequency × power grid rendered as intensity through a colormap, per the accumulation model in §7.3. User controls: persistence time (decay τ), intensity/gamma, colormap. This surface must look spectacular by default; it is the marketing asset.
- **FR-D2 — Live spectrum trace.** Short-time-averaged line overlaid on the histogram (§7.4); averaging time configurable.
- **FR-D3 — Max-hold trace.** Toggleable; pure-max and slow-decay variants (§7.5).
- **FR-D4 — Waterfall.** Scrolling time-frequency strip below the histogram; adjustable time span; histogram/waterfall split ratio draggable from 0 (waterfall only) to 1 (histogram only); row aggregation per §7.6.
- **FR-D5 — Axes, grid, labels.** Frequency axis from source metadata or CLI (`--center`, `--rate`); power axis with reference level and dB/div; waterfall time labels; ~10 grid divisions; all text legible at HiDPI.
- **FR-D6 — Cursor readout & zoom.** Mouse position → live frequency/power readout; drag-select frequency zoom (display-side zoom into the existing FFT; optionally recompute with narrower span when we control the device); scroll-wheel zoom; double-click or key to reset.
- **FR-D7 — Keyboard map.** Familiar to fosphor users: Up/Down = reference level; **the horizontal arrows belong to the radio** — Left/Right = centre-frequency tune by one window (coarse, the §7 tiling step) and Shift+Left/Right = tune by one tenth of a window (fine), with **dB/div off the arrows entirely** on its own free pair (D-066 §2 — tuning is the constantly-used action on a live radio and the modifier means *smaller step*, not *different quantity*; D-062 introduced the modifier and D-065 §1 was the first half of the move). Plus persistence, split ratio, colormap cycle, pause/freeze, screenshot, help overlay (`?`). Full map defined in the UI spec.
  - **Changing dB/div pins the floor and derives the ceiling** (D-066 §1): `db_bottom` holds — it is where the noise sits and where the eye is — and `db_top = db_bottom + 10 × dB/div`, so finer dB/div zooms *toward* the floor. The derived ceiling is never raised above **0 dBFS** (full scale; no signal can occupy the space above it, so empty headroom there is not a scale), and the division count falls out of what fits.
- **FR-D8 — Colormaps.** Ship ≥4: *inferno*-like (default), *viridis* (colorblind-safe), *turbo*, and an in-house long-persistence ramp named *p7* (blue-flash / yellow-green, after the WWII radar-scope phosphor). Sources per §4.3 only.
- **FR-D9 — Frame rate.** 60 fps target on integrated GPUs (Intel Iris / Apple base M-series); never below 30 fps at supported rates; window freely resizable.
- **FR-D10 — FFT & window control.** FFT size 512–32768, default 1024; window functions: Hann (default), Blackman-Harris 4-term, rectangular (others optional). Changing size/window live without restart.
- **FR-D11 — Honesty HUD.** Status bar always shows: input rate, % of samples processed (100% until overloaded), FFTs/s, fps, active compute backend (cpu / cpu-simd / gpu), and drop counter. **Overload sheds whole batches and reports it; it never silently decimates** (see NFR-P3).
- **FR-D12 — Pause/freeze** display (source keeps draining), and **screenshot to PNG** with one key (§11 badge rules).

### 5.2 Input sources (FR-S) — phased; every backend tagged A/B/C per §4.5

- **FR-S1 (P0) — Raw IQ on stdin / FIFO / file.** Formats: `cf32` (GNU Radio .cfile native), `cs32` (SigMF `ci32_le`; D-062), `cs16`, `cs8` (HackRF), `cu8` (RTL-SDR). Flags: `--format`, `--rate`, `--center`. File mode adds throttled real-time replay (default), `--loop`, and fast-forward. SigMF metadata sidecar reading is a P2 nicety (annotation rendering out of scope).
- **FR-S2 (P0) — ZMQ SUB client** compatible with GNU Radio's ZMQ PUB sink (raw cf32 vectors) — instantly usable from any GRC flowgraph with a stock block; this replaces the OOT-module role of gr-fosphor. Pure-Rust `zeromq` crate preferred (MIT); `libzmq` (MPL-2.0) acceptable fallback.
- **FR-S3 (P1) — rtl_tcp client** (protocol reimplementation, Pattern-B-adjacent: network, no linking): connect, set freq/rate/gain, stream cu8.
- **FR-S4 (P1) — Subprocess adapters (Pattern B):** curated wrappers that spawn `rtl_sdr`, `hackrf_transfer`, (and if stdout-capable, a UHD example utility) with correct arguments derived from `--sdr`, `--center`, `--rate`, `--gain`; parse stderr for diagnostics; supervise/restart. This is how `--sdr rtlsdr` and `--sdr hackrf` work *before* native backends exist.
- **FR-S5 (P1) — SoapySDR backend (Pattern C):** optional cargo feature + runtime detection; enumerate devices, stream, and control (freq/rate/gain/antenna/bandwidth). One backend lights up uhd, limesdr, rtlsdr, hackrf, airspy, pluto wherever the user has Soapy modules installed. Evaluate the existing `soapysdr` Rust crate (MIT/Apache) vs. thin in-house FFI in the detailed spec.
- **FR-S6 (P2) — Native backends (Pattern A)** where licensing permits, for zero-middleware UX: libhackrf (link), libiio/Pluto incl. `ip:`/`usb:` URIs (dynamic link), LimeSuite (link), libbladeRF (dynamic link), libairspy (link). Each is a cargo feature; absence never breaks the build.
- **FR-S7 — Device control panel.** When the active source supports control (C, and A backends; B where the protocol allows): center frequency (type-in + step buttons + keyboard), sample rate from device-valid list, gain(s), antenna, bias-T where applicable. Piped sources show metadata as read-only.
- **FR-S8 — Source selection UX.** `phosphene` with no args: enumerate available sources (devices found, plus "pipe/file/zmq" hints) and offer a picker; `--sdr <type>[:args]` goes direct: `uhd`, `zmq`, `pluto`, `limesdr`, `hackrf`, `rtlsdr`, `airspy`, `bladerf`, `soapy:<args>`, `stdin`, `file:<path>`, `tcp:<host:port>`. Each `--sdr` value maps to the best available backend (native → soapy → subprocess) with a clear message about which was chosen.

**Launch matrix (v1.0 must-have):** stdin/file/ZMQ (S1–S2), rtl_tcp (S3), subprocess rtlsdr + hackrf (S4), SoapySDR (S5). Native backends (S6) ship incrementally in point releases — the `--sdr` UX (S8) hides the difference.

### 5.3 Sweep / wideband scan (FR-W) — post-v1 (decided)

Design the source and accumulator abstractions so a future sweep mode (retune-and-stitch spans wider than device IBW, hackrf_sweep-style; per-segment persistence semantics; settling-time discard) can be added without re-architecture. `hackrf_sweep` subprocess ingest may arrive earlier as a cheap preview. No v1 implementation work beyond not painting this into a corner.

### 5.4 Configuration (FR-C)

- **FR-C1** CLI via clap with `--help` good enough to teach from; man page generated.
- **FR-C2** TOML config (`$XDG_CONFIG_HOME/phosphene/config.toml`; `~/Library/Application Support/` on macOS) for defaults: colormap, persistence, gains per device, named presets (`--preset lab-b210`).
- **FR-C3** Zero-config first run must work: sensible defaults everywhere; missing `--center` just labels the axis in relative Hz.
- **FR-C4 (P2)** `--headless --frames N --out shot.png`: render offscreen to PNG for docs, CI golden images, and scripted use.

---

## 6. Non-functional requirements

- **NFR-P1 — CPU-only throughput (the headline guarantee):** ≥20 MS/s cf32 sustained, 100% of samples processed, on a 2-core 2020-class x86 laptop; ≥61.44 MS/s on Apple M1-class and 6-core x86. (Container-measured basis in §3.2; re-baseline in M0.)
- **NFR-P2 — Accelerated targets:** cpu-simd ≥2× cpu-basic on the mag/log/histogram stages; GPU path ≥200 MS/s end-to-end on a mid-range dGPU / Apple Pro-class SoC, with CPU load cut ≥50% at 61.44 MS/s.
- **NFR-P3 — Degradation contract:** when input outpaces compute, drop *whole FFT batches*, count them, and show the honest percentage (FR-D11). Never partial-process silently. Ring buffer sized ≥250 ms at max supported rate.
- **NFR-P4 — Latency:** antenna-to-pixel ≤50 ms typical at ≥10 MS/s (excluding source-side buffering we don't control).
- **NFR-P5 — Footprint:** idle-with-no-source <1% CPU; binary ≤15 MB per platform; RAM <300 MB at defaults.
- **NFR-I1 — Install UX:** one download-and-run binary per platform from GitHub Releases (Linux x86_64 + aarch64, macOS arm64 — Apple Silicon only, D-119); `cargo install`; Homebrew tap; `.deb`. No mandatory runtime deps beyond the OS graphics stack. Unsigned-macOS-binary friction documented; notarization decision in §13.
- **NFR-I2 — Wayland is first-class.** Native Wayland via winit — no X11/GLX/XWayland requirement anywhere (this is a marquee fix over fosphor; say so in the README).
- **NFR-Q1 — Code quality gates:** rustfmt + clippy clean; `unsafe` only in audited, feature-gated hot paths (SIMD intrinsics, FFI) with `// SAFETY:` comments; public items documented; CONTRIBUTING.md written for human + agent contributors.
- **NFR-Q2 — Maintainability by owner:** Paul reviews Rust but lives in Python/C — favor boring, explicit Rust; small crates with sharp boundaries (§8.1); no macro-heavy cleverness; every crate has a doc-level architecture comment.
- **NFR-T1 — No telemetry, no network calls** except user-requested sources and an optional manual "check for updates" (off by default). State this in the README; it's a trust selling point.

---

## 7. Algorithm definitions (normative, clean-room reference)

Everything below is standard published DSP/instrumentation practice; it is the *only* algorithm reference implementers need (LC-1). Constants marked *tune* get final values during M1 visual tuning; ranges given are starting points.

### 7.0 Pipeline

```
                 ┌────────────────────────────────────────────── compute thread pool ─┐
source thread    │  ┌────────┐  ┌─────┐  ┌──────────┐  ┌────────────────────────────┐ │   render/main thread
[SDR/pipe/net] ─►│─►│ window │─►│ FFT │─►│ |·|² dBFS│─►│ accumulate: histo counts,  │ │──► snapshot swap ──► GPU:
  SPSC ring      │  │ (Hann) │  │ N pt│  │ fftshift │  │ live EMA, max, wf rows     │ │      (triple buffer)   decay+colormap,
  + drop count   │  └────────┘  └─────┘  └──────────┘  └────────────────────────────┘ │      textures, traces, egui overlay
                 └────────────────────────────────────────────────────────────────────┘
```

Compute runs in batches of `K` consecutive N-sample frames (K sized so one batch ≈ 2–8 ms of work). Display consumes the latest accumulated state at its own cadence (~60 Hz); the two rates are fully decoupled.

### 7.1 Segmentation & windowing

Non-overlapping N-sample frames (N = FFT size), covering 100% of input samples — the transient-visibility guarantee. Window w[n]: periodic Hann default. Optional overlap (50/75%) is a post-v1 enhancement for low sample rates where FFT rate would otherwise be sluggish (< a few hundred FFTs/s).

### 7.2 Power spectrum

X = FFT(x·w), length N, then fftshift (DC centered). Per-bin power in dBFS:

```
P_k = 10·log10( |X_k|² / (N·Σw²[n]/N · N) ) + C_cal
```

with the convention pinned so that a full-scale complex exponential reads 0 dBFS at its peak bin (coherent-gain correction; exact constant-folding and the noise-marker/ENBW footnote are finalized in the DSP spec with golden vectors). Floor at −200 dB to avoid −inf. All math f32.

### 7.3 Persistence histogram (FR-D1)

State: intensity grid `I[k][ℓ]` (f32, k = 0..N−1 frequency bins, ℓ = 0..L−1 power levels; **L fixed at 128**, D-045 — vertical resolution beyond 128 levels is not visible at any realistic panel height, so it is not exposed as a user setting), spanning the **displayed** dB range — the same `[db_bottom, db_top]` pair FR-D5's axis draws, which is 10 divisions of dB/div below full scale at the default (D-065 §5) and, once dB/div has been changed, whatever the FR-D7 derivation left (D-066 §1: the floor is pinned, the ceiling derived and capped at 0 dBFS). The grid spans what the axis says it spans; it does not compute a second range of its own.

Per batch of K spectra — counts:
```
for each spectrum, each bin k:
    ℓ = clamp( round( (P_k − P_bottom) / (P_top − P_bottom) · (L−1) ), 0, L−1 )
    C[k][ℓ] += 1
```
Per display frame (Δt since last), fold counts into intensity with asymmetric exponential smoothing:
```
T[k][ℓ] = C[k][ℓ] / K_total_this_frame          # hit ratio 0..1
α_rise  = 1 − exp(−Δt / τ_rise)                  # τ_rise  ~ 0.02–0.07 s   (tune)
α_decay = 1 − exp(−Δt / τ_decay)                 # τ_decay ~ 0.25–2 s      (user "persistence")
I += (T ≥ I ? α_rise : α_decay) · (T − I);  clamp to [0,1];  then clear C
```
Render: `color = cmap( gamma_adjust(I) )` with an intensity/gamma control (tune). Cells below a small ε render as background. This asymmetric-EMA phosphor emulation is the textbook approach used across hardware RTSAs.

### 7.4 Live trace (FR-D2)

Per-bin EMA across spectra: `L_k ← L_k + α_live·(P_k − L_k)`, α_live from τ_live default 0.1 s; computed incrementally per batch (apply the per-spectrum EMA within the batch in closed form: one pass computing the batch's exponentially-weighted mean, then blend — the DSP spec gives the closed-form so cost is O(N) per batch, not per spectrum).

### 7.5 Max hold (FR-D3)

`M_k = max(M_k · d, max over batch of P_k)` where d = 1 (pure hold) or a per-frame decay toward the live trace (style chosen in detailed spec, §13); reset key.

### 7.6 Waterfall (FR-D4)

Ring texture of H rows (H ≥ 1024); each new row is written once and scrolling is a read-side offset.

**Feed.** The waterfall consumes the pipeline's spectra stream: **every** spectrum the compute path publishes is folded into the open row's interval (D-051 — a feed that samples one spectrum per display frame cannot honour the aggregation below and is non-conforming). Each row aggregates all spectra in its interval by **max** (default — transients survive) or mean (toggle); the mean is an arithmetic mean of the dBFS values, consistent with §7.4. An interval that closes with no spectrum in it — a stalled stretch, or signal time covered only by counted drops (NFR-P3/D-013: shed batches are folded in as elapsed time with no data) — emits a **floor row** (§7.2's −200 dB floor): an empty stretch of time must look empty, never repeat stale rows, and never be spliced out of the axis. Completed rows are uploaded **batched per display frame**, never one GPU write per row (D-047).

**Two speed modes (D-046, owner-accepted).** They differ in which quantity the user sets; with spectrum interval `t_s = N / f_s` and a panel of `P` rows:

* **traditional** — the user sets the panel **duration** `T` (time span); the row interval derives as `Δt = T / P` (R = P / T rows/s). For slow trends and occupancy.
* **fast** — the user sets the **row interval** `Δt` directly (time resolution); the panel duration derives as `T = P · Δt`. `Δt` is clamped no finer than one spectrum interval: `Δt ≥ t_s`. **Fast is the default**, at `Δt = 1 ms` (tune-class, D-046: 1 ms resolves an LTE subframe exactly).

One mode toggle and one pair of finer/coarser keys adjust whichever quantity the active mode owns, through the single FR-D7 keymap table.

**Rate-less sources (D-054, extends clarification C5).** With no declared sample rate there is no mapping to seconds, so the cadence is measured in **spectra** — the only clock such a source has: fast mode pins one row per spectrum (the finest resolution the FFT clock can state; the finer/coarser keys have nothing to adjust); traditional mode's span is a **row count** `S` (rows of spectra), aggregating `max(1, S / P)` spectra per row. The waterfall's time axis is then labelled in spectra/rows and **never in seconds** — the same rule the normalized frequency axis follows: the display never invents a unit it does not have.

**Colour mapping and its intensity control.** A row's power maps to colour in two steps. First **normalise** the value against the colour-scale range `[P_lo, P_hi]` — either shared with the histogram's displayed dB window, or the **independent auto-range** over the retained rows (toggle; **auto-range is the default**, D-065 §2 — and because an auto-ranged waterfall no longer agrees with the dB axis above it, the chrome states which mapping is in use):

```
t = clamp( (P − P_lo) / max(P_hi − P_lo, ε), 0, 1 )
```

Then **shape** the normalised value with a power-law intensity control and look the result up in the colormap:

```
color = cmap( t^γ_wf )        # γ_wf > 0, user control (tune); γ_wf = 1 is the linear mapping
```

`γ_wf` is the same power law §7.3 applies to the persistence intensity grid, applied here to the waterfall's own normalised power — and it is a **separate quantity from §7.3's `gamma_adjust`**, which is read only by the histogram: neither control affects the other's surface (D-065 §3). `γ_wf < 1` lifts low-power detail out of the noise floor; `γ_wf > 1` keeps only the strongest returns. **The control is applied after normalisation, deliberately**, so it composes with either range mode rather than competing with it: the range mode chooses *which* dB window the colours span, `γ_wf` chooses *how* they are distributed inside it — so the control is as meaningful under auto-range as under the shared range. Default `γ_wf = 1` (the mapping the waterfall had before the control existed). Exposed as a keyboard pair in the single FR-D7 table and as clickable controls in the chrome; its value is stated on the status bar (D-043 — a control whose effect cannot be read is indistinguishable from one that does not work).

The FR-D5 time labels state the row interval the rows actually carried, read back from the row store, never a value set elsewhere (D-031).

### 7.7 Memory & precision

f32 DSP; u32 counts; I as f32 (f16 acceptable on GPU). At N=8192, L=256: grid ≈ 8 MB — trivial. No allocation on the hot path; all buffers preallocated per (N, L, K).

### 7.8 Acceleration paths (in priority order)

1. **cpu-basic (always present, the guarantee):** rustfft; scalar/autovectorized everything; rayon or hand-rolled pool across batches.
2. **cpu-simd (feature `simd`):** NEON/AVX2 fused mag²→log2 via polynomial approximation (≤0.05 dB error budget — tune), vectorized level-index + histogram; runtime CPU-feature dispatch.
3. **gpu (feature `gpu`):** wgpu compute in WGSL — batched Stockham radix FFT (**original implementation**; textbook/permissively-published references only, per LC-1/LC-2), then per-workgroup shared-memory histogram with atomics merged to the global grid, decay and colormap on GPU, results written directly into the render textures (same device/queue — the interop problem fosphor had does not exist here). An intermediate hybrid (CPU FFT + GPU histogram/decay/render) is a legitimate stepping stone.
4. Selection: `--accel auto|cpu|simd|gpu`, runtime probe, graceful fallback, active backend shown in HUD (FR-D11).

---

## 8. Architecture

### 8.1 Workspace layout (crate boundaries are the maintainability contract — NFR-Q2)

```
phosphene/
├─ crates/
│  ├─ phosphene-core      # DSP + accumulators (§7). No GUI, no I/O, no unsafe (except simd feature). Fully unit-tested.
│  ├─ phosphene-sources   # SampleSource trait + backends, each a cargo feature: stdin/file, zmq, rtl_tcp,
│  │                      #   subprocess, soapy, hackrf, pluto, lime, bladerf, airspy, siggen (demo/test)
│  ├─ phosphene-render    # wgpu pipelines (histogram/waterfall/traces) + egui chrome + colormaps
│  └─ phosphene-app       # binary: CLI (clap), config, wiring, headless mode
├─ assets/                # font (OFL), colormap tables (CC0/Apache), demo IQ generator configs
├─ xtask/                 # dev automation (goldens, packaging)
└─ .github/workflows/     # CI matrix, release, cargo-deny
```

### 8.2 Key abstractions

```rust
trait SampleSource {                                   // implemented per backend
    fn open(desc: &SourceDesc) -> Result<Self>;
    fn stream(&mut self, sink: RingProducer<Complex<f32>>) -> Result<()>; // converts cu8/cs8/cs16 → cf32 at ingest
    fn caps(&self) -> ControlCaps;                     // tune/rate/gain/antenna/none
    fn set(&mut self, ctl: Control) -> Result<()>;
    fn meta(&self) -> SourceMeta;                      // rate, center, label, format provenance
}

trait ComputeBackend {                                 // cpu-basic | cpu-simd | gpu
    fn configure(&mut self, cfg: DspConfig);           // N, window, L, dB range, K
    fn process(&mut self, batch: &[Complex<f32>], acc: &mut Accumulators);
}
```

`Accumulators` (histogram counts, live, max, waterfall rows, drop stats) snapshot into a triple buffer; the render thread swaps in the latest complete snapshot — no locks on the hot path. Ring buffer: SPSC (e.g., `rtrb`), sized per NFR-P3, with drop accounting at the producer.

### 8.3 Threading

Source thread (blocking reads / callbacks) → SPSC ring → compute pool (batch tasks; rayon or fixed pool) → snapshot swap → main thread renders (macOS requires UI on the main thread; winit event loop owns it). Device control commands flow back over a channel. Every thread boundary carries backpressure semantics defined in the detailed spec.

### 8.4 Error UX (the anti-fosphor)

Every failure a student can hit gets a *specific, actionable* message: device not found (list what was found; udev-rules hint on Linux with the exact file to install), permission denied, sample-rate invalid (print the device's valid rates), source binary missing for Pattern B ("install rtl-sdr: `sudo apt install rtl-sdr` / `brew install librtlsdr`"), hot-unplug (pause + offer reconnect, don't crash). Friendly errors are a feature with marketing value; budget real effort here.

### 8.5 Bill of materials (all verified license-compatible; lock versions in detailed spec)

| Crate | Purpose | License |
|---|---|---|
| wgpu | GPU render + compute (Vulkan/Metal/GL fallback) | MIT/Apache-2.0 |
| winit | Windowing incl. native Wayland + macOS | Apache-2.0 |
| egui + egui-wgpu | UI chrome, overlays, custom paint callbacks | MIT/Apache-2.0 |
| rustfft (+ realfft if useful) | CPU FFT | MIT/Apache-2.0 |
| rtrb, crossbeam | Lock-free ring / channels | MIT/Apache-2.0 |
| clap, serde, toml | CLI + config | MIT/Apache-2.0 |
| zeromq (pure Rust; fallback: zmq → libzmq MPL-2.0) | FR-S2 | MIT |
| soapysdr crate (evaluate) or in-house FFI | FR-S5 | MIT/Apache-2.0 (crate); BSL-1.0 (lib) |
| tracing, anyhow/thiserror | Diagnostics | MIT/Apache-2.0 |
| png / image | Screenshots | MIT/Apache-2.0 |
| criterion | Benchmarks | MIT/Apache-2.0 |
| cargo-deny (CI tool) | License gate (LC-2) | MIT/Apache-2.0 |

Native-backend FFI (libhackrf, libiio, LimeSuite, libbladeRF, libairspy) per the §4.4 rulings — prefer existing permissive binding crates where maintained; otherwise thin in-house `-sys` crates that link but never vendor GPL/LGPL sources (LGPL: dynamic only).

---

## 9. Roadmap — milestones, each releasable with a demo GIF

**M0 — Skeleton (foundation).** winit+wgpu window on Linux (X11+Wayland) and macOS; egui overlay; built-in **signal generator source** (tones, noise floor, chirps, bursty hopper — doubles as the demo mode and the test fixture); cpu-basic FFT → live trace rendering; CI building all three artifact targets; benchmark harness re-baselined on real hardware (updates §3.2 numbers).
*Exit:* 60 fps on both OSes; `phosphene --sdr siggen` looks alive; CI green.

**M1 — "It looks like fosphor."** Persistence histogram (§7.3) + waterfall + max hold + axes/labels + cursor readout + colormaps + keyboard map; stdin/file sources with all four formats; calibration golden tests; visual-tuning pass on τ/gamma defaults against hardware-RTSA reference imagery.
*Exit:* NFR-P1 at 20 MS/s on the reference laptop; screenshot indistinguishable in class from an RTSA display; drop-HUD honest under overload.

**M2 — Interop & first public release.** ZMQ (FR-S2), rtl_tcp (FR-S3), subprocess rtlsdr/hackrf (FR-S4); config/presets; screenshots; `--headless`; README with hero GIF + quickstarts; name finalized; repo public; v0.x announced quietly.
*Exit:* a GRC flowgraph and a $30 RTL-SDR both drive it with one command each.

**M3 — Devices.** SoapySDR backend (FR-S5) + device control panel (FR-S7) + `--sdr` resolution UX (FR-S8); native hackrf + pluto backends (FR-S6) if bindings cooperate; udev/permissions docs.
*Exit:* `phosphene --sdr hackrf`, `--sdr rtlsdr`, `--sdr pluto`, `--sdr uhd` (via soapy) work on a clean machine per README alone.

**M4 — Acceleration & 1.0 launch.** cpu-simd path; GPU compute path; perf HUD; NFR-P2 targets; packaging complete (brew tap, .deb, macOS arm64, checksums) (D-119); docs site; launch per §11.
*Exit:* v1.0 tag; launch checklist executed.

**M5+ (post-1.0).** Sweep mode (FR-W); SigMF; remaining native backends (lime, bladerf, airspy); overlap mode; IQ snippet capture; Windows enablement; Python bindings (PyO3) if demand shows.

---

## 10. Testing & CI

- **DSP goldens:** synthetic vectors (full-scale tone → 0 dBFS ±0.1 dB at the right bin; two-tone; known-σ noise floor vs ENBW) run per-commit; any calibration change is a reviewed golden update.
- **Persistence step-response:** feed a signal that appears/disappears; fit the intensity curve; assert τ_rise/τ_decay within tolerance. Deterministic mode (fixed seed, fixed Δt) for all accumulator tests.
- **Throughput regression:** criterion benches for window+FFT, mag/log, histogram, end-to-end synthetic pipe; CI compares against stored baselines per runner class and fails on >10% regression.
- **Soak:** 2-hour synthetic run in CI nightly; RSS flat, no fps decay, drop counter zero at sub-capacity rate.
- **Fuzz:** cargo-fuzz on format parsers (IQ ingest, rtl_tcp frames, ZMQ payloads, TOML config).
- **Rendering smoke:** headless golden PNGs via wgpu on lavapipe/llvmpipe (Linux CI) with perceptual-diff tolerance; macOS runner renders natively.
- **Matrix:** ubuntu-latest (X11 + Wayland-headless), macos-14 (arm64) only (D-119); `cargo deny check licenses` gate (LC-2); clippy/rustfmt gates (NFR-Q1).
- **Release workflow:** tag → build all artifacts → checksums + SBOM → GitHub Release draft.

---

## 11. Distribution, credit & launch (the actual point)

- **In-app credit:** About dialog (`?` → About) with version, links, MIT notice. `--version` prints the URL.
- **Screenshot badge:** exported PNGs (FR-D12) carry a small, tasteful corner caption "made with phosphene — factoriallabs.com", **on by default, one flag/config switch to disable**. Live view is never watermarked.
- **Channels:** GitHub Releases (canonical), Homebrew tap `factoriallabs/tap`, `.deb`, `cargo install`. AUR and nixpkgs will happen communally; be responsive.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| egui text/plot cost at 60 fps over large textures | Heavy surfaces are custom wgpu passes; egui draws chrome only. Validate in M0. |
| wgpu on ancient Linux iGPUs/drivers | wgpu GL backend fallback; document a floor; cpu path unaffected; lavapipe CI catches API misuse. |
| macOS Gatekeeper friction for raw downloads | Homebrew as the blessed path; document `xattr -d com.apple.quarantine`; decide on paid signing/notarization (§13). |
| Rust maintenance load on a Python/C-native owner | NFR-Q2 (boring Rust, sharp crate boundaries, doc comments); agent-assisted review workflow; CONTRIBUTING.md sets expectations. |
| License drift in transitive deps | cargo-deny gate fails CI on any non-allowlisted license (LC-2). |
| libhackrf per-file license surprises | Integration-time per-file audit (§4.4); fallback is Pattern B/C for HackRF too. |
| Scope creep toward "another receiver app" | §1 non-goals are the contract; out-of-scope work is declined. |
| Perf shortfall on unknown student hardware | cpu-basic is the guaranteed floor; honest drop HUD (NFR-P3) rather than mystery jank; overlap-free design keeps worst case linear. |
| Name collision discovered late | §2 checks before M2; repo starts under codename. |
| Visual "wow" misses (τ/gamma defaults look flat) | M1 exit criterion includes a deliberate visual-tuning pass judged against hardware-RTSA reference imagery, by Paul. |

---

## 13. Open questions

1. ~~Final name + CLI command~~ — **resolved 2026-08-24: `phosphene`** (§2).
2. Exact dBFS calibration convention + golden tolerances (§7.2).
3. Max-hold decay style: pure hold + reset, decay-toward-live, or both (§7.5).
4. Default FFT size 1024 (fosphor-classic) vs 2048 (modern screens) — recommend deciding from M1 visuals.
5. `soapysdr` crate vs in-house FFI (FR-S5); binding-crate choices for native backends (FR-S6).
6. UHD subprocess viability: does a stock UHD utility stream cf32/cs16 to stdout cleanly? If not, Soapy-only for USRP (fine).
7. macOS signing/notarization: pay for a Developer ID now or rely on Homebrew + documented workaround at launch?
8. Minimum supported GPU/driver floor for the gpu feature; GL-fallback policy.
9. egui-only UI vs custom-drawn axes (proto in M0 decides).
10. Config schema versioning/migration policy.
11. Demo IQ file hosting (repo LFS vs Releases assets).

---

## Appendix A — Target UX (command sketches; the product in eight lines)

```bash
# Pipes (works at M2):
rtl_sdr -f 100e6 -s 2.4e6 -g 30 - | phosphene --format cu8 --rate 2.4e6 --center 100e6
hackrf_transfer -r - -f 915e6 -s 20e6 | phosphene --format cs8 --rate 20e6 --center 915e6
phosphene --sdr file:capture.cfile --rate 10e6 --center 2.44e9 --loop
phosphene --sdr zmq:tcp://localhost:5555 --rate 8e6      # from a GRC ZMQ PUB sink

# Devices (M3+): tool picks native → soapy → subprocess automatically
phosphene --sdr rtlsdr --freq 100e6
phosphene --sdr hackrf --freq 915e6 --rate 20e6
phosphene --sdr pluto:ip:192.168.2.1 --freq 2.44e9 --rate 4e6
phosphene --sdr uhd --freq 2.44e9 --rate 30.72e6         # via SoapyUHD module
```

Typical sustained device rates for perf planning: RTL-SDR 2.4–3.2 MS/s · Pluto (USB2) ~4–8 MS/s sustained · HackRF 20 MS/s · Airspy R2 10 MS/s · LimeSDR / bladeRF 2.0 / USRP B2xx up to 61.44 MS/s. NFR-P1's 61.44 MS/s CPU-only target covers the entire USB-SDR fleet.

## Appendix B — Benchmark methodology (for re-baselining in M0)

Feasibility numbers in §3.2 were measured 2026-08-23 on a 2-vCPU 2.8 GHz Xeon container: (1) batched windowed 1024-pt complex64 FFT via scipy/pocketfft over a 4096×1024 array, throughput = samples/wall-time; (2) magnitude²+log10 via numpy over the same batch; (3) C histogram microbench, `-O3 -march=native`, 1024-bin × 128-level grid, precomputed level indices, 200k spectra, single thread. M0 must port these to criterion benches and re-baseline on: M1 Max MacBook Pro, one modern x86 laptop, one Threadripper-class desktop; those numbers become the CI regression baselines and the README perf table.

---

*End of high-level specification. Requirement IDs are stable; downstream specs cite them. Questions → Paul Clark / Factorial Labs.*
