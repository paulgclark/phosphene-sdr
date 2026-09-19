# 001 — RTSA v1 (fixed-tune)

**Product source of truth:** [`docs/rtsa-high-level-spec.md`](../../docs/rtsa-high-level-spec.md).
This spec scopes implementation 001 to **v1 = fixed-tune RTSA display** (D-001); sweep/scan is
explicitly out (spec §9 M5+).

Requirement IDs (`FR-*`, `NFR-*`, `LC-*`) are preserved verbatim from the product spec for
traceability — do not renumber them.

## Status

**Historical planning draft**, predating the shipped v0.1.0 implementation. §13 of the product
spec lists open items from that planning phase; see the shipped source and README for current
behavior.

## Decomposition (product spec §0)

The product spec prescribes six implementation specs. Their natural boundaries are the
workspace crates (§8.1), which are also the disjoint-lane globs:

| Implementation spec | Crate(s) | Notes |
|---|---|---|
| DSP core | `phosphene-core` | normative §7 math, calibration conventions, golden vectors |
| Rendering & UI | `phosphene-render` | wgpu pipelines, egui chrome, layout, input map |
| Source backends | `phosphene-sources` | `SampleSource` trait; one section per §5.2 backend, each tagged with its §4.5 licensing pattern |
| Performance & acceleration | `core` + `render` | SIMD and GPU paths, bench harness, regression thresholds |
| Build, CI, release, packaging | `xtask`, `.github/` | §9/§10, artifact matrix, licence gating |
| Docs & launch | — | README, demo assets, distribution (§11) |

## Non-negotiables inherited

**LC-1 (clean-room)** and **LC-2 (licence policy)** must appear in every downstream spec and
every task prompt. They bind assessors as well as implementers.

---

## Clarifications — 2026-08-24

Clarify pass against the product spec. Owner-blocking items were already resolved
(D-006…D-010); everything below was master-resolvable and is recorded so no lane has to
guess. Substantive resolutions carry a `D-NNN`.

### C1 — Internal conflict: FR-D6 zoom vs "v1 is fixed-tune" (D-001) → **D-011**

FR-D6 says drag-zoom may "optionally recompute with narrower span when we control the
device." Retuning the device is exactly the sweep-adjacent behaviour D-001 puts post-v1, and
it silently drags device-control coupling into the display path. **v1 zoom is display-side
only** — reinterpreting the existing FFT. Device-assisted zoom is deferred with sweep.

### C2 — NFR-P1's throughput guarantee had no FFT size attached → **D-012**

FR-D10 makes FFT size user-variable 512–32768, but NFR-P1 states ≥20 MS/s with "100% of
samples processed" without saying at what size — and the §3.2 basis was measured at 1024.
The guarantee is now **specified at the default FFT size**, with the bench harness reporting
across the range so the shape is known rather than assumed.

### C3 — "Drop whole FFT batches" was untestable as written → **D-013**

NFR-P3 and FR-D11 hinge on a "batch" that nothing defined, so no seal could assert it.
Defined below, along with what the percentage actually counts.

### C4 — M1's exit criterion was not observable

*"Screenshot indistinguishable in class from an RTSA display"* cannot gate anything. Replaced
with the machinery §10 already specifies — headless golden PNG + perceptual diff — plus an
explicit owner look-review against D-010. The prose stays as **intent**; the gate is the
diff plus sign-off. Beauty is a requirement here (§1) but it is reviewed, not asserted.

### C5 — File-source defaults when `--rate` is absent

FR-C3 covers a missing `--center` (label in relative Hz) but not a missing `--rate`, which
throttled replay needs. Resolution: **absent `--rate` in file/stdin mode replays as fast as
the consumer accepts** (no throttle) and the HUD labels the axis in normalised frequency;
`--rate` enables real-time throttling. Never invent a rate.

### C6 — Remaining §13 items, resolved (none needed the owner)

| §13 item | Resolution |
|---|---|
| 4. FFT default 1024 vs 2048 | **1024** (FR-D10 already says so; the §13 phrasing conflicted). Revisit from M1 visuals — a change is a D-entry, not a silent edit. |
| 5. `soapysdr` crate vs in-house FFI | **Defer to the M3 source-backend spec.** Not on M0/M1's path; deciding now would be guessing at a dependency we have not exercised. |
| 6. UHD subprocess viability | **Defer to M3**, and treat Soapy-only for USRP as an acceptable outcome (the product spec already says "fine"). |
| 8. GPU/driver floor | **wgpu's own baseline** (Vulkan 1.1 / Metal 2 class). No GL fallback in v1 — CPU-only is the guaranteed path (D-005), so a GL tier would be a third code path for no guarantee. |
| 9. egui-only vs custom axes | **egui for chrome, custom-drawn for the data surface and its axes** — axes must align exactly with the wgpu-rendered grid, which egui layout cannot guarantee. M0's proto confirms. |
| 10. Config schema versioning | **`version` key, unknown keys ignored with a warning, never a hard failure.** A config that blocks startup after an upgrade violates the zero-friction goal (§1). |
| 11. Demo IQ hosting | **GitHub Releases assets, not git-LFS.** LFS on a public repo bills bandwidth on every clone and breaks shallow-clone CI. |

### C7 — Risk flagged, not resolved: NFR-P5's ≤15 MB binary

A wgpu + egui + rustfft binary with four colormaps and a bundled font is plausibly **over**
15 MB. This is a measurement, not a decision — **M0 must report actual artifact sizes** and,
if the target is infeasible, the number gets revised by D-entry rather than quietly missed.

### C8 — Dependency: LC-1 makes §7 completeness load-bearing

LC-1 forbids consulting the reference implementation, so **any gap in §7 is unfixable by
lookup** — it must be closed by specification. The DSP-core spec must therefore be reviewed
for completeness *as a precondition for M1*, not discovered mid-lane. If an implementer finds
§7 underspecified, the correct action is to stop and raise it, never to infer from behaviour.
