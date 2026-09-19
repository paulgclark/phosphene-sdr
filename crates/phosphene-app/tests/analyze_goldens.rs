// SPDX-License-Identifier: MIT

//! AI-1 fix pass 1 — committed golden references for the annotation
//! design's [Golden]/[Headless] acceptance criteria (AC-1 through AC-5,
//! AC-13), through the real production binary (D-031: the CLI dispatch is
//! inside the seal, not beside it), exactly like `headless_golden.rs`.
//!
//! Every scene here is a **bursty** single-frequency signal (a
//! single-offset "hopper": on then off at one fixed bin), never a
//! continuous, 100%-duty tone — the one documented blind spot of the A0
//! per-bin noise-floor estimator (`phosphene_analyze::detect::noise` module
//! docs; independently confirmed by
//! `phosphene_analyze::detect::seal_tests::
//! robust_floor_is_not_pulled_up_by_strong_occupants`'s own
//! `floor[tone_bin] > -60.0` assertion). The dwell (10 ms) is kept shorter
//! than the estimator's 64-frame (~16 ms at this rate) window so the floor
//! never saturates on the signal itself.
//!
//! ⚠ **Lavapipe pinning (D-014).** `VK_ICD_FILENAMES` is set to the
//! system's lavapipe ICD for every spawned run, so goldens are produced on
//! the same software rasteriser CI uses, never whatever hardware adapter
//! this machine happens to have. Every run's stdout/stderr line
//! `adapter: <name> (<backend>)` is asserted to name a software rasteriser
//! (never a hardware GPU) — see `assert_software_adapter`. If the lavapipe
//! ICD file is missing, every test in this file fails loudly naming the
//! expected path, rather than silently falling back to a hardware adapter.
//!
//! **FL-4: the capture is deterministic (NFR-A4), not merely usual.** The
//! source data is a `file:<path>` replay with `--rate` given, which
//! `headless::run_source_frames` now recognises and routes to the
//! **lockstep** capture path (`headless::run_file_lockstep_frames`): the
//! file is read synchronously, and detect → track → measure → annotate runs
//! frame-count-synchronously — never on a real clock. There is no source
//! thread, no compute thread, and no clock anywhere in the capture path, so
//! the same scene and config always produce the same sequence of frames —
//! proved directly by [`lockstep_capture_is_byte_identical_across_two_runs`],
//! not merely asserted. That is also why every test below calls
//! [`run_analyze`] plainly rather than retrying: a deterministic capture is
//! either right or wrong on the first try, so there is no "cold frame" to
//! retry past.
//!
//! **Full coverage (D-125 item 6), not coverage-matched.** An earlier pass
//! of this lane (D-110) reproduced the *live worker's* coverage here
//! instead — each analysis cycle folding only the newest
//! `crate::analyze::TAP_HISTORY_FRAMES` of its `crate::analyze::CYCLE`'s
//! worth of stream frames, exactly as the live tap's bounded ring would —
//! because the tracker's then-frame-counted allowances (3-of-5 birth, a
//! 2-frame coast) could not hold one track through this scene's off
//! periods under full observation: a fresh track id about every 33 ms, no
//! label ever finishing its fade-in. D-125 fixed the tracker itself
//! (allowances in seconds, not frames — `phosphene_analyze::track::
//! tracker`) and the engine (`AnalysisState::cycle` now publishes a track
//! confirmed and retired within one cycle, not only one still alive at the
//! cycle's last frame), so this capture switched to full coverage — every
//! raw frame reaches the engine, matching what a real file replay now does
//! too (D-125 Amendment 4 item 3: a file source falls behind real time by
//! backpressure rather than shedding, so it is never coverage-limited
//! either). One consequence, visible in the goldens below: a *repeating*
//! burst (a hopper cycling on and off) now confirms and dies once per real
//! repetition, so its label legitimately prints more than once within a
//! single analysis cycle — `assert_labels_structurally`'s own doc comment
//! says why the structural check asserts identity and position, not an
//! exact repeat count.
//!
//! Regenerate deliberately with `PHOSPHENE_UPDATE_GOLDENS=1 cargo test
//! --features analyze -p phosphene-app --test analyze_goldens`, then
//! eyeball the committed PNGs before committing them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use phosphene_sources::{HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FFT: usize = 512;
const RATE_HZ: f64 = 2_048_000.0;

/// `--frames` for every bursty-hopper scene below, chosen so the schedule's
/// **last analysis cycle** (not merely the last render tick) is deterministically
/// hot, with margin.
///
/// Under the D-110 coverage-matched schedule, a new cycle fires every ~15.6
/// render ticks (`CYCLE` / one FFT's duration at this FFT size and rate,
/// divided by the render tick's own ~64-frame draw), each publishing a fresh
/// annotation snapshot that then sits unchanged in the feed until the next
/// cycle — so what matters is not which render tick is captured, but how
/// long *after* the most recent cycle fired it lands: the overlay's fade-in
/// (`overlay::FADE_S`, 0.2 s ≈ 12 render ticks) needs that long to finish
/// before the label is fully opaque. Measured directly against the real
/// `SigGen` scene (not derived from the arithmetic alone): a cycle fires hot
/// at render tick 93, the next (cold-or-not, it doesn't matter) at tick 109,
/// and `108` sits solidly inside that window — about 250 ms (15 ticks) after
/// the hot fire, comfortably past the fade-in, with one tick of margin
/// before the next cycle would overwrite it. The same value works for every
/// scene here (the fire schedule depends only on the render/cycle cadence,
/// never on a scene's seed, offset or hopper count), confirmed directly
/// against each one, including the crowded ten-hopper scene (10 of 10
/// confirmed) and the near-Nyquist one (AC-5).
const FRAMES_HOT: u32 = 108;

/// D-115: the structural label check's own position tolerance, Hz. Ten FFT
/// bins at this FFT size and rate (`RATE_HZ / FFT` = 4 kHz/bin) — comfortably
/// past the OBW/noise jitter actually observed (the anchor lands within a
/// fraction of a Hz of the hopper's true offset in practice, see
/// `d114_smallest_regressions_against_the_tolerance`'s own numbers for the
/// pixel-domain analogue), and about 8× tighter than the 313.7 kHz shift
/// (one label width, this scene's own backing-plate pixel width converted
/// through the grid's Hz-per-pixel) that must fail it — see
/// `structural_position_check_fails_when_band_moved_by_one_label_width`.
const LABEL_POSITION_TOLERANCE_HZ: f64 = 40_000.0;

/// The lavapipe Vulkan ICD this repo's CI (and this test) pins to (D-014).
/// Named explicitly rather than left to whatever `VK_ICD_FILENAMES` the
/// ambient shell happens to have, so a missing file is loud and specific.
const LAVAPIPE_ICD: &str = "/usr/share/vulkan/icd.d/lvp_icd.json";

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn decode_rgba(path: &Path, what: &str) -> (Vec<u8>, u32, u32) {
    let file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {what} {path:?} ({e})"));
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .unwrap_or_else(|e| panic!("{what} {path:?} unreadable: {e}"));
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG too large")];
    let info = reader
        .next_frame(&mut buf)
        .unwrap_or_else(|e| panic!("{what} {path:?} frame unreadable: {e}"));
    assert_eq!(info.color_type, png::ColorType::Rgba, "{what} color type");
    buf.truncate((info.width * info.height * 4) as usize);
    (buf, info.width, info.height)
}

/// The adapter sidecar's path next to a golden PNG: `<name>.adapter.txt`,
/// holding exactly the `adapter: <name> (<backend>)` line's payload that
/// produced that reference (D-101's own brief text: "each reference records
/// the adapter that produced it, and any mismatch names both").
fn adapter_sidecar(name: &str) -> PathBuf {
    goldens_dir().join(format!("{name}.adapter.txt"))
}

fn assert_matches_golden(rendered: &Path, name: &str, adapter: &str) {
    let (img, w, h) = decode_rgba(rendered, "rendered PNG");
    assert_eq!((w, h), (WIDTH, HEIGHT), "rendered PNG has the wrong size");

    let golden = goldens_dir().join(name);
    let sidecar = adapter_sidecar(name);
    if std::env::var_os("PHOSPHENE_UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(golden.parent().unwrap()).unwrap();
        std::fs::copy(rendered, &golden).expect("cannot write the golden");
        std::fs::write(&sidecar, adapter).expect("cannot write the adapter sidecar");
        return;
    }
    let recorded_adapter = std::fs::read_to_string(&sidecar).unwrap_or_else(|e| {
        panic!(
            "no adapter record for golden {name} at {sidecar:?} ({e}) — \
             regenerate with PHOSPHENE_UPDATE_GOLDENS=1"
        )
    });
    assert_eq!(
        recorded_adapter, adapter,
        "{name}'s golden was produced on a different adapter than this run: \
         recorded {recorded_adapter:?}, this run rendered on {adapter:?} — a \
         pixel diff between two different rasterisers is not meaningful \
         evidence either way, so this stops here rather than comparing"
    );
    let (golden_img, gw, gh) =
        decode_rgba(&golden, "golden (generate with PHOSPHENE_UPDATE_GOLDENS=1)");
    assert_eq!(
        (gw, gh),
        (WIDTH, HEIGHT),
        "golden {name} has the wrong size"
    );

    let mut sum = 0u64;
    let mut outliers = 0u64;
    let mut n = 0u64;
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            let a = &img[o..o + 4];
            let b = &golden_img[o..o + 4];
            let d = a[..3]
                .iter()
                .zip(&b[..3])
                .map(|(x, y)| x.abs_diff(*y) as u64)
                .collect::<Vec<_>>();
            sum += d.iter().sum::<u64>();
            if d.iter().copied().max().unwrap() > 32 {
                outliers += 1;
            }
            n += 1;
        }
    }
    let mean = sum as f64 / (n as f64 * 3.0);
    let outlier_frac = outliers as f64 / n as f64;
    // Reported unconditionally (not only in the panic message on failure),
    // for the after arm's outlier-fraction distribution in the PR.
    eprintln!("{name}: mean abs diff {mean:.5}, outlier fraction {outlier_frac:.6}");
    assert!(
        mean <= MEAN_LIMIT && outlier_frac <= OUTLIER_FRAC_LIMIT,
        "{name} deviates from its golden (D-014): mean abs diff {mean:.4} \
         (limit {MEAN_LIMIT}), outlier fraction {outlier_frac:.5} \
         (limit {OUTLIER_FRAC_LIMIT})"
    );
}

/// D-014's thresholds for these goldens, re-derived for the lockstep capture
/// (FL-4), then re-examined twice more (D-114). No row is excluded any more:
/// the waterfall's own row cadence is now driven by the same frame-count
/// schedule as everything else in the picture (`run_file_lockstep_frames`),
/// not by a real-time row clock racing the render loop.
///
/// **D-114: keep two different nondeterminism sources apart, and know which
/// instrument catches which.**
/// * **Ordering nondeterminism, within one session and one build** (the
///   overlay's track-draw order, or anything like it) — caught by
///   [`lockstep_capture_is_byte_identical_across_two_runs`] and its negative
///   control ([`plate_pixel_determinism_check_has_power`]), which demand
///   exact equality and prove the check has power to fail. This is
///   deterministic and reproduces on demand; it held across 50+ repeated
///   full-suite runs at one commit (the after-fix arm's own evidence,
///   reported in the PR).
/// * **Renderer drift, across sessions, rebuilds and machines** — caught (to
///   whatever extent it can be) by the tolerance below, **never** by the
///   byte-identical check above. The two must not blur: if the tolerance
///   were the thing standing between a real ordering regression and a green
///   run, widening it for drift would silently cover for that regression
///   too. It does not, here, only because the byte-identical check already
///   catches ordering regressions on its own, at zero tolerance.
///
/// **Measured cross-session residual: real, nonzero, cause unconfirmed.** A
/// later pass of this same lane, on this same machine, same code, same
/// adapter string, found the *committed* goldens no longer matched a fresh
/// capture — mean abs diff 0.64-0.81, outlier fraction 0.0057-0.0071 across
/// all four scenes, reproduced identically by a direct binary invocation,
/// the test harness, and a full `cargo clean` rebuild, while two fresh
/// captures still matched each other exactly (the ordering check above,
/// unaffected). The `.cf32` scene bytes were checked and are bit-identical
/// across regeneration, and the adapter string and rasteriser library files
/// are unchanged. Beyond that, **the cause is an open hypothesis, not a
/// finding**: a rendering-session-scoped effect such as Mesa's on-disk
/// shader cache recompiling the software rasteriser's shader with different
/// float rounding after eviction would fit the evidence, but has not been
/// shown. (A candidate trigger — a background-task memory-pressure event —
/// was checked against the lane host's system journal and found
/// **unconfirmed**: the journal shows no OOM or memory-pressure kill in that
/// window, so this account does not rest on it.) The goldens were
/// re-rendered once more after this finding, under the session current at
/// commit time.
///
/// **D-114 found no real gap** between that residual and the smallest
/// regression the goldens must catch, measured directly against the
/// committed AC-2 golden
/// (`d114_smallest_regressions_against_the_tolerance`, `#[ignore]`d; numbers
/// in the PR): one label missing measures mean 1.81, outlier fraction
/// 0.0153 — both **under** D-014's 3.0 / 0.02, so it would pass; a band
/// moved by one label width (313.7 kHz, the plate's own measured pixel
/// width converted through the grid's Hz-per-pixel) measures mean 2.77,
/// outlier fraction 0.0211 — the mean also under 3.0, the outlier fraction
/// only barely over 0.02. D-114 forbade tuning a number into that span or
/// setting anything above D-014.
///
/// **D-115 (the federator's ruling): choose no limit, and narrow the job
/// instead.** These two thresholds stay at exactly D-014's own long-standing
/// 3.0 / 0.02, unchanged, and their scope is narrowed to what they can
/// actually see: **gross rendering regressions and renderer drift, never
/// the label property.**
///
/// **D-118 (the federator and the PM, on the wording of the blind spot),
/// verbatim:** "These goldens cannot detect a missing label (measured at
/// 1.81 / 0.0153 against limits of 3.0 / 0.02), and cannot reliably detect a
/// band moved by one label width (2.77 / 0.0211: over the 0.02 outlier
/// limit by 0.0011, with the cross-machine drift under that limit
/// unmeasured). The label property, position included, is asserted
/// structurally instead." That structural assertion reads the run's own
/// printed annotation output (`assert_labels_structurally`,
/// `parse_final_annotations`), with its own deterministic negative control
/// (`structural_position_check_fails_when_band_moved_by_one_label_width`,
/// D-095): identity alone would not have caught the one-label-width shift
/// above, which changes where a label draws, not its count or text, so the
/// structural check asserts position too, within a stated tolerance
/// (`LABEL_POSITION_TOLERANCE_HZ`).
const MEAN_LIMIT: f64 = 3.0;
const OUTLIER_FRAC_LIMIT: f64 = 0.02;

/// Write `scene` to a temporary raw cf32 file and return its path — no
/// binary fixture committed to the repo (public-bound; AGENTS.md).
fn write_scene(name: &str, mut scene: SigGen, seconds: f64) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "phosphene-analyze-golden-{name}-{}.cf32",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).expect("cannot create the IQ fixture");
    let total_samples = (seconds * RATE_HZ) as usize;
    let mut chunk = vec![phosphene_core::Complex::new(0.0f32, 0.0); FFT];
    let mut written = 0usize;
    let mut bytes = Vec::with_capacity(FFT * 8);
    while written < total_samples {
        scene.fill(&mut chunk);
        bytes.clear();
        for c in &chunk {
            bytes.extend_from_slice(&c.re.to_le_bytes());
            bytes.extend_from_slice(&c.im.to_le_bytes());
        }
        file.write_all(&bytes).expect("cannot write the IQ fixture");
        written += chunk.len();
    }
    path
}

fn bursty_hopper(offset_hz: f64, level_dbfs: f32) -> HopperConfig {
    HopperConfig {
        offsets_hz: vec![offset_hz],
        dwell_s: 0.010,
        period_s: 0.030,
        level_dbfs,
        order: HopOrder::Cycle,
    }
}

fn noise_scene(seed: u64) -> SigGenConfig {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = seed;
    cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
    cfg
}

struct RunOutput {
    stdout: String,
    stderr: String,
    png: PathBuf,
}

/// Serializes this file's tests against each other. FL-4: this is no longer
/// load-bearing for *correctness* — the lockstep capture has no wall-clock
/// cadence for concurrent CPU contention to degrade, which is exactly what
/// the app crate's own unit test
/// `headless::tests::lockstep_annotation_sequence_is_independent_of_wall_clock_jitter`
/// proves directly. Kept anyway so this file does not fire several
/// full-chrome software-rendering subprocesses at once on a shared CI
/// runner.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Spawn the real binary against `iq` with `--analyze`, pinned to lavapipe
/// (D-014), and capture what it printed.
fn run_analyze(test_name: &str, iq: &Path, frames: u32) -> RunOutput {
    let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        Path::new(LAVAPIPE_ICD).is_file(),
        "the lavapipe Vulkan ICD is not present at {LAVAPIPE_ICD} on this machine — \
         STOPPING rather than silently substituting a hardware adapter for a D-014 \
         golden reference (install mesa-vulkan-drivers, or point LAVAPIPE_ICD at the \
         right path for this platform)"
    );
    let png = std::env::temp_dir().join(format!(
        "phosphene-analyze-golden-{test_name}-{}.png",
        std::process::id()
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .env("VK_ICD_FILENAMES", LAVAPIPE_ICD)
        .args([
            "--headless",
            "--analyze",
            "--source",
            &format!("file:{}", iq.display()),
            "--rate",
            &RATE_HZ.to_string(),
            "--format",
            "cf32",
            "--loop",
            "--frames",
            &frames.to_string(),
            "--fft",
            &FFT.to_string(),
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
            "--out",
        ])
        .arg(&png)
        .output()
        .expect("failed to spawn the phosphene binary");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "phosphene --headless --analyze exited with {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    RunOutput {
        stdout,
        stderr,
        png,
    }
}

/// The `adapter: <name> (<backend>)` line's payload — the same string
/// [`assert_matches_golden`] records next to a golden and compares against
/// on later runs (D-101).
fn adapter_of(run: &RunOutput) -> String {
    let combined = format!("{}\n{}", run.stdout, run.stderr);
    let adapter_line = combined
        .lines()
        .find(|l| l.starts_with("adapter:"))
        .unwrap_or_else(|| panic!("no 'adapter: ...' line was printed:\n{combined}"));
    adapter_line
        .trim_start_matches("adapter:")
        .trim()
        .to_owned()
}

/// D-014: the run must actually have rendered on a software rasteriser
/// (lavapipe/llvmpipe), never a hardware GPU — a golden pinned to
/// `VK_ICD_FILENAMES` that silently fell back to hardware would be
/// worthless as a cross-machine reference.
fn assert_software_adapter(run: &RunOutput) {
    let adapter = adapter_of(run);
    let lower = adapter.to_lowercase();
    assert!(
        lower.contains("llvmpipe") || lower.contains("lavapipe") || lower.contains("swiftshader"),
        "expected a software rasteriser, got: {adapter}"
    );
}

fn annotation_count(run: &RunOutput) -> usize {
    let combined = format!("{}\n{}", run.stdout, run.stderr);
    let line = combined
        .lines()
        .find(|l| l.starts_with("annotations:"))
        .unwrap_or_else(|| panic!("no 'annotations: ...' line was printed:\n{combined}"));
    line.trim_start_matches("annotations:")
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("could not parse annotation count from: {line}"))
}

/// What the captured PNG's own pixels show — see
/// `SourceOutcome::final_annotation_count`'s doc comment for why this can
/// honestly differ from [`annotation_count`].
fn final_annotation_count(run: &RunOutput) -> usize {
    let combined = format!("{}\n{}", run.stdout, run.stderr);
    let line = combined
        .lines()
        .find(|l| l.starts_with("final_annotations:"))
        .unwrap_or_else(|| panic!("no 'final_annotations: ...' line was printed:\n{combined}"));
    line.trim_start_matches("final_annotations:")
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("could not parse final annotation count from: {line}"))
}

/// Runs `run_analyze` once and asserts the captured PNG itself actually
/// shows at least `min_final` live annotations — a golden reference for a
/// design mockup with a visible shade band is not evidence of that mockup if
/// the band never made it into the picture.
///
/// FL-4: this is a single run, on purpose. The old version of this helper
/// retried up to eight times because the live worker's wall-clock cadence,
/// racing real GPU rendering, made whether the literal final frame was "hot"
/// close to a coin flip. Under the lockstep capture, `final_annotations` is a
/// pure function of the scene, `--frames`, and the FFT/rate config — either
/// every run of a given scene lands hot, or every run lands cold, and no
/// number of retries changes which; [`FRAMES_HOT`]'s own doc comment derives
/// why it lands hot, with margin, for every scene here. A cold result here
/// means that arithmetic — or the scene, or the engine — actually changed,
/// which is exactly what should fail loudly instead of quietly retrying
/// past it.
fn assert_hot(test_name: &str, run: &RunOutput, min_final: usize) {
    let final_count = final_annotation_count(run);
    assert!(
        final_count >= min_final,
        "{test_name}: the captured final frame is cold (final_annotations \
         {final_count} < {min_final}) — under the deterministic lockstep \
         capture this is not a fluke to retry past; either the scene's \
         hopper/cycle timing no longer lines up with FRAMES_HOT (see its own \
         doc comment), or a real regression stopped the engine from \
         confirming the track:\n{}\n{}",
        run.stdout,
        run.stderr
    );
}

/// D-115: `(anchor_hz, label)` for every `final_annotation: anchor_hz=...
/// label=...` line the binary printed — the structural label check's own
/// input. Reads the real binary's output; never re-derives the frequency
/// mapping or re-runs any analysis itself (D-113's own condition: a
/// structural check on a copy of the pipeline would not be a check on the
/// binary at all).
fn parse_final_annotations(run: &RunOutput) -> Vec<(f64, String)> {
    let combined = format!("{}\n{}", run.stdout, run.stderr);
    combined
        .lines()
        .filter_map(|l| l.strip_prefix("final_annotation: anchor_hz="))
        .map(|rest| {
            let (hz, label) = rest
                .split_once(" label=")
                .unwrap_or_else(|| panic!("malformed final_annotation line: {rest:?}"));
            let anchor_hz: f64 = hz
                .parse()
                .unwrap_or_else(|e| panic!("bad anchor_hz {hz:?}: {e}"));
            (anchor_hz, label.to_owned())
        })
        .collect()
}

/// D-115: the label property's own gate — identity and position, read from
/// the run's own printed output, never from pixels. Sized so it covers
/// exactly what D-014's pixel tolerance was shown not to (its own blind
/// spot, documented beside `MEAN_LIMIT`/`OUTLIER_FRAC_LIMIT`): a missing or
/// wrong-content label, or one drawn in the wrong place.
///
/// D-125 item 6 (full coverage): checks structure, in both directions —
/// every `expected` identity/position is matched by at least one printed
/// label, and every printed label matches at least one `expected`
/// identity/position — rather than an exact one-to-one, positionally
/// zipped count. Under full coverage a repeating burst confirms and dies
/// once per real repetition that lands inside a single 250ms cycle
/// (`AnalysisState::cycle`'s own D-125 item 4), so the *same* expected
/// signal legitimately prints more than once in one cycle's final
/// snapshot — how many times is a function of this scene's hop timing
/// against the cycle boundary, not a meaningful count to pin exactly. This
/// is not a loosening of precision: a label with the wrong identity, the
/// wrong position, or no match at all still fails, on both sides.
///
/// `position_tolerance_hz` is deliberately far tighter than the "smallest
/// regression" this replaces (a shift of one label width, 313.7 kHz, per
/// `FRAMES_HOT`'s neighbour doc comment) and far looser than this scene's
/// own measured precision (the anchor lands within a fraction of a Hz of
/// the hopper's true offset in practice) — it exists to absorb genuine
/// measurement jitter (OBW estimation, noise-floor wander) across scenes and
/// runs, not to paper over a real positional miss.
fn assert_labels_structurally(
    test_name: &str,
    run: &RunOutput,
    expected: &[(f64, &str)],
    position_tolerance_hz: f64,
) {
    let actual = parse_final_annotations(run);
    let matches = |actual_hz: f64, actual_label: &str, anchor_hz: f64, label_contains: &str| {
        actual_label.contains(label_contains)
            && (actual_hz - anchor_hz).abs() <= position_tolerance_hz
    };
    for &(anchor_hz, label_contains) in expected {
        assert!(
            actual
                .iter()
                .any(|(hz, label)| matches(*hz, label, anchor_hz, label_contains)),
            "{test_name}: no label matched identity {label_contains:?} near \
             {anchor_hz} Hz (± {position_tolerance_hz} Hz) in the run's own \
             output: {actual:?}"
        );
    }
    for (actual_hz, actual_label) in &actual {
        assert!(
            expected
                .iter()
                .any(|&(hz, label)| matches(*actual_hz, actual_label, hz, label)),
            "{test_name}: unexpected label {actual_label:?} at {actual_hz} Hz \
             matched none of the expected identities/positions: {expected:?}"
        );
    }
}

/// AC-1: analysis on, zero tracks — the INSPECTOR chip shows a zero count
/// and no shade bands draw. A pure-noise scene is trivially reliable here:
/// nothing ever crosses the CFAR threshold.
#[test]
fn ac1_empty_band_shows_the_inspector_chip_at_zero() {
    let scene = SigGen::new(noise_scene(1)).unwrap();
    let iq = write_scene("ac1-empty", scene, 1.0);
    let run = run_analyze("ac1", &iq, 90);
    assert_software_adapter(&run);
    assert_eq!(
        annotation_count(&run),
        0,
        "a pure-noise scene must produce no annotations"
    );
    assert_matches_golden(&run.png, "analyze-ac1-empty-band.png", &adapter_of(&run));
}

/// AC-2: one signal in a quiet band — exactly one shade band and one label
/// at the track's true position (design mock-up A).
#[test]
fn ac2_one_signal_in_a_quiet_band() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("ac2-one-signal", scene, 1.5);
    // The golden is meant to show design mock-up A's shade band and label on
    // screen, not just prove detection happened at some point during the
    // run — `assert_hot` checks the captured frame itself is live.
    let run = run_analyze("ac2", &iq, FRAMES_HOT);
    assert_software_adapter(&run);
    assert_hot("ac2", &run, 1);
    // D-115: the label property itself — count, identity and position — is
    // asserted structurally, from the run's own printed output; the pixel
    // goldens' job is narrowed to gross rendering regressions and renderer
    // drift (see `MEAN_LIMIT`'s own doc comment for why: they cannot see a
    // missing label at all).
    assert_labels_structurally(
        "ac2",
        &run,
        &[(300_000.0, "SIGNAL")],
        LABEL_POSITION_TOLERANCE_HZ,
    );
    assert_matches_golden(&run.png, "analyze-ac2-one-signal.png", &adapter_of(&run));
}

/// D-115's negative control (D-095: deterministic, not "usually red"): a
/// band moved by one label width must fail the structural position check
/// above. Reuses AC-2's own scene, shifted by the backing plate's own
/// measured pixel width converted to Hz (see `FRAMES_HOT`'s neighbour doc
/// comment for the 313.7 kHz derivation) — far more than
/// `LABEL_POSITION_TOLERANCE_HZ` can absorb, and far more than plausible
/// measurement jitter, so this is not a coin flip.
#[test]
fn structural_position_check_fails_when_band_moved_by_one_label_width() {
    const SHIFT_HZ: f64 = 313_700.0;
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0 + SHIFT_HZ, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("structural-nc", scene, 1.5);
    let run = run_analyze("structural-nc", &iq, FRAMES_HOT);
    std::fs::remove_file(&iq).ok();
    assert_hot("structural-nc", &run, 1);

    // D-125 item 6 (full coverage): the repeating hopper can now print more
    // than one label (once per repetition landing inside the hot cycle,
    // `assert_labels_structurally`'s own doc comment) — the negative
    // control's job is that *none* of them is mistaken for a match at the
    // original 300 kHz, not that there is exactly one.
    let actual = parse_final_annotations(&run);
    assert!(
        !actual.is_empty(),
        "expected at least one label: {actual:?}"
    );
    for (actual_hz, label) in &actual {
        let off_by = (actual_hz - 300_000.0).abs();
        assert!(
            off_by > LABEL_POSITION_TOLERANCE_HZ,
            "a band moved a full label-width away (to {actual_hz} Hz, label \
             {label:?}) was still within the structural check's \
             {LABEL_POSITION_TOLERANCE_HZ} Hz tolerance of the original 300 kHz \
             (off by only {off_by} Hz) — the position check has no power to \
             catch this regression"
        );
    }
}

/// D-102 item 7, fix pass 1 — a pixel-level complement to
/// `label_backing_plate_draws_directly_behind_its_own_text_and_smaller_than_the_band`
/// in `phosphene_render::overlay`'s own unit tests: that test only proves
/// the *shapes* are correctly ordered, which the review's own finding was
/// not sufficient evidence for ("the pixels inside the text box must not
/// show the edge"). This scans the real composited PNG.
///
/// AC-2's scene (a single narrow signal at +300 kHz) puts the label's own
/// two band-edge hairlines almost exactly on the label's horizontal middle,
/// because the label is centred on the same anchor the hairlines straddle —
/// confirmed empirically across several captures of this scene: the
/// hairlines always land inside the rendered text row, never off to the
/// side.
#[test]
fn label_backing_plate_hides_the_band_edge_crossing_its_own_text() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("plate-hides-edge", scene, 1.5);
    let run = run_analyze("plate_hides_edge", &iq, FRAMES_HOT);
    assert_software_adapter(&run);
    assert_hot("plate_hides_edge", &run, 1);

    let (img, w, h) = decode_rgba(&run.png, "one-signal PNG");
    let px = |x: i64, y: i64| -> (i32, i32, i32) {
        let xi = x.clamp(0, w as i64 - 1) as u32;
        let yi = y.clamp(0, h as i64 - 1) as u32;
        let o = ((yi * w + xi) * 4) as usize;
        (img[o] as i32, img[o + 1] as i32, img[o + 2] as i32)
    };

    // The same anchor-fraction/margin arithmetic AC-13's own black-box probe
    // uses: the signal sits at +300 kHz of a ±1.024 MHz span, column
    // fraction ≈ 0.6465 of the grid width, margins matching `Layout`'s own
    // constants (crate-internal; duplicated here as a literal, same as
    // AC-13, since this is a black-box binary test).
    let grid_left = 56.0;
    let grid_right = f64::from(w) - 16.0;
    let anchor_x = (grid_left + (grid_right - grid_left) * 0.6465).round() as i64;

    // Locate the edge hairline's own column(s) by where they actually are:
    // bright well *below* the label lane (y=140, deep in the shade band,
    // past any glyph or leader-line pixel), searched in a small window
    // around the computed anchor column to absorb OBW-driven span jitter
    // between captures — never assumed at a hand-picked exact pixel.
    const BELOW_LABEL_Y: i64 = 140;
    const BRIGHT: i32 = 150;
    let edge_cols: Vec<i64> = (-15..=15)
        .map(|dx| anchor_x + dx)
        .filter(|&x| {
            let (r, g, b) = px(x, BELOW_LABEL_Y);
            r.max(g).max(b) > BRIGHT
        })
        .collect();
    assert!(
        !edge_cols.is_empty(),
        "could not locate the band's own edge hairline below the label at \
         x≈{anchor_x}, y={BELOW_LABEL_Y} — scene geometry changed?"
    );

    // The label's own text row for this 640x360 layout — empirically
    // measured the same way `WATERFALL_ROWS` above documents its own
    // measurement (a scan for theme.text-coloured pixels across several
    // captures of this exact scene).
    const LABEL_TEXT_ROWS: std::ops::RangeInclusive<i64> = 100..=107;
    let max_in_label = edge_cols
        .iter()
        .flat_map(|&x| LABEL_TEXT_ROWS.map(move |y| (x, y)))
        .map(|(x, y)| {
            let (r, g, b) = px(x, y);
            r.max(g).max(b)
        })
        .max()
        .expect("edge_cols is non-empty");
    // D-112: reported unconditionally (not only in the panic message on
    // failure) so the after arm's PR account can state this as a margin —
    // the worst value observed and its distance to the bound — rather than
    // a bare pass/fail.
    eprintln!(
        "plate-pixel margin: {max_in_label}/255 against the 32/255 bound \
         ({} to spare)",
        32 - max_in_label
    );
    assert!(
        max_in_label <= 32,
        "a band-edge pixel crossing the label's own text row reached \
         {max_in_label}/255 at x∈{edge_cols:?}, y∈{LABEL_TEXT_ROWS:?} — the \
         backing plate (D-102 item 7) must hide it under this codebase's \
         own 32/255 'meaningfully different' bound (the same bound \
         `assert_matches_golden`'s own outlier check uses), not merely \
         dim it (fix pass 1's finding: 140/255 alpha measured ~147/255 \
         here, clearly visible)"
    );
}

/// AC-3/AC-4: a crowded band past the declutter cap (default N = 8) —
/// enough simultaneous bursty signals that some are unlabelled-but-marked
/// (AC-3), and close enough in frequency that the labelled ones' fixed-lane
/// boxes collide and degrade (AC-4). One scene, one golden, both
/// situations: they compose in a single crowded band exactly as §2.3/§2.4
/// describe.
#[test]
fn ac3_ac4_crowded_band_past_the_cap_with_colliding_labels() {
    let mut cfg = noise_scene(3);
    // Ten bursty carriers, closely spaced (40 kHz apart) so their fixed-
    // lane label boxes are guaranteed to overlap, at levels that rank them
    // in a strict, distinguishable order (§2.3's total order).
    for i in 0..10i64 {
        let offset = (i - 5) as f64 * 40_000.0;
        let level = -15.0 - i as f32; // -15, -16, ..., -24 dBFS
        cfg.hoppers.push(bursty_hopper(offset, level));
    }
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("ac3-ac4-crowded", scene, 2.0);
    let run = run_analyze("ac3ac4", &iq, FRAMES_HOT);
    assert_software_adapter(&run);
    assert_hot("ac3ac4", &run, 8);
    assert_matches_golden(
        &run.png,
        "analyze-ac3-ac4-crowded-band.png",
        &adapter_of(&run),
    );
}

/// AC-13: a screenshot taken while annotations are on screen contains them
/// pixel-identical to the live frame — the same headless PNG-capture pass a
/// screenshot key would trigger (design §2.8: "no new code path"), so
/// AC-2's own golden is the evidence: annotation pixels (the halo band) are
/// present at the signal's own column, absent at the same column in the
/// AC-1 (empty-band) golden.
#[test]
fn ac13_the_captured_png_contains_real_annotation_pixels() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("ac13-one-signal", scene, 1.5);
    // The pixel probe below reads whatever the literal final frame shows,
    // so (unlike a plain `annotation_count` check) it needs the captured
    // frame itself to be hot, not just "detected at some point."
    let run = run_analyze("ac13", &iq, FRAMES_HOT);
    assert_software_adapter(&run);
    assert_hot("ac13", &run, 1);

    let empty_scene = SigGen::new(noise_scene(1)).unwrap();
    let empty_iq = write_scene("ac13-empty", empty_scene, 1.0);
    let empty_run = run_analyze("ac13-empty", &empty_iq, 90);

    // The signal sits at +300 kHz of a ±1.024 MHz span: column fraction
    // ≈ 0.5 + 300_000/2_048_000 ≈ 0.6465 of the grid width. The grid's
    // left margin (56pt) and right margin (16pt) match `Layout`'s own
    // constants (crate-internal in phosphene-render; duplicated here as a
    // literal since this is a black-box binary test).
    let (img, w, h) = decode_rgba(&run.png, "one-signal PNG");
    let (empty_img, ew, eh) = decode_rgba(&empty_run.png, "empty-band PNG");
    assert_eq!((w, h), (ew, eh));
    let grid_left = 56.0;
    let grid_right = w as f64 - 16.0;
    let x = (grid_left + (grid_right - grid_left) * 0.6465) as u32;
    // Inside the shaded band's full-height fill, clear of the label lane.
    // D-105: the default style is now the *stacked* label, whose three-line
    // lane (and backing plate) reaches noticeably further down than the
    // single line's ever did — `h / 3` (the original single-line-era
    // measurement) now lands inside that lane, which is why this is a
    // fixed fraction further down instead: empirically confirmed (a probe
    // of this exact scene) to sit well below the stacked lane, still
    // comfortably above the axis/status chrome.
    let y = h * 5 / 12;
    let px = |buf: &[u8], w: u32, x: u32, y: u32| {
        let o = ((y * w + x) * 4) as usize;
        (buf[o] as i32, buf[o + 1] as i32, buf[o + 2] as i32)
    };
    // Average a small neighborhood to absorb rasteriser/text jitter.
    let mut signal_sum = 0i64;
    let mut empty_sum = 0i64;
    let mut n = 0i64;
    for dx in -3i32..=3 {
        for dy in -3i32..=3 {
            let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
            let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
            let (r, g, b) = px(&img, w, sx, sy);
            signal_sum += (r + g + b) as i64;
            let (r, g, b) = px(&empty_img, ew, sx, sy);
            empty_sum += (r + g + b) as i64;
            n += 1;
        }
    }
    let signal_mean = signal_sum as f64 / n as f64;
    let empty_mean = empty_sum as f64 / n as f64;
    assert!(
        (signal_mean - empty_mean).abs() > 5.0,
        "expected the annotation's halo/shade pixels to differ from the same \
         column with no signal (signal {signal_mean:.1} vs empty {empty_mean:.1}) — \
         AC-13 needs real annotation pixels in the captured PNG, not just the \
         printed count"
    );
}

/// AC-5: a signal cut off at the band edge — the shade band clips exactly
/// to the grid edge, with an edge-arrow glyph on the clipped side (§3's
/// situation 5; the pure placement/clip arithmetic is unit-tested
/// deterministically in `phosphene_render::overlay`'s own AC-5 tests — this
/// is the rendered-pixel complement). A bursty carrier placed a few bins
/// short of Nyquist: its own occupied-bandwidth measurement (a few bins of
/// Hann-window mainlobe/sidelobe spread) pushes its span past the display's
/// fixed axis edge without needing any runtime zoom control.
#[test]
fn ac5_a_signal_cut_off_at_the_band_edge() {
    let mut cfg = noise_scene(5);
    // Half span is RATE_HZ/2 = 1_024_000 Hz; 1_012_000 leaves ~12 kHz (3
    // bins) of headroom before Nyquist for the OBW's own spread to cross.
    cfg.hoppers.push(bursty_hopper(1_012_000.0, -15.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("ac5-edge", scene, 1.5);
    let run = run_analyze("ac5", &iq, FRAMES_HOT);
    assert_software_adapter(&run);
    assert_hot("ac5", &run, 1);
    assert_matches_golden(&run.png, "analyze-ac5-edge-clip.png", &adapter_of(&run));
}

/// FL-4 seal: the lockstep capture is genuinely deterministic (NFR-A4), not
/// merely usually-agreeing — two independent subprocess runs of the same
/// scene and config, on the same adapter, must be byte-for-byte identical.
/// No comparison tolerance at all: this is what licenses `assert_matches_golden`'s
/// own tolerance (see its doc comment) to be read as "committed-golden noise
/// plus a margin," not "capture noise plus a margin."
#[test]
fn lockstep_capture_is_byte_identical_across_two_runs() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("determinism", scene, 1.5);

    let run_a = run_analyze("determinism-a", &iq, FRAMES_HOT);
    let run_b = run_analyze("determinism-b", &iq, FRAMES_HOT);
    std::fs::remove_file(&iq).ok();

    assert_software_adapter(&run_a);
    assert_software_adapter(&run_b);
    assert_eq!(
        adapter_of(&run_a),
        adapter_of(&run_b),
        "two runs on this machine rendered on different adapters — not a \
         meaningful determinism test"
    );
    assert_eq!(
        final_annotation_count(&run_a),
        final_annotation_count(&run_b)
    );

    let (img_a, wa, ha) = decode_rgba(&run_a.png, "run A PNG");
    let (img_b, wb, hb) = decode_rgba(&run_b.png, "run B PNG");
    assert_eq!((wa, ha), (wb, hb));
    assert_eq!(
        img_a, img_b,
        "two lockstep captures of the same scene and config produced \
         different pixels (NFR-A4): the capture must be byte-identical, not \
         merely close"
    );
}

/// D-111/D-112: the determinism proof above shares `run_analyze` with
/// `label_backing_plate_hides_the_band_edge_crossing_its_own_text`, but
/// sharing a call graph is a claim about the code, not the bytes — D-112
/// requires the assertion to run on the plate-pixel test's own scene, not be
/// inferred from a neighbour. Same scene construction as that test.
#[test]
fn plate_pixel_scene_capture_is_byte_identical_across_two_runs() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq = write_scene("plate-determinism", scene, 1.5);

    let run_a = run_analyze("plate-determinism-a", &iq, FRAMES_HOT);
    let run_b = run_analyze("plate-determinism-b", &iq, FRAMES_HOT);
    std::fs::remove_file(&iq).ok();

    assert_software_adapter(&run_a);
    assert_software_adapter(&run_b);
    assert_eq!(
        adapter_of(&run_a),
        adapter_of(&run_b),
        "two runs on this machine rendered on different adapters — not a \
         meaningful determinism test"
    );

    let (img_a, wa, ha) = decode_rgba(&run_a.png, "plate scene run A PNG");
    let (img_b, wb, hb) = decode_rgba(&run_b.png, "plate scene run B PNG");
    assert_eq!((wa, ha), (wb, hb));
    assert_eq!(
        img_a, img_b,
        "two lockstep captures of the plate-pixel test's own scene produced \
         different pixels (NFR-A4/D-112): the capture must be byte-identical"
    );
}

/// D-112/D-113's negative control for the plate-pixel scene: the
/// byte-identical comparison above has power there — it is not vacuously
/// true because two runs of this binary always happen to agree regardless of
/// content. Perturbs the **input** only (D-113: no production hook for
/// this), by dropping exactly one frame's worth of samples (`FFT` × 8 bytes,
/// cf32) from the front of run B's own copy of the IQ file — a real,
/// on-purpose difference in what run B reads, never "usually different" the
/// way swapping in the wall-clock worker would be. Every sample after the
/// cut shifts one frame earlier in the stream, which alone re-indexes the
/// noise floor's RNG sequence into different values from that point on, so
/// this does not depend on hitting the hopper's own phase in any particular
/// way.
#[test]
fn plate_pixel_determinism_check_has_power() {
    let mut cfg = noise_scene(2);
    cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
    let scene = SigGen::new(cfg).unwrap();
    let iq_a = write_scene("plate-power-a", scene, 1.5);

    let bytes = std::fs::read(&iq_a).expect("cannot read the IQ fixture");
    let frame_bytes = FFT * 8;
    assert!(
        bytes.len() > frame_bytes,
        "fixture too short to drop a frame from"
    );
    let iq_b = std::env::temp_dir().join(format!(
        "phosphene-analyze-golden-plate-power-b-{}.cf32",
        std::process::id()
    ));
    std::fs::write(&iq_b, &bytes[frame_bytes..]).expect("cannot write the perturbed fixture");

    let run_a = run_analyze("plate-power-a", &iq_a, FRAMES_HOT);
    let run_b = run_analyze("plate-power-b", &iq_b, FRAMES_HOT);
    std::fs::remove_file(&iq_a).ok();
    std::fs::remove_file(&iq_b).ok();

    let (img_a, _, _) = decode_rgba(&run_a.png, "plate power run A PNG");
    let (img_b, _, _) = decode_rgba(&run_b.png, "plate power run B PNG");
    assert_ne!(
        img_a, img_b,
        "dropping one frame's worth of samples from the IQ file produced a \
         byte-identical PNG (D-112/D-113): the determinism check has no \
         power to detect a real difference on this scene"
    );
}

/// D-114's own required measurement, kept as a reproducible test rather than
/// a one-off: does the smallest regression D-014's goldens must catch
/// (one label missing; a band moved by one label width) actually clear
/// `MEAN_LIMIT`/`OUTLIER_FRAC_LIMIT`? Answered directly against the
/// committed AC-2 golden, not estimated. `#[ignore]`d because it renders two
/// extra subprocesses beyond the normal suite and exists to document a
/// measurement, not to gate every run — see the PR for the numbers and the
/// decision they led to (D-114 item 2: no safe gap, so no pixel threshold
/// change was made past D-014's own 3.0/0.02).
///
/// (b)'s shift (313.7 kHz) is the backing plate's own measured width (87 px
/// at this 640-wide, ±1.024 MHz grid: 87 px × 2_048_000 Hz / 568 px of grid
/// width) — measured directly off the committed golden with a pixel scan
/// (reported in the PR), not assumed.
#[test]
#[ignore = "renders two extra subprocesses to measure a regression's own \
            footprint against the committed golden; run explicitly \
            (D-114)"]
fn d114_smallest_regressions_against_the_tolerance() {
    // (a) one label missing: AC-2's own scene, analyze on (matches the
    // committed golden) vs analyze off (AC-11: exactly v1's display, no
    // shade band, no label, no plate at all -- for a single-signal scene
    // this removes exactly the one label the golden has, and is if anything
    // a larger footprint than "label text only" would be, since the shade
    // band and halo go with it).
    {
        let mut cfg = noise_scene(2);
        cfg.hoppers.push(bursty_hopper(300_000.0, -20.0));
        let scene = SigGen::new(cfg).unwrap();
        let iq = write_scene("reg-a", scene, 1.5);
        let png_off = std::env::temp_dir().join("reg-a-off.png");
        let output = Command::new(env!("CARGO_BIN_EXE_phosphene"))
            .env("VK_ICD_FILENAMES", LAVAPIPE_ICD)
            .args([
                "--headless",
                "--source",
                &format!("file:{}", iq.display()),
                "--rate",
                &RATE_HZ.to_string(),
                "--format",
                "cf32",
                "--loop",
                "--frames",
                &FRAMES_HOT.to_string(),
                "--fft",
                &FFT.to_string(),
                "--width",
                &WIDTH.to_string(),
                "--height",
                &HEIGHT.to_string(),
                "--out",
            ])
            .arg(&png_off)
            .output()
            .expect("failed to spawn (analyze off)");
        assert!(output.status.success(), "analyze-off run failed");
        let golden = goldens_dir().join("analyze-ac2-one-signal.png");
        let (img_golden, w, h) = decode_rgba(&golden, "ac2 golden");
        let (img_off, w2, h2) = decode_rgba(&png_off, "analyze-off PNG");
        assert_eq!((w, h), (w2, h2));
        let mut sum = 0u64;
        let mut outliers = 0u64;
        let mut n = 0u64;
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let d: Vec<u64> = img_golden[o..o + 3]
                    .iter()
                    .zip(&img_off[o..o + 3])
                    .map(|(a, b)| a.abs_diff(*b) as u64)
                    .collect();
                sum += d.iter().sum::<u64>();
                if d.iter().copied().max().unwrap() > 32 {
                    outliers += 1;
                }
                n += 1;
            }
        }
        eprintln!(
            "(a) one label missing (analyze off vs golden): mean {:.4}, outlier_frac {:.5}",
            sum as f64 / (n as f64 * 3.0),
            outliers as f64 / n as f64
        );
        std::fs::remove_file(&iq).ok();
    }

    // (b) a band moved by one label width: AC-2's own scene, hopper shifted
    // by the label backing plate's own measured width (87 px at this
    // 640-wide, ±1.024 MHz grid -> 87 * (2_048_000 / 568) ~= 313.7 kHz),
    // measured directly off the committed golden (see the PR for the pixel
    // scan), vs the golden's own 300 kHz offset.
    {
        let shift_hz = 313_700.0;
        let mut cfg = noise_scene(2);
        cfg.hoppers.push(bursty_hopper(300_000.0 + shift_hz, -20.0));
        let scene = SigGen::new(cfg).unwrap();
        let iq = write_scene("reg-b", scene, 1.5);
        let run = run_analyze("reg-b", &iq, FRAMES_HOT);
        let golden = goldens_dir().join("analyze-ac2-one-signal.png");
        let (img_golden, w, h) = decode_rgba(&golden, "ac2 golden");
        let (img_shift, w2, h2) = decode_rgba(&run.png, "shifted-band PNG");
        assert_eq!((w, h), (w2, h2));
        let mut sum = 0u64;
        let mut outliers = 0u64;
        let mut n = 0u64;
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let d: Vec<u64> = img_golden[o..o + 3]
                    .iter()
                    .zip(&img_shift[o..o + 3])
                    .map(|(a, b)| a.abs_diff(*b) as u64)
                    .collect();
                sum += d.iter().sum::<u64>();
                if d.iter().copied().max().unwrap() > 32 {
                    outliers += 1;
                }
                n += 1;
            }
        }
        eprintln!(
            "(b) band moved by one label width (+{shift_hz}Hz vs golden): mean {:.4}, outlier_frac {:.5}",
            sum as f64 / (n as f64 * 3.0),
            outliers as f64 / n as f64
        );
        std::fs::remove_file(&iq).ok();
    }
}
