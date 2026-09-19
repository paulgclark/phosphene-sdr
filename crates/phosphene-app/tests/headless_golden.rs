// SPDX-License-Identifier: MIT

//! The app-level golden for the DEFAULT headless render (D-049, protecting
//! D-014): run the real binary with `--source` absent at fixed parameters
//! and compare the PNG perceptually against `tests/goldens/`. M1 exited on
//! D-014's gate, and M2-A changed the code path that produces this render —
//! this test is what detects the default drifting.
//!
//! The comparison is perceptual, never byte-exact: GPU rasterisation
//! differs across the four CI legs (lavapipe/Vulkan vs Metal), which is
//! D-014's original reasoning. The tolerance also absorbs the status bar's
//! `frame` readout — the one deliberately nondeterministic value on the
//! chrome (measured wall time, per the M0-C seal wording); its handful of
//! digit glyphs stays far inside the outlier budget while a wrong scene,
//! colormap, layout or surface fails loudly.
//!
//! Regenerate deliberately with `PHOSPHENE_UPDATE_GOLDENS=1 cargo test`,
//! then eyeball the committed PNG before committing it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Fixed render parameters: the golden is deterministic for a given
/// rasterizer only because every input is pinned here.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 120;
const FFT: u32 = 512;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join("headless-default.png")
}

/// Decode a PNG to tightly-packed RGBA8, asserting the pinned dimensions.
fn decode_rgba(path: &Path, what: &str) -> Vec<u8> {
    let file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {what} {path:?} ({e})"));
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .unwrap_or_else(|e| panic!("{what} {path:?} unreadable: {e}"));
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG too large")];
    let info = reader
        .next_frame(&mut buf)
        .unwrap_or_else(|e| panic!("{what} {path:?} frame unreadable: {e}"));
    assert_eq!(
        (info.width, info.height),
        (WIDTH, HEIGHT),
        "{what} has the wrong size"
    );
    assert_eq!(info.color_type, png::ColorType::Rgba, "{what} color type");
    buf.truncate((WIDTH * HEIGHT * 4) as usize);
    buf
}

/// D-014's invariant, sealed on the production path end to end: no
/// `--source` renders the unchanged deterministic scene, asserted against
/// the committed golden — through the real binary, so the CLI dispatch is
/// inside the seal, not beside it (D-031).
#[test]
fn default_headless_render_matches_the_committed_golden() {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("headless-default.png");
    let status = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .args([
            "--headless",
            "--frames",
            &FRAMES.to_string(),
            "--fft",
            &FFT.to_string(),
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
            "--out",
        ])
        .arg(&out)
        .status()
        .expect("failed to spawn the phosphene binary");
    assert!(
        status.success(),
        "phosphene --headless exited with {status}"
    );
    let img = decode_rgba(&out, "rendered PNG");

    let golden = golden_path();
    if std::env::var_os("PHOSPHENE_UPDATE_GOLDENS").is_some() {
        // The binary already wrote exactly the bytes to seal; adopt them as
        // the new golden verbatim.
        std::fs::create_dir_all(golden.parent().unwrap()).unwrap();
        std::fs::copy(&out, &golden).expect("cannot write the golden");
        return;
    }
    let golden_img = decode_rgba(&golden, "golden (generate with PHOSPHENE_UPDATE_GOLDENS=1)");

    // Perceptual gate, duplicated deliberately from
    // `crates/phosphene-render/tests/surface.rs::assert_matches_golden` —
    // that copy is the source of truth for the constants; the render
    // harness lives in another crate's `tests/` and is not importable, and
    // D-049 records the shared-helper extraction as carried debt. Mean
    // per-channel error and an outlier budget, both sized to absorb
    // rasteriser differences while failing loudly on a wrong scene.
    let mut sum = 0u64;
    let mut outliers = 0u64;
    let mut n = 0u64;
    for (a, b) in img
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden_img.as_chunks::<4>().0)
    {
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
    let mean = sum as f64 / (n as f64 * 3.0);
    let outlier_frac = outliers as f64 / n as f64;
    assert!(
        mean <= 3.0 && outlier_frac <= 0.02,
        "the default headless render deviates from its golden (D-014/D-049): \
         mean abs diff {mean:.2} (limit 3.0), outlier fraction \
         {outlier_frac:.4} (limit 0.02)"
    );
}
