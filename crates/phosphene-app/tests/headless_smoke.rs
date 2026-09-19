// SPDX-License-Identifier: MIT

//! Headless smoke render (the M0-C seal item M0-D's CI golden consumes):
//! run the real binary offscreen, decode the PNG, and sanity-check the
//! D-010 ground rules — dark ground, luminous trace — without asserting
//! exact pixels (that is M0-D's perceptual-diff job).

use std::path::PathBuf;
use std::process::Command;

#[test]
fn headless_render_produces_a_dark_luminous_png() {
    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let out = out_dir.join("smoke.png");
    let (width, height) = (640u32, 360u32);

    let status = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .args([
            "--headless",
            "--frames",
            "30",
            "--width",
            &width.to_string(),
            "--height",
            &height.to_string(),
            "--out",
        ])
        .arg(&out)
        .status()
        .expect("failed to spawn the phosphene binary");
    assert!(
        status.success(),
        "phosphene --headless exited with {status}"
    );

    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(&out).expect("PNG missing"),
    ));
    let mut reader = decoder.read_info().expect("PNG unreadable");
    let mut buf = vec![0; reader.output_buffer_size().expect("PNG too large")];
    let info = reader.next_frame(&mut buf).expect("PNG frame unreadable");
    assert_eq!((info.width, info.height), (width, height));
    assert_eq!(info.color_type, png::ColorType::Rgba);
    let rgba = &buf[..info.buffer_size()];

    // Dark ground (D-010): the mean luminance of the frame stays low.
    let mut sum: u64 = 0;
    let mut brightest: u8 = 0;
    for px in rgba.as_chunks::<4>().0 {
        let luma = px[0].max(px[1]).max(px[2]);
        sum += u64::from(luma);
        brightest = brightest.max(luma);
    }
    let mean = sum / (u64::from(width) * u64::from(height));
    // The bound is 64, raised from 48 by D-065 §2 — deliberately, with the
    // measurement written down rather than nudged until it passed.
    //
    // Auto-range is the waterfall's default now, and at this test's 30
    // frames the panel is only partly filled, so the few rows it holds are
    // stretched across the whole colormap: this frame measures ~48, where
    // 48 was the old ceiling. The display as a whole did **not** get
    // brighter — the committed 120-frame golden measures ~48 against ~56
    // before this lane, because the D-065 §5 range change darkens the
    // histogram floor. 64 keeps a real D-010 guard (a washed-out or
    // light-ground frame fails it by a wide margin) with headroom over both
    // figures.
    assert!(mean < 64, "frame is not dark: mean luminance {mean}");
    // Luminous trace: something in the frame approaches full brightness.
    assert!(
        brightest > 200,
        "no luminous content: max luminance {brightest}"
    );

    // The trace hue is the p7-family yellow-green: the brightest pixels
    // should be green-dominant rather than grey.
    let mut green_hot = 0u32;
    for px in rgba.as_chunks::<4>().0 {
        if px[1] > 200 && px[1] > px[2] {
            green_hot += 1;
        }
    }
    assert!(
        green_hot > 100,
        "expected a visible yellow-green trace, found {green_hot} hot green pixels"
    );
}
