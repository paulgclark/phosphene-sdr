// SPDX-License-Identifier: MIT

//! `--headless --source` end to end (D-041 / M2-A): run the real binary over
//! a short raw IQ capture and assert the whole contract at the process
//! boundary — a clean exit, a full-size PNG, and the honest stderr report of
//! the frames rendered vs requested when the capture ends first. The
//! rendered-content and axis-labelling seals live beside the loop itself
//! (`src/headless.rs`); this test seals the CLI dispatch and the report.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn short_capture_renders_exits_cleanly_and_reports_the_shortfall() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let iq = dir.join("short-capture.cf32");
    let out = dir.join("short-capture.png");
    let (width, height, fft) = (320u32, 200u32, 512usize);
    // Ten whole batches of cf32 silence — content is irrelevant here; the
    // subject is termination and the report.
    std::fs::write(&iq, vec![0u8; 10 * fft * 8]).expect("cannot write the capture");

    let output = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .args([
            "--headless",
            "--frames",
            "600",
            "--fft",
            &fft.to_string(),
            "--width",
            &width.to_string(),
            "--height",
            &height.to_string(),
        ])
        .arg("--source")
        .arg(format!("file:{}", iq.display()))
        .arg("--out")
        .arg(&out)
        .output()
        .expect("failed to spawn the phosphene binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "a capture shorter than --frames must exit cleanly, got {}: {stderr}",
        output.status
    );

    // The shortfall is said out loud on stderr: rendered vs requested.
    assert!(
        stderr.contains("of 600 requested frames"),
        "stderr must report frames rendered vs requested, got: {stderr}"
    );
    assert!(
        stderr.contains("source ended before --frames 600"),
        "stderr must say the source ended early, got: {stderr}"
    );
    // And the D-013 accounting is surfaced, not recomputed: 10 whole
    // batches of 512 samples were consumed.
    assert!(
        stderr.contains("5120 samples consumed in 10 FFTs"),
        "stderr must report the samples consumed, got: {stderr}"
    );

    // The PNG really exists at full size.
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(&out).expect("PNG missing"),
    ));
    let mut reader = decoder.read_info().expect("PNG unreadable");
    let mut buf = vec![0; reader.output_buffer_size().expect("PNG too large")];
    let info = reader.next_frame(&mut buf).expect("PNG frame unreadable");
    assert_eq!((info.width, info.height), (width, height));
}

#[test]
fn missing_capture_is_an_actionable_error_not_a_panic() {
    let output = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .args([
            "--headless",
            "--frames",
            "1",
            "--source",
            "file:/nonexistent/capture.cf32",
        ])
        .output()
        .expect("failed to spawn the phosphene binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "a missing capture must fail");
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
    assert!(
        stderr.contains("capture.cf32"),
        "the error must name the offending path, got: {stderr}"
    );
}
