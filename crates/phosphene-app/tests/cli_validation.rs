// SPDX-License-Identifier: MIT

//! Headless CLI argument validation: `--headless` is the scripted/CI surface
//! (FR-C4), so bad arguments must produce an actionable error naming the
//! offending value — never a panic. Sealed on both sides of zero: zero values
//! hit our validation, negative values are rejected by clap's `u32` parsing.

use std::process::Command;

/// Run the binary headless with the given extra args; return (success, stderr).
fn run_headless(extra: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_phosphene"))
        .arg("--headless")
        .args(extra)
        .output()
        .expect("failed to spawn the phosphene binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn zero_frames_is_an_error_not_a_panic() {
    let (ok, stderr) = run_headless(&["--frames", "0"]);
    assert!(!ok, "zero frames must fail");
    assert!(
        stderr.contains("--frames") && stderr.contains("0"),
        "error must name the offending value, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
}

#[test]
fn zero_width_is_an_error_not_a_panic() {
    let (ok, stderr) = run_headless(&["--frames", "1", "--width", "0"]);
    assert!(!ok, "zero width must fail");
    assert!(
        stderr.contains("--width") && stderr.contains("0"),
        "error must name the offending value, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
}

#[test]
fn zero_height_is_an_error_not_a_panic() {
    let (ok, stderr) = run_headless(&["--frames", "1", "--height", "0"]);
    assert!(!ok, "zero height must fail");
    assert!(
        stderr.contains("--height") && stderr.contains("0"),
        "error must name the offending value, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
}

#[test]
fn negative_values_are_rejected_at_parse() {
    for flag in ["--frames", "--width", "--height"] {
        let (ok, stderr) = run_headless(&[flag, "-1"]);
        assert!(!ok, "{flag} -1 must fail");
        assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
        assert!(
            !stderr.is_empty(),
            "{flag} -1 must produce a diagnostic on stderr"
        );
    }
}

#[test]
fn invalid_fft_size_is_an_error_not_a_panic() {
    let (ok, stderr) = run_headless(&["--frames", "1", "--fft", "1000"]);
    assert!(!ok, "non-power-of-two FFT size must fail");
    assert!(
        stderr.contains("--fft") && stderr.contains("1000"),
        "error must name the flag and the offending value, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
    // Usage errors are validated before any GPU work, so this diagnostic
    // must never be masked by an adapter failure on GPU-less machines.
    assert!(
        !stderr.contains("adapter"),
        "usage error must not depend on GPU availability, got: {stderr}"
    );
}
