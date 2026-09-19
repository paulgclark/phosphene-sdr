// SPDX-License-Identifier: MIT

//! The LC-2 licence gate must demonstrably FAIL on a GPL crate (lane M0-D:
//! "a gate nobody has seen fail is not known to work"). This test generates a
//! throwaway workspace in a temp dir containing an *empty* canary crate whose
//! manifest declares `license = "GPL-3.0-only"`, points cargo-deny at it with
//! the repo's real `deny.toml`, and asserts the check fails. A second run with
//! the canary relicensed to MIT must pass — proving the failure was caused by
//! the licence, not by broken plumbing.
//!
//! No GPL code is involved at any point: the canary is a generated empty
//! `lib.rs` that exists only for the duration of the test, and only its
//! metadata claims GPL. Nothing GPL-tagged is ever committed to the repo
//! (LC-2 forbids that too).

use std::fs;
use std::path::Path;
use std::process::Command;

/// Build the throwaway workspace: root package depending on a path-only
/// canary crate with the given licence, plus a copy of the repo's deny.toml.
fn write_fixture(dir: &Path, canary_license: &str) {
    let canary = dir.join("canary");
    fs::create_dir_all(canary.join("src")).unwrap();
    fs::write(
        canary.join("Cargo.toml"),
        format!(
            "[package]\nname = \"lc2-canary\"\nversion = \"0.0.0\"\n\
             edition = \"2021\"\nlicense = \"{canary_license}\"\n"
        ),
    )
    .unwrap();
    fs::write(canary.join("src/lib.rs"), "// empty on purpose\n").unwrap();

    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        // The empty [workspace] table keeps the fixture out of the real
        // workspace (it lives under target/, inside the repo).
        "[package]\nname = \"lc2-fixture\"\nversion = \"0.0.0\"\n\
         edition = \"2021\"\nlicense = \"MIT\"\n\n\
         [workspace]\n\n\
         [dependencies]\nlc2-canary = { path = \"canary\" }\n",
    )
    .unwrap();
    fs::write(dir.join("src/lib.rs"), "// empty on purpose\n").unwrap();

    let repo_deny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../deny.toml");
    fs::copy(repo_deny, dir.join("deny.toml")).unwrap();
}

/// Run `cargo deny check licenses` against the fixture; returns the exit
/// success flag and combined output.
fn run_deny(dir: &Path) -> (bool, String) {
    let out = Command::new("cargo-deny")
        .current_dir(dir)
        .args(["check", "licenses"])
        .output()
        .expect("cargo-deny is installed (checked by the caller)");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn cargo_deny_available() -> bool {
    Command::new("cargo-deny")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn lc2_gate_fails_on_gpl_and_passes_on_mit() {
    if !cargo_deny_available() {
        // The gate itself still runs in the CI licence job, which installs
        // cargo-deny; skipping here only skips the local self-test.
        eprintln!("SKIP: cargo-deny not on PATH; install it to run the LC-2 gate self-test");
        return;
    }

    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("lc2-gate");
    let _ = fs::remove_dir_all(&base);

    // The gate must FAIL on a GPL-licensed dependency.
    let gpl = base.join("gpl");
    write_fixture(&gpl, "GPL-3.0-only");
    let (ok, output) = run_deny(&gpl);
    assert!(
        !ok,
        "LC-2 gate DID NOT FAIL on a GPL-3.0-only dependency — the gate is not working:\n{output}"
    );
    assert!(
        output.contains("GPL-3.0-only"),
        "gate failed but not for the GPL licence:\n{output}"
    );

    // Control: the identical fixture under MIT must PASS, proving the
    // failure above was licence-caused rather than fixture breakage.
    let mit = base.join("mit");
    write_fixture(&mit, "MIT");
    let (ok, output) = run_deny(&mit);
    assert!(ok, "control fixture (MIT) unexpectedly failed:\n{output}");
}
