// SPDX-License-Identifier: MIT

//! NFR-P5 artifact-size report (clarification C7).
//!
//! C7 is explicit: the ≤15 MB binary target is a *measurement to report
//! honestly*, not a number to engineer around — "the number gets revised by
//! decision, not quietly missed. Do not strip or compress to hit a target and
//! call it met." So this command measures the release binaries exactly as
//! `cargo build --release` produces them and states the verdict either way.
//! It never fails the build: an over-target size is surfaced for a D-entry
//! decision, not silently gated (and not silently ignored — CI prints the
//! verdict into the job summary).

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::XtResult;

/// NFR-P5 says "≤15 MB"; measured in decimal megabytes (10^6 bytes), the
/// unit download sizes are quoted in. The MiB figure is printed alongside so
/// nobody has to wonder which convention was used.
const LIMIT_BYTES: u64 = 15_000_000;

/// Workspace bin targets that are dev tooling, not distributable artifacts.
const NON_DISTRIBUTABLE: &[&str] = &["xtask"];

pub fn report(root: &Path, target: Option<&str>) -> XtResult<()> {
    let bins = distributable_bins(root)?;
    let mut summary = String::from("## Artifact size vs NFR-P5 (≤15 MB) — clarification C7\n\n");

    if bins.is_empty() {
        let msg = "No distributable binary target exists in the workspace yet \
                   (`phosphene-app` lands with lane M0-C). The NFR-P5 verdict is \
                   deferred until there is a real artifact to measure — this report \
                   runs per-commit so the number appears the moment it exists.";
        println!("{msg}");
        summary.push_str(msg);
        summary.push('\n');
        write_step_summary(&summary);
        return Ok(());
    }

    build_release(root, target)?;

    let release_dir = match target {
        Some(t) => root.join("target").join(t).join("release"),
        None => root.join("target/release"),
    };
    let platform = target.unwrap_or("host");
    summary.push_str(&format!(
        "Target: `{platform}`\n\n| binary | size | NFR-P5 ≤15 MB |\n|---|---|---|\n"
    ));
    for bin in &bins {
        let path = release_dir.join(bin);
        let bytes = fs::metadata(&path)?.len();
        let mb = bytes as f64 / 1e6;
        let mib = bytes as f64 / (1024.0 * 1024.0);
        let verdict = if bytes <= LIMIT_BYTES {
            "PASS"
        } else {
            "**OVER — needs a D-entry**"
        };
        let line = format!("| `{bin}` | {mb:.1} MB ({mib:.1} MiB, {bytes} bytes) | {verdict} |");
        println!("{line}");
        summary.push_str(&line);
        summary.push('\n');
        if bytes > LIMIT_BYTES {
            // GitHub Actions warning annotation — visible without failing the
            // job (C7: revised by decision, not gated here).
            println!(
                "::warning title=NFR-P5::{bin} is {mb:.1} MB (> 15 MB target); \
                 revise by D-entry, do not strip/compress to hit the number"
            );
        }
    }
    write_step_summary(&summary);
    Ok(())
}

/// Bin targets of workspace members, minus dev tooling — the set NFR-P5
/// actually constrains. Read from `cargo metadata` so this never has to know
/// crate internals.
fn distributable_bins(root: &Path) -> XtResult<Vec<String>> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .current_dir(root)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let meta: Value = serde_json::from_slice(&out.stdout)?;
    let mut bins = Vec::new();
    for pkg in meta["packages"]
        .as_array()
        .ok_or("cargo metadata has no packages")?
    {
        let pkg_name = pkg["name"].as_str().unwrap_or_default();
        if NON_DISTRIBUTABLE.contains(&pkg_name) {
            continue;
        }
        for t in pkg["targets"].as_array().into_iter().flatten() {
            let kinds = t["kind"].as_array().cloned().unwrap_or_default();
            if kinds.iter().any(|k| k == "bin") {
                if let Some(name) = t["name"].as_str() {
                    bins.push(name.to_owned());
                }
            }
        }
    }
    bins.sort();
    Ok(bins)
}

fn build_release(root: &Path, target: Option<&str>) -> XtResult<()> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root)
        .args(["build", "--release", "--workspace"]);
    if let Some(t) = target {
        cmd.args(["--target", t]);
    }
    let status = cmd.status()?;
    if !status.success() {
        return Err(format!("cargo build --release failed with {status}").into());
    }
    Ok(())
}

/// Append to the GitHub Actions job summary when running in CI; a no-op
/// locally.
fn write_step_summary(text: &str) {
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        if let Ok(mut f) = fs::OpenOptions::new().append(true).create(true).open(path) {
            let _ = writeln!(f, "{text}");
        }
    }
}
