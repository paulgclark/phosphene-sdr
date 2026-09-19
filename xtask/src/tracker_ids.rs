// SPDX-License-Identifier: MIT

//! `cargo run -p xtask -- tracker-ids` — an internal-citation-ID hygiene
//! check: every tracked file, from the repository's top level regardless
//! of the current directory, scanned for a `LETTER-digits` citation shape.
//!
//! Unlike `scrub`, the shape this check looks for is public syntax, not a
//! secret, so it needs no external pattern file and is safe to commit
//! outright — the check itself is the record of what it guards.
//!
//! Reports `file:line: id` for every match. Exit 1 on any match, exit 0 on
//! a clean scan, exit 2 when the scan itself could not be established (not
//! a git repository, zero tracked files, or a tracked file that cannot be
//! read) — fails closed, the same posture as `scrub`.
//!
//! This check's own test fixtures (`xtask/tests/fixtures/`) are excluded
//! from the scan: they exist to contain a planted id on purpose, and
//! excluding a tool's own fixtures from its own scan is ordinary test
//! hygiene, not a loophole in what the check guards elsewhere in the tree.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use regex::Regex;

/// The citation-ID shape this check guards: a single capital `F` or `H`, a
/// hyphen, one to four digits, as a whole word.
const ID_SHAPE: &str = r"\b[FH]-\d{1,4}\b";

/// This check's own fixtures directory — excluded from the scan (see the
/// module docs).
const FIXTURES_PREFIX: &str = "xtask/tests/fixtures/";

pub fn run() -> ExitCode {
    let re = Regex::new(ID_SHAPE).expect("ID_SHAPE is a fixed, valid pattern");

    let Some(toplevel) = repo_toplevel() else {
        eprintln!(
            "tracker-ids: could not determine the repository's top level (not a git repository?)"
        );
        return ExitCode::from(2);
    };

    let Some(files) = tracked_files(&toplevel) else {
        eprintln!("tracker-ids: could not list tracked files");
        return ExitCode::from(2);
    };

    if files.is_empty() {
        eprintln!(
            "tracker-ids: zero tracked files scanned — an empty scan cannot be told apart from a clean tree"
        );
        return ExitCode::from(2);
    }

    let mut findings = Vec::new();
    let mut content_scanned = 0usize;

    for rel_path in &files {
        if rel_path.starts_with(FIXTURES_PREFIX) {
            continue;
        }
        let bytes = match fs::read(toplevel.join(rel_path)) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("tracker-ids: {rel_path} could not be read");
                return ExitCode::from(2);
            }
        };
        if bytes.contains(&0) {
            continue;
        }
        content_scanned += 1;
        let text = String::from_utf8_lossy(&bytes);
        for (lineno, line) in text.lines().enumerate() {
            for m in re.find_iter(line) {
                findings.push(format!("{rel_path}:{}: {}", lineno + 1, m.as_str()));
            }
        }
    }

    if !findings.is_empty() {
        for f in &findings {
            println!("{f}");
        }
        return ExitCode::from(1);
    }

    println!("tracker-ids: clean — {content_scanned} files scanned");
    ExitCode::SUCCESS
}

fn repo_toplevel() -> Option<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        return None;
    }
    Some(PathBuf::from(path))
}

fn tracked_files(toplevel: &Path) -> Option<Vec<String>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(toplevel)
        .args(["ls-files", "-z"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.split('\0')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
    )
}
