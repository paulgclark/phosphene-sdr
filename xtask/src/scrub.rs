// SPDX-License-Identifier: MIT

//! `cargo run -p xtask -- scrub` (PS-1 / D-082, fix pass D-083).
//!
//! Two mutually exclusive modes, both against a private pattern list named
//! by `PHOSPHENE_SCRUB_PATTERNS_FILE` (one `regex`-syntax pattern per line,
//! blank and `#`-comment lines ignored, matched case-insensitively and as a
//! whole word):
//!
//! * **Tree mode** (default): every tracked file's content and path, from
//!   the repository's top level regardless of the current directory (D-083
//!   §4).
//! * **Messages mode** (`--messages <rev-range>`): only the commit messages
//!   in that range — it does not read the tree at all (D-083 §1).
//!
//! The check fails closed: anything that stops it from establishing a real
//! answer — the variable unset, the list missing or unreadable, no usable
//! pattern, a pattern that won't compile, a tracked file that cannot be
//! read (D-083 §3), or a scan that covered zero tracked files / zero
//! commits — is exit 2, distinct from "found nothing" (exit 0). A result
//! never repeats what it guards: a match names a file, line and pattern
//! number only, never the matched text or the pattern — and a tracked
//! *path* that is itself the leak is never printed either, literally or in
//! an error message; it is named by its position in `git ls-files` order
//! instead (D-083 §2).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use regex::{Regex, RegexBuilder};

const PATTERNS_FILE_VAR: &str = "PHOSPHENE_SCRUB_PATTERNS_FILE";

struct Pattern {
    /// 1-based position among the *usable* patterns in the list — the `#N`
    /// a match report names.
    index: usize,
    re: Regex,
}

pub fn run(messages_range: Option<&str>) -> ExitCode {
    let patterns = match load_patterns() {
        Ok(p) => p,
        Err(code) => return code,
    };

    let Some(toplevel) = repo_toplevel() else {
        eprintln!("scrub: could not determine the repository's top level (not a git repository?)");
        return ExitCode::from(2);
    };

    match messages_range {
        Some(range) => run_messages_mode(&toplevel, &patterns, range),
        None => run_tree_mode(&toplevel, &patterns),
    }
}

fn run_messages_mode(toplevel: &Path, patterns: &[Pattern], range: &str) -> ExitCode {
    let Some(commits) = commit_messages(toplevel, range) else {
        eprintln!("scrub: could not read commit messages for {range}");
        return ExitCode::from(2);
    };

    if commits.is_empty() {
        eprintln!(
            "scrub: zero commits in {range} — an empty scan cannot be told apart from a clean range"
        );
        return ExitCode::from(2);
    }

    let mut findings = Vec::new();
    for (sha, message) in &commits {
        for p in patterns {
            if p.re.is_match(message) {
                findings.push(format!("commit {sha}: pattern #{}", p.index));
            }
        }
    }

    if !findings.is_empty() {
        for f in &findings {
            println!("{f}");
        }
        return ExitCode::from(1);
    }

    println!(
        "scrub: clean — {} commits scanned, {} patterns loaded",
        commits.len(),
        patterns.len()
    );
    ExitCode::SUCCESS
}

fn run_tree_mode(toplevel: &Path, patterns: &[Pattern]) -> ExitCode {
    let Some(files) = tracked_files(toplevel) else {
        eprintln!("scrub: could not list tracked files");
        return ExitCode::from(2);
    };

    if files.is_empty() {
        eprintln!(
            "scrub: zero tracked files scanned — an empty scan cannot be told apart from a clean tree"
        );
        return ExitCode::from(2);
    }

    let mut findings = Vec::new();
    let mut content_scanned = 0usize;
    let mut binaries_skipped = 0usize;

    for (i, rel_path) in files.iter().enumerate() {
        // 1-based position in `git ls-files` order (D-083 §2) — the label
        // used for this file's findings when its own path is the leak, so
        // the literal path is never the thing that prints it.
        let ordinal = i + 1;
        let path_is_leak = patterns.iter().any(|p| p.re.is_match(rel_path));

        if path_is_leak {
            for p in patterns {
                if p.re.is_match(rel_path) {
                    findings.push(format!("tracked path #{ordinal}: pattern #{}", p.index));
                }
            }
        }

        let bytes = match fs::read(toplevel.join(rel_path)) {
            Ok(b) => b,
            Err(_) => {
                // D-083 §3: unreadable is exit 2, never a skip. Abort
                // without printing any findings collected so far — the scan
                // is incomplete, which is "cannot determine", not a result.
                if path_is_leak {
                    eprintln!("scrub: tracked path #{ordinal} could not be read");
                } else {
                    eprintln!("scrub: {rel_path} could not be read");
                }
                return ExitCode::from(2);
            }
        };
        if bytes.contains(&0) {
            binaries_skipped += 1;
            continue;
        }
        content_scanned += 1;
        let text = String::from_utf8_lossy(&bytes);
        for (lineno, line) in text.lines().enumerate() {
            for p in patterns {
                if p.re.is_match(line) {
                    findings.push(if path_is_leak {
                        format!(
                            "tracked path #{ordinal}:{}: pattern #{}",
                            lineno + 1,
                            p.index
                        )
                    } else {
                        format!("{rel_path}:{}: pattern #{}", lineno + 1, p.index)
                    });
                }
            }
        }
    }

    if !findings.is_empty() {
        for f in &findings {
            println!("{f}");
        }
        return ExitCode::from(1);
    }

    println!(
        "scrub: clean — {content_scanned} files scanned, {} paths scanned, \
         {binaries_skipped} binaries skipped, {} patterns loaded",
        files.len(),
        patterns.len()
    );
    ExitCode::SUCCESS
}

fn load_patterns() -> Result<Vec<Pattern>, ExitCode> {
    let path = match env::var(PATTERNS_FILE_VAR) {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("scrub: {PATTERNS_FILE_VAR} is not set");
            return Err(ExitCode::from(2));
        }
    };

    let raw = fs::read_to_string(&path).map_err(|e| {
        eprintln!("scrub: cannot read pattern list: {e}");
        ExitCode::from(2)
    })?;

    let mut patterns = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let wrapped = format!(r"\b(?:{line})\b");
        match RegexBuilder::new(&wrapped).case_insensitive(true).build() {
            Ok(re) => patterns.push(Pattern {
                index: patterns.len() + 1,
                re,
            }),
            Err(_) => {
                eprintln!("scrub: pattern list line {}: does not compile", lineno + 1);
                return Err(ExitCode::from(2));
            }
        }
    }

    if patterns.is_empty() {
        eprintln!("scrub: no usable pattern in the list (empty, or only comments/blanks)");
        return Err(ExitCode::from(2));
    }

    Ok(patterns)
}

/// The repository's top level, resolved from wherever the process happens
/// to be running (D-083 §4) — so a run from a subdirectory still scans the
/// whole tree instead of silently narrowing to that subdirectory.
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

fn commit_messages(toplevel: &Path, range: &str) -> Option<Vec<(String, String)>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(toplevel)
        .args(["log", range, "--format=%H%x1f%B%x1e"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.split('\u{1e}')
            .filter(|s| !s.trim().is_empty())
            .filter_map(|record| {
                let mut parts = record.splitn(2, '\u{1f}');
                let sha = parts.next()?.trim().to_string();
                let message = parts.next()?.to_string();
                Some((sha, message))
            })
            .collect(),
    )
}
