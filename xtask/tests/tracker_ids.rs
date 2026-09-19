// SPDX-License-Identifier: MIT

//! Behavioural tests for `cargo run -p xtask -- tracker-ids` (SC-2). Each
//! test builds its own throwaway git repository under the OS temp dir and
//! copies in the committed fixture at `fixtures/tracker-id-planted.txt`,
//! which contains a planted citation-shaped id on purpose. This file
//! assembles that id from separate pieces at runtime (see `planted_id`)
//! rather than spelling it out, so this test source does not itself match
//! the shape the check under test guards.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE: &str = include_str!("fixtures/tracker-id-planted.txt");

/// The id planted in the fixture, assembled from its letter and its digits
/// separately so this source file never spells the shape out as one
/// contiguous token.
fn planted_id() -> String {
    let letter = "F";
    let digits = "123";
    format!("{letter}-{digits}")
}

fn unique_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "phosphene-tracker-ids-test-{tag}-{}-{nanos}-{n}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git is on PATH");
    assert!(
        out.status.success(),
        "git {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn init_repo() -> PathBuf {
    let dir = unique_dir("repo");
    git(&dir, &["init", "--quiet", "--initial-branch=main"]);
    git(
        &dir,
        &["config", "user.email", "tracker-ids-test@example.invalid"],
    );
    git(&dir, &["config", "user.name", "tracker-ids test"]);
    dir
}

fn commit_all(dir: &Path, message: &str) {
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "--quiet", "-m", message]);
}

fn run_tracker_ids(cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("tracker-ids")
        .current_dir(cwd)
        .output()
        .expect("xtask binary runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("process exits with a code")
}

// The fixture's planted id fails the check, reported as file:line: id.
#[test]
fn fixture_with_planted_id_fails() {
    let repo = init_repo();
    fs::write(repo.join("planted.txt"), FIXTURE).unwrap();
    commit_all(&repo, "add fixture with a planted tracker id");

    let out = run_tracker_ids(&repo);
    assert_eq!(code(&out), 1, "stdout:\n{}", stdout(&out));
    let text = stdout(&out);
    let expected = format!("planted.txt:2: {}", planted_id());
    assert!(text.contains(&expected), "got:\n{text}");
}

// A clean tree (no citation-shaped ids) passes.
#[test]
fn clean_tree_passes() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "nothing to see here\n").unwrap();
    commit_all(&repo, "clean file");

    let out = run_tracker_ids(&repo);
    assert_eq!(code(&out), 0, "stdout:\n{}", stdout(&out));
}

// A file under xtask/tests/fixtures/ is excluded from the scan even when
// it plants a matching id — this check's own fixtures are not its target.
#[test]
fn own_fixtures_directory_is_excluded() {
    let repo = init_repo();
    fs::create_dir_all(repo.join("xtask/tests/fixtures")).unwrap();
    fs::write(repo.join("xtask/tests/fixtures/planted.txt"), FIXTURE).unwrap();
    commit_all(&repo, "add a file under the fixtures directory");

    let out = run_tracker_ids(&repo);
    assert_eq!(code(&out), 0, "stdout:\n{}", stdout(&out));
}
