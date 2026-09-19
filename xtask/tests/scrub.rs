// SPDX-License-Identifier: MIT

//! Behavioural tests for `cargo run -p xtask -- scrub` (PS-1 / D-082, fix
//! pass D-083). Each test builds its own throwaway git repository under the
//! OS temp dir and uses invented tokens only — nothing from the real,
//! private pattern list is read, referenced or planted here.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const PATTERNS_VAR: &str = "PHOSPHENE_SCRUB_PATTERNS_FILE";

/// A patterns file using only invented tokens, never anything from the real
/// list. `zqonkfrelb` and `wibbleplarn` are the two planted tokens.
const TEST_PATTERNS: &str = "\
# invented-token test list — not the real pattern list\n\
zqonkfrelb\n\
wibbleplarn\n\
";

fn unique_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "phosphene-scrub-test-{tag}-{}-{nanos}-{n}",
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
        &["config", "user.email", "scrub-test@example.invalid"],
    );
    git(&dir, &["config", "user.name", "scrub test"]);
    dir
}

fn commit_all(dir: &Path, message: &str) {
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "--quiet", "-m", message]);
}

fn write_patterns_file(contents: &str) -> PathBuf {
    let dir = unique_dir("patterns");
    let file = dir.join("patterns.txt");
    fs::write(&file, contents).unwrap();
    file
}

/// Run the built `xtask` binary's `scrub` subcommand with `cwd` as its
/// working directory (not necessarily the repository's top level — see the
/// `l_*` test, which relies on that).
fn run_scrub(cwd: &Path, patterns_file: Option<&Path>, extra_args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("scrub").args(extra_args).current_dir(cwd);
    cmd.env_remove(PATTERNS_VAR);
    if let Some(p) = patterns_file {
        cmd.env(PATTERNS_VAR, p);
    }
    cmd.output().expect("xtask binary runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("process exits with a code")
}

// (a) planted token in file content -> 1; output has path:line and a
// pattern number; output does NOT contain the token.
#[test]
fn a_content_match() {
    let repo = init_repo();
    fs::write(repo.join("notes.txt"), "before\nzqonkfrelb\nafter\n").unwrap();
    commit_all(&repo, "add notes");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 1, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("notes.txt:2: pattern #1"), "got:\n{text}");
    assert!(
        !text.contains("zqonkfrelb"),
        "output repeated the token:\n{text}"
    );
}

// (b, changed by D-083 §2) planted token only in a tracked path -> 1; the
// path is never printed (it IS the leak) — reported as `tracked path #N`
// by its position in `git ls-files` order instead.
#[test]
fn b_path_match_is_never_printed() {
    let repo = init_repo();
    fs::write(
        repo.join("zqonkfrelb-notes.txt"),
        "nothing interesting here\n",
    )
    .unwrap();
    commit_all(&repo, "add file with token in its path");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 1, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);
    // Exactly one tracked file, so it is position #1.
    assert!(text.contains("tracked path #1: pattern #1"), "got:\n{text}");
    assert!(
        !text.contains("zqonkfrelb"),
        "output repeated the token via the literal path:\n{text}"
    );
}

// (c, changed by D-083 §1) planted token only in a commit message, under
// the now commit-messages-only `--messages` mode -> 1.
#[test]
fn c_commit_message_match() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "base\n").unwrap();
    commit_all(&repo, "base commit");
    fs::write(repo.join("b.txt"), "second\n").unwrap();
    commit_all(&repo, "mentions wibbleplarn in passing");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &["--messages", "HEAD~1..HEAD"]);
    assert_eq!(code(&out), 1, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("pattern #2"), "got:\n{text}");
    assert!(text.contains("commit "), "got:\n{text}");
    assert!(
        !text.contains("wibbleplarn"),
        "output repeated the token:\n{text}"
    );
}

// (c2, new — D-083 §1) the tree holds a token but the commit messages in
// range are clean, under `--messages` -> 0: the mode never reads the tree.
#[test]
fn c2_messages_mode_never_reads_the_tree() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "base\n").unwrap();
    commit_all(&repo, "base commit");
    fs::write(
        repo.join("leaky.txt"),
        "zqonkfrelb lives in the tree only\n",
    )
    .unwrap();
    commit_all(&repo, "a perfectly clean commit message");
    let patterns = write_patterns_file(TEST_PATTERNS);

    // The default tree scan would report leaky.txt; --messages must not.
    let tree_out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&tree_out), 1, "tree scan should have found it");

    let out = run_scrub(&repo, Some(&patterns), &["--messages", "HEAD~1..HEAD"]);
    assert_eq!(
        code(&out),
        0,
        "stdout:\n{}\nstderr:\n{}",
        stdout(&out),
        stderr(&out)
    );
}

// (d) token embedded in a longer word: no match. Token in a different
// case, standalone: match.
#[test]
fn d_whole_word_and_case_insensitive() {
    let repo = init_repo();
    fs::write(repo.join("embedded.txt"), "zqonkfrelbXYZ stays put\n").unwrap();
    fs::write(repo.join("cased.txt"), "ZQONKFRELB stands alone\n").unwrap();
    commit_all(&repo, "two files");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 1, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);
    assert!(
        !text.contains("embedded.txt"),
        "embedded (longer-word) form should not match:\n{text}"
    );
    assert!(
        text.contains("cased.txt:1: pattern #1"),
        "different-case standalone form should match:\n{text}"
    );
}

// (e) variable unset -> 2.
#[test]
fn e_variable_unset() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");

    let out = run_scrub(&repo, None, &[]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (f) patterns file empty; patterns file of only comments and blanks -> 2
// for both.
#[test]
fn f_no_usable_pattern() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");

    let empty = write_patterns_file("");
    let out = run_scrub(&repo, Some(&empty), &[]);
    assert_eq!(code(&out), 2, "empty list, stdout:\n{}", stdout(&out));

    let comments_only = write_patterns_file("# just a comment\n\n   \n");
    let out = run_scrub(&repo, Some(&comments_only), &[]);
    assert_eq!(
        code(&out),
        2,
        "comments/blanks only, stdout:\n{}",
        stdout(&out)
    );
}

// (g) patterns file path unreadable or missing -> 2.
#[test]
fn g_pattern_file_missing() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");

    let missing = unique_dir("missing-patterns").join("does-not-exist.txt");
    let out = run_scrub(&repo, Some(&missing), &[]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (h) one pattern that will not compile -> 2; output names the line
// number and does NOT contain the pattern.
#[test]
fn h_pattern_does_not_compile() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");

    // An unbalanced group is not a valid regex.
    let bad = write_patterns_file("zqonkfrelb\n(unbalanced[\n");
    let out = run_scrub(&repo, Some(&bad), &[]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
    let err = stderr(&out);
    assert!(err.contains("line 2"), "got:\n{err}");
    assert!(
        !err.contains("unbalanced["),
        "output repeated the pattern:\n{err}"
    );
}

// (i) clean tree -> 0; printed counts of files scanned and patterns
// loaded are both greater than zero.
#[test]
fn i_clean_tree() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "nothing to see here\n").unwrap();
    fs::write(repo.join("b.txt"), "also clean\n").unwrap();
    commit_all(&repo, "clean files");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 0, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);

    let files_scanned = extract_count(&text, "files scanned");
    let patterns_loaded = extract_count(&text, "patterns loaded");
    assert!(files_scanned > 0, "got:\n{text}");
    assert!(patterns_loaded > 0, "got:\n{text}");
}

// (j) scan whose file set is empty -> 2, not 0. Covers both a repo with
// no tracked files and a run outside any repository.
#[test]
fn j_empty_scan_is_not_a_pass() {
    let patterns = write_patterns_file(TEST_PATTERNS);

    // A real repository, but nothing has ever been committed to it.
    let empty_repo = init_repo();
    let out = run_scrub(&empty_repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 2, "empty repo, stdout:\n{}", stdout(&out));

    // Not a repository at all.
    let not_a_repo = unique_dir("not-a-repo");
    let out = run_scrub(&not_a_repo, Some(&patterns), &[]);
    assert_eq!(
        code(&out),
        2,
        "outside any repository, stdout:\n{}",
        stdout(&out)
    );
}

// (k, new — D-083 §3) a tracked file that cannot be read -> 2, never a
// skip. Made unreadable by replacing it with a directory of the same
// name — that fails for every reader, root included, unlike chmod.
#[test]
fn k_unreadable_tracked_file_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("clean.txt"), "nothing to see\n").unwrap();
    fs::write(repo.join("swapped.txt"), "will be replaced\n").unwrap();
    commit_all(&repo, "two files");
    fs::remove_file(repo.join("swapped.txt")).unwrap();
    fs::create_dir(repo.join("swapped.txt")).unwrap();
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &[]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (l, new — D-083 §4) run from a subdirectory, with the token in a file
// outside that subdirectory -> 1: the scan always covers the whole
// repository from its top level.
#[test]
fn l_scan_always_covers_the_whole_repo_from_a_subdirectory() {
    let repo = init_repo();
    fs::write(repo.join("top.txt"), "zqonkfrelb at the top\n").unwrap();
    let sub = repo.join("sub");
    fs::create_dir(&sub).unwrap();
    fs::write(sub.join("inner.txt"), "nothing here\n").unwrap();
    commit_all(&repo, "top and sub");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&sub, Some(&patterns), &[]);
    assert_eq!(code(&out), 1, "stderr:\n{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("top.txt:1: pattern #1"), "got:\n{text}");
}

// (m, new — D-083 §1) `--messages` over a range with no commits -> 2.
#[test]
fn m_messages_over_an_empty_range_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &["--messages", "HEAD..HEAD"]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (n, new — D-084) `scrub --messages` with no range -> 2, and no scan
// runs. Run at the command-line boundary: a range-less --messages must not
// silently fall through to the tree scan.
#[test]
fn n_messages_flag_with_no_range_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &["--messages"]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
    assert!(
        !stdout(&out).contains("scanned"),
        "a malformed invocation must not run any scan:\n{}",
        stdout(&out)
    );
}

// (o, new — D-084) an unknown argument (a typo of --messages) -> 2, not a
// silent tree scan.
#[test]
fn o_unknown_argument_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &["--mesages", "HEAD~1..HEAD"]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (p, new — D-084) `--messages` given twice -> 2.
#[test]
fn p_messages_flag_given_twice_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(
        &repo,
        Some(&patterns),
        &["--messages", "HEAD~1..HEAD", "--messages", "HEAD~1..HEAD"],
    );
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// (q, new — D-084) `--messages -x`, a value that looks like a flag -> 2.
#[test]
fn q_messages_value_that_looks_like_a_flag_is_exit_2() {
    let repo = init_repo();
    fs::write(repo.join("a.txt"), "hello\n").unwrap();
    commit_all(&repo, "init");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out = run_scrub(&repo, Some(&patterns), &["--messages", "-x"]);
    assert_eq!(code(&out), 2, "stdout:\n{}", stdout(&out));
}

// Pull the integer immediately preceding `label` out of a "clean" report
// line like `"N <label>, ..."`.
fn extract_count(text: &str, label: &str) -> u64 {
    let idx = text
        .find(label)
        .unwrap_or_else(|| panic!("{label} not found in:\n{text}"));
    let before = &text[..idx];
    let digits: String = before
        .chars()
        .rev()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.chars().rev().collect::<String>().parse().unwrap()
}
