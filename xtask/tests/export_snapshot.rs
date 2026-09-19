// SPDX-License-Identifier: MIT

//! Behavioural tests for `cargo run -p xtask -- export-snapshot` (EXP-1 /
//! D-085 §1). Each test builds its own throwaway *source* git repository
//! under the OS temp dir, writes its own throwaway manifest, and exports
//! into its own throwaway `--out` path — none of this touches the real
//! repository or its real manifest, and only invented tokens (never the
//! real pattern list) are ever planted.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const PATTERNS_VAR: &str = "PHOSPHENE_SCRUB_PATTERNS_FILE";

/// A patterns file using only an invented token, never anything from the
/// real list.
const TEST_PATTERNS: &str = "\
# invented-token test list — not the real pattern list\n\
zqonkfrelb\n\
";

fn unique_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "phosphene-export-snapshot-test-{tag}-{}-{nanos}-{n}",
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
    let dir = unique_dir("src");
    git(&dir, &["init", "--quiet", "--initial-branch=main"]);
    git(
        &dir,
        &[
            "config",
            "user.email",
            "export-snapshot-test@example.invalid",
        ],
    );
    git(&dir, &["config", "user.name", "export-snapshot test"]);
    dir
}

fn commit_all(dir: &Path, message: &str) {
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "--quiet", "-m", message]);
}

fn write(dir: &Path, rel: &str, contents: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn write_manifest(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("manifest.txt");
    fs::write(&path, contents).unwrap();
    path
}

fn write_patterns_file(contents: &str) -> PathBuf {
    let dir = unique_dir("patterns");
    let file = dir.join("patterns.txt");
    fs::write(&file, contents).unwrap();
    file
}

/// Build a small, ordinary source repository: a manifested set of files
/// plus one un-manifested file, so tests can assert the export contains
/// exactly the manifested set and nothing else.
fn ordinary_source_repo() -> PathBuf {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    write(&src, "crates/a/src/lib.rs", "pub fn a() {}\n");
    write(&src, "crates/a/Cargo.toml", "[package]\nname = \"a\"\n");
    write(&src, "decisions.md", "internal only, never manifested\n");
    commit_all(&src, "ordinary tree");
    src
}

fn ordinary_manifest(dir: &Path) -> PathBuf {
    write_manifest(dir, "Cargo.toml\ncrates/**\n")
}

fn run_export(src: &Path, manifest: &Path, out: &Path, patterns_file: Option<&Path>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--manifest")
        .arg(manifest)
        .arg("--out")
        .arg(out)
        .current_dir(src);
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

/// List every file under `dir`, relative to `dir`, skipping `.git`.
fn list_files(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.file_name().and_then(|n| n.to_str()) == Some(".git") {
                continue;
            }
            if path.is_dir() {
                walk(base, &path, out);
            } else {
                out.push(
                    path.strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

// Happy path: exactly the manifested files land in the output, one commit,
// the safety gate reports clean, and the un-manifested file never appears.
#[test]
fn happy_path_exports_exactly_the_manifested_files() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));

    let mut files = list_files(&out);
    files.sort();
    assert_eq!(
        files,
        vec![
            "Cargo.toml".to_string(),
            "crates/a/Cargo.toml".to_string(),
            "crates/a/src/lib.rs".to_string(),
        ]
    );

    // Exactly one commit, and it is the export's own root — no history
    // predates it.
    let commit_count = Command::new("git")
        .args(["rev-list", "--all", "--count"])
        .current_dir(&out)
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&commit_count.stdout).trim(),
        "1",
        "expected exactly one commit"
    );

    let parents = Command::new("git")
        .args(["log", "-1", "--format=%P", "HEAD"])
        .current_dir(&out)
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&parents.stdout).trim().is_empty(),
        "the sole commit must have no parent"
    );
}

// The safety gate: a planted host-name-pattern hit inside a manifested
// file causes a hard refusal — non-zero exit, output directory not left
// usable, and the finding named in the tool's own output.
#[test]
fn safety_gate_refuses_on_a_scrub_finding() {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    write(&src, "crates/a/src/lib.rs", "// zqonkfrelb leaked here\n");
    commit_all(&src, "leaky tree");
    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(
        !out.exists(),
        "output directory must not survive a gate failure"
    );
    assert!(
        stderr(&result).contains("safety gate failed"),
        "stderr:\n{}",
        stderr(&result)
    );
}

// The safety gate: a planted tracker-ID-shaped string inside a manifested
// file causes the same hard refusal via `tracker-ids`.
#[test]
fn safety_gate_refuses_on_a_tracker_id_finding() {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    // Assembled from separate pieces so this test source is not itself a
    // match for the shape under test.
    let letter = "F";
    let digits = "42";
    write(
        &src,
        "crates/a/src/lib.rs",
        &format!("// see {letter}-{digits}\n"),
    );
    commit_all(&src, "tracker-id-leaky tree");
    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(
        !out.exists(),
        "output directory must not survive a gate failure"
    );
}

// The safety gate also refuses closed when it cannot run at all, exactly
// as `scrub` itself does — an unset pattern-file variable is never treated
// as "nothing to check".
#[test]
fn safety_gate_refuses_when_scrub_cannot_run_closed() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");

    let result = run_export(&src, &manifest, &out, None);
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(!out.exists());
}

// Fail-closed manifest handling: a manifest entry naming a path that does
// not exist in the source tree is a hard error, not a silent skip — and no
// output is produced at all.
#[test]
fn manifest_entry_naming_a_missing_path_is_a_hard_error() {
    let src = ordinary_source_repo();
    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\ndoes/not/exist.txt\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(
        !out.exists(),
        "no output should be produced on a bad manifest"
    );
    assert!(
        stderr(&result).contains("does/not/exist.txt"),
        "stderr:\n{}",
        stderr(&result)
    );
}

// Re-run behaviour: running the tool twice at the same output path — the
// second run refuses rather than merging or overwriting.
#[test]
fn rerun_at_the_same_output_path_refuses() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let first = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&first), 0, "stderr:\n{}", stderr(&first));

    let second = run_export(&src, &manifest, &out, Some(&patterns));
    assert_ne!(code(&second), 0, "stdout:\n{}", stdout(&second));
    assert!(
        stderr(&second).contains("already exists"),
        "stderr:\n{}",
        stderr(&second)
    );
}

// A `dir/**` manifest entry pulls in every tracked file under that
// directory, at any depth, and nothing outside it.
#[test]
fn directory_glob_matches_recursively_and_only_that_directory() {
    let src = init_repo();
    write(&src, "crates/a/src/deep/mod.rs", "pub mod x;\n");
    write(&src, "crates/b/src/lib.rs", "pub fn b() {}\n");
    write(&src, "not-crates/src/lib.rs", "pub fn c() {}\n");
    commit_all(&src, "nested tree");
    let manifest = write_manifest(&src, "crates/**\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));

    let files = list_files(&out);
    assert_eq!(
        files,
        vec![
            "crates/a/src/deep/mod.rs".to_string(),
            "crates/b/src/lib.rs".to_string(),
        ]
    );
}

// Output-path collision: an existing non-empty directory at `--out`, never
// touched by this run, is refused before anything else happens.
#[test]
fn existing_nonempty_output_path_refuses_before_any_work() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("pre-existing.txt"), "not ours\n").unwrap();

    let result = run_export(&src, &manifest, &out, None);
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(
        out.join("pre-existing.txt").exists(),
        "must not touch existing content"
    );
    assert!(
        !out.join(".git").exists(),
        "must not have started a git init"
    );
}

// CLI parsing: both flags are required, and neither has a default.
#[test]
fn missing_required_flags_are_rejected() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--manifest")
        .arg(&manifest)
        .current_dir(&src);
    let result = cmd.output().unwrap();
    assert_ne!(code(&result), 0, "missing --out must be rejected");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--out")
        .arg(&out)
        .current_dir(&src);
    let result = cmd.output().unwrap();
    assert_ne!(code(&result), 0, "missing --manifest must be rejected");
}

// CLI parsing: an unknown argument and a duplicated flag are both
// rejected rather than silently accepted.
#[test]
fn malformed_arguments_are_rejected() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--manifest")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .arg("--bogus")
        .current_dir(&src);
    assert_ne!(code(&cmd.output().unwrap()), 0);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--manifest")
        .arg(&manifest)
        .arg("--manifest")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .current_dir(&src);
    assert_ne!(code(&cmd.output().unwrap()), 0);
}

/// As `run_export`, plus extra arguments after the two required flags.
fn run_export_with(
    src: &Path,
    manifest: &Path,
    out: &Path,
    patterns_file: Option<&Path>,
    extra: &[&str],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.arg("export-snapshot")
        .arg("--manifest")
        .arg(manifest)
        .arg("--out")
        .arg(out)
        .args(extra)
        .current_dir(src);
    cmd.env_remove(PATTERNS_VAR);
    if let Some(p) = patterns_file {
        cmd.env(PATTERNS_VAR, p);
    }
    cmd.output().expect("xtask binary runs")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

// THE regression test for the review's blocker: the exporter must read each
// file's bytes from the commit's own objects, never from the working tree.
// A checkout with uncommitted local edits is the normal state of a machine
// mid-review, and an export that picks those edits up is not an export of
// the commit it claims.
#[test]
fn dirty_working_tree_exports_the_committed_content_not_the_working_copy() {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    write(&src, "crates/a/src/lib.rs", "pub fn committed() {}\n");
    commit_all(&src, "the content that is actually committed");

    // Dirty two manifested files without committing: one tracked file
    // edited in place, and one deleted from disk entirely. Neither change
    // is in the commit, so neither may reach the export.
    write(
        &src,
        "crates/a/src/lib.rs",
        "pub fn uncommitted_edit() {}\n",
    );
    fs::remove_file(src.join("Cargo.toml")).unwrap();

    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));

    assert_eq!(
        read(&out.join("crates/a/src/lib.rs")),
        "pub fn committed() {}\n",
        "the export must carry the committed bytes, not the dirty working copy"
    );
    assert_eq!(
        read(&out.join("Cargo.toml")),
        "[workspace]\n",
        "a file deleted from the working tree is still in the commit, so it still exports"
    );
}

// The same guarantee for the file *list*: a file staged in the index but
// never committed is not part of the commit, so a manifest entry naming it
// matches nothing and the run fails closed rather than exporting index
// content.
#[test]
fn a_staged_but_uncommitted_file_is_not_part_of_the_export() {
    let src = ordinary_source_repo();
    write(&src, "crates/a/src/staged.rs", "pub fn staged() {}\n");
    git(&src, &["add", "crates/a/src/staged.rs"]);

    let manifest = write_manifest(&src, "Cargo.toml\ncrates/a/src/staged.rs\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(
        stderr(&result).contains("crates/a/src/staged.rs"),
        "stderr:\n{}",
        stderr(&result)
    );
    assert!(!out.exists());
}

// `--commit <rev>` exports that commit, not `HEAD` — and picks up neither
// the newer commit's content nor the working tree's.
#[test]
fn an_explicit_commit_exports_that_commit() {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    write(&src, "crates/a/src/lib.rs", "pub fn first() {}\n");
    commit_all(&src, "first");
    let first = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&src)
        .output()
        .unwrap();
    let first = String::from_utf8_lossy(&first.stdout).trim().to_string();

    write(&src, "crates/a/src/lib.rs", "pub fn second() {}\n");
    commit_all(&src, "second");
    write(&src, "crates/a/src/lib.rs", "pub fn dirty() {}\n");

    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\n");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let out_first = unique_dir("out").join("export");
    let result = run_export_with(
        &src,
        &manifest,
        &out_first,
        Some(&patterns),
        &["--commit", &first],
    );
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));
    assert_eq!(
        read(&out_first.join("crates/a/src/lib.rs")),
        "pub fn first() {}\n"
    );

    // Default is HEAD — the second commit, still not the dirty file.
    let out_head = unique_dir("out").join("export");
    let result = run_export(&src, &manifest, &out_head, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));
    assert_eq!(
        read(&out_head.join("crates/a/src/lib.rs")),
        "pub fn second() {}\n"
    );
}

// A revision that names no commit is rejected before anything is written.
#[test]
fn an_unknown_commit_is_rejected() {
    let src = ordinary_source_repo();
    let manifest = ordinary_manifest(&src);
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export_with(
        &src,
        &manifest,
        &out,
        Some(&patterns),
        &["--commit", "no-such-ref"],
    );
    assert_ne!(code(&result), 0, "stdout:\n{}", stdout(&result));
    assert!(!out.exists(), "no output should be produced");
}

// Reading bytes out of the object database must not lose the file mode:
// `scripts/install.sh` ships in the real manifest and has to stay
// executable in the export.
#[cfg(unix)]
#[test]
fn the_executable_bit_survives_the_export() {
    use std::os::unix::fs::PermissionsExt;

    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    write(&src, "scripts/install.sh", "#!/bin/sh\nexit 0\n");
    fs::set_permissions(
        src.join("scripts/install.sh"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    commit_all(&src, "an executable script");

    let manifest = write_manifest(&src, "Cargo.toml\nscripts/install.sh\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));

    let mode = fs::metadata(out.join("scripts/install.sh"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o111, 0o111, "the executable bit must survive");
}

// Binary content survives the object-database round trip byte for byte —
// `cat-file blob` is raw bytes, and nothing in the copy path assumes UTF-8.
#[test]
fn binary_content_round_trips_byte_for_byte() {
    let src = init_repo();
    write(&src, "Cargo.toml", "[workspace]\n");
    let bytes: Vec<u8> = (0u8..=255).collect();
    fs::create_dir_all(src.join("crates/a")).unwrap();
    fs::write(src.join("crates/a/blob.bin"), &bytes).unwrap();
    commit_all(&src, "a binary file");

    let manifest = write_manifest(&src, "Cargo.toml\ncrates/**\n");
    let out = unique_dir("out").join("export");
    let patterns = write_patterns_file(TEST_PATTERNS);

    let result = run_export(&src, &manifest, &out, Some(&patterns));
    assert_eq!(code(&result), 0, "stderr:\n{}", stderr(&result));
    assert_eq!(fs::read(out.join("crates/a/blob.bin")).unwrap(), bytes);
}
