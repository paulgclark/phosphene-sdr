// SPDX-License-Identifier: MIT

//! `cargo run -p xtask -- export-snapshot --manifest <path> --out <dir>`
//! (EXP-1): the one-time public-export mechanism (D-085 §1 — "phosphene is
//! published as a new repository built from the scrubbed tree").
//!
//! This is **export, not sync**: every run builds a brand new repository
//! (`git init` in an empty directory, never `git checkout --orphan` on this
//! repository's own object store) from a specific manifest and reports it
//! done. There is no update mode, no `--since`, no state file — an
//! already-populated `--out` path is refused, never merged into.
//!
//! **A specific commit, read from the object database.** The export's
//! source is one commit (`--commit <rev>`, default `HEAD`), resolved once
//! at the top of the run to a full object id — the same
//! resolve-once-then-use discipline `scrub` applies to the repository top
//! level. Both halves of the copy come from that commit's tree: the file
//! list from `git ls-tree -r`, and each file's **bytes** from
//! `git cat-file blob <oid>`. The working tree is never read. A checkout
//! with uncommitted local changes is the normal state of a machine
//! mid-review, and "export a specific commit" has to mean the commit, not
//! whatever happens to be on disk when someone runs the tool.
//!
//! **The manifest** (`--manifest`) is a newline-separated list of paths
//! from the source commit, one entry per line; blank lines and `#`-comment
//! lines are ignored (the pattern-file convention `scrub` already uses). An
//! entry ending in `/**` includes every file under that directory; any
//! other entry is a single file's exact path. **Include-only**: nothing
//! lands in the export unless the manifest names it, and a manifest entry
//! that matches nothing in the source commit is a hard error — a stale
//! entry is exactly the drift this tool must never paper over.
//!
//! **The safety gate** re-runs this same binary's own `scrub` and
//! `tracker-ids` subcommands (SC-2 / D-132) against the export's new
//! location, after the files are copied in and staged but before the
//! commit is made. A finding from either — or either check being unable to
//! run closed, e.g. `PHOSPHENE_SCRUB_PATTERNS_FILE` unset — deletes the
//! output directory and fails the whole command. Success is reported only
//! once both checks pass clean.
//!
//! The first commit's message states its own provenance (D-085 §1) without
//! citing any source SHA (D-082: citing this repository's SHAs anywhere
//! public defeats part of the point of a fresh history).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::XtResult;

const COMMIT_MESSAGE: &str = "snapshot: initial public export\n\n\
This repository is a one-time snapshot exported from a private development \
history. It begins here, with no relationship to any earlier commit \
history.\n";

/// The default source revision when `--commit` is not given.
const DEFAULT_COMMIT: &str = "HEAD";

struct Args {
    manifest: PathBuf,
    out: PathBuf,
    /// The revision to export — whatever the caller typed, not yet
    /// resolved.
    commit: String,
}

/// `export-snapshot` requires `--manifest <path>` and `--out <dir>`, in
/// either order — no default manifest, no default output path (matches this
/// project's fail-closed CLI convention, D-084's posture for `scrub`).
/// `--commit <rev>` is optional and defaults to `HEAD`; unlike the other
/// two it has a defensible default, because "the commit you are standing
/// on" is unambiguous whereas "the output path you meant" is not. A flag
/// with no value, an unknown argument, or any flag given twice all reject
/// rather than falling through to a default.
fn parse_args(rest: &[String]) -> Result<Args, String> {
    let mut manifest: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut commit: Option<String> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--manifest" => {
                let v = rest.get(i + 1).ok_or("--manifest requires a value")?;
                if manifest.is_some() {
                    return Err("--manifest given twice".to_string());
                }
                manifest = Some(PathBuf::from(v));
                i += 2;
            }
            "--out" => {
                let v = rest.get(i + 1).ok_or("--out requires a value")?;
                if out.is_some() {
                    return Err("--out given twice".to_string());
                }
                out = Some(PathBuf::from(v));
                i += 2;
            }
            "--commit" => {
                let v = rest.get(i + 1).ok_or("--commit requires a value")?;
                if commit.is_some() {
                    return Err("--commit given twice".to_string());
                }
                if v.starts_with('-') {
                    return Err(format!("--commit requires a revision, got `{v}`"));
                }
                commit = Some(v.clone());
                i += 2;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let manifest = manifest.ok_or("--manifest <path> is required")?;
    let out = out.ok_or("--out <dir> is required")?;
    Ok(Args {
        manifest,
        out,
        commit: commit.unwrap_or_else(|| DEFAULT_COMMIT.to_string()),
    })
}

pub fn run(rest: &[String]) -> std::process::ExitCode {
    let args = match parse_args(rest) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("export-snapshot: {msg}");
            return std::process::ExitCode::from(2);
        }
    };

    match export(&args) {
        Ok(summary) => {
            println!("{summary}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("export-snapshot: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn export(args: &Args) -> XtResult<String> {
    // One-time, not idempotent-against-a-moving-target: refuse rather than
    // merge into an output path that already has something in it.
    if path_has_content(&args.out) {
        return Err(format!(
            "output path {} already exists; refusing to merge or overwrite it — remove it first",
            args.out.display()
        )
        .into());
    }

    let src_root = repo_toplevel()
        .ok_or("could not determine the source repository's top level (not a git repository?)")?;

    // Resolved once, at the top, and used for both the file list and every
    // file's content — so a ref that moves mid-run cannot produce an export
    // stitched together from two different commits.
    let commit = resolve_commit(&src_root, &args.commit)?;

    let entries = parse_manifest(&args.manifest)?;
    let tree = commit_tree(&src_root, &commit)?;

    // Keyed by path, so an entry matched by two manifest lines is still
    // copied once, and the export's file order is stable.
    let mut selected: BTreeMap<&str, &TreeEntry> = BTreeMap::new();
    for entry in &entries {
        for matched in resolve_entry(entry, &tree)? {
            selected.insert(matched.path.as_str(), matched);
        }
    }

    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create output directory {}: {e}", args.out.display()))?;

    for (rel, entry) in &selected {
        let dst = args.out.join(rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create directory for {rel}: {e}"))?;
        }
        write_from_object_db(&src_root, entry, &dst)?;
    }

    // Fresh history, not `--orphan` on this repository's own object store:
    // a brand new repository, zero shared object-store history by
    // construction.
    run_git(&args.out, &["init", "--quiet", "--initial-branch=main"])?;
    run_git(&args.out, &["add", "-A"])?;

    if let Err(e) = run_safety_gate(&args.out) {
        // A finding from either check is a hard failure: the output
        // directory is removed, never left around in a usable state, and
        // never committed.
        let _ = fs::remove_dir_all(&args.out);
        return Err(e);
    }

    run_git(
        &args.out,
        &[
            "-c",
            "user.name=phosphene snapshot export",
            "-c",
            "user.email=noreply@phosphene.invalid",
            "commit",
            "--quiet",
            "-m",
            COMMIT_MESSAGE,
        ],
    )?;

    Ok(format!(
        "export-snapshot: wrote {} file(s) from source commit {} to {} — 1 commit, safety gate clean",
        selected.len(),
        &commit[..commit.len().min(12)],
        args.out.display()
    ))
}

/// One path in the source commit's tree. The `oid` is the whole point: the
/// bytes this export writes are read from that object, never from the file
/// of the same name in the working tree.
struct TreeEntry {
    /// The tree's mode for this path — `100644` or `100755` for a regular
    /// file; anything else this tool refuses to export rather than guess at.
    mode: String,
    oid: String,
    path: String,
}

/// A manifest line kept for error reporting: which line named a path that
/// turned out not to exist.
struct ManifestEntry {
    line_no: usize,
    raw: String,
}

fn parse_manifest(path: &Path) -> XtResult<Vec<ManifestEntry>> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("cannot read manifest {}: {e}", path.display()))?;

    let mut entries = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        entries.push(ManifestEntry {
            line_no: i + 1,
            raw: trimmed.to_string(),
        });
    }

    if entries.is_empty() {
        return Err(format!(
            "manifest {} has no usable entries (empty, or only comments/blanks)",
            path.display()
        )
        .into());
    }

    Ok(entries)
}

/// Resolve one manifest entry against the source commit's tree. A trailing
/// `/**` includes every file under that directory; anything else must match
/// one path exactly. An entry that matches nothing is a hard error (never a
/// silent skip) — a stale manifest entry is exactly the drift this tool must
/// never paper over. Note that "matches nothing" is measured against the
/// *commit*, so a file that exists only in the working tree, or only staged
/// in the index, does not satisfy a manifest entry.
fn resolve_entry<'a>(entry: &ManifestEntry, tree: &'a [TreeEntry]) -> XtResult<Vec<&'a TreeEntry>> {
    let matches: Vec<&TreeEntry> = if let Some(dir) = entry.raw.strip_suffix("/**") {
        let prefix = format!("{dir}/");
        tree.iter()
            .filter(|e| e.path.starts_with(&prefix))
            .collect()
    } else {
        tree.iter().filter(|e| e.path == entry.raw).collect()
    };

    if matches.is_empty() {
        return Err(format!(
            "manifest line {}: `{}` does not match any file in the source commit",
            entry.line_no, entry.raw
        )
        .into());
    }

    Ok(matches)
}

fn path_has_content(path: &Path) -> bool {
    match fs::metadata(path) {
        Err(_) => false,
        Ok(m) if m.is_dir() => fs::read_dir(path)
            .map(|mut d| d.next().is_some())
            .unwrap_or(true),
        Ok(_) => true,
    }
}

/// Write one file into the export from the git object database. `git
/// cat-file blob` is the raw stored bytes — no working-tree read, and no
/// `git show` textconv/diff layer that a `.gitattributes` could interpose
/// between the object and what lands in a public repository.
fn write_from_object_db(src_root: &Path, entry: &TreeEntry, dst: &Path) -> XtResult<()> {
    let executable = match entry.mode.as_str() {
        "100644" => false,
        "100755" => true,
        // Symlinks and submodule gitlinks both have a defensible export
        // meaning, and neither is worth guessing at in a one-way operation
        // that publishes its result. This tree has none; if one appears,
        // the tool stops and says so.
        other => {
            return Err(format!(
                "{}: unsupported mode {other} in the source commit \
                 (only regular files are exported)",
                entry.path
            )
            .into());
        }
    };

    let bytes = git_stdout(src_root, &["cat-file", "blob", &entry.oid])
        .map_err(|e| format!("cannot read {} from the object database: {e}", entry.path))?;
    fs::write(dst, &bytes).map_err(|e| format!("cannot write {}: {e}", entry.path))?;

    #[cfg(unix)]
    if executable {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dst, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot set the executable bit on {}: {e}", entry.path))?;
    }
    #[cfg(not(unix))]
    let _ = executable;

    Ok(())
}

/// Re-run this same binary's own `scrub` and `tracker-ids` subcommands
/// (the exact checks `scrub.rs` / `tracker_ids.rs` implement — never
/// reimplemented here) against the exported tree at its new location, by
/// pointing a child process's working directory there. Both tools resolve
/// what they scan from `git rev-parse --show-toplevel` off their own
/// process's cwd, so this reaches the export, not the source tree.
fn run_safety_gate(out_dir: &Path) -> XtResult<()> {
    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot determine this tool's own binary path: {e}"))?;

    let scrub_status = Command::new(&exe)
        .arg("scrub")
        .current_dir(out_dir)
        .status()
        .map_err(|e| format!("failed to run the scrub safety check: {e}"))?;
    if !scrub_status.success() {
        return Err(
            "safety gate failed: scrub found a match (or could not run closed) in the exported tree"
                .into(),
        );
    }

    let tracker_status = Command::new(&exe)
        .arg("tracker-ids")
        .current_dir(out_dir)
        .status()
        .map_err(|e| format!("failed to run the tracker-ids safety check: {e}"))?;
    if !tracker_status.success() {
        return Err("safety gate failed: tracker-ids found a match in the exported tree".into());
    }

    Ok(())
}

fn run_git(dir: &Path, args: &[&str]) -> XtResult<()> {
    let status = Command::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .map_err(|e| format!("failed to run git {args:?}: {e}"))?;
    if !status.success() {
        return Err(format!("git {args:?} failed in {}", dir.display()).into());
    }
    Ok(())
}

/// Run a git command in `dir` and hand back its raw stdout bytes — raw
/// because file content comes through here and a file is not necessarily
/// UTF-8.
fn git_stdout(dir: &Path, args: &[&str]) -> XtResult<Vec<u8>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run git {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )
        .into());
    }
    Ok(out.stdout)
}

/// The source repository's top level, resolved from wherever this process
/// happens to be running — same posture as `scrub`/`tracker-ids` (D-083
/// §4).
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

/// Pin the requested revision to one full commit id. `^{commit}` makes a
/// tag or a tree spelled as a revision fail here rather than half-way
/// through the copy.
fn resolve_commit(src_root: &Path, rev: &str) -> XtResult<String> {
    let spec = format!("{rev}^{{commit}}");
    let out = git_stdout(src_root, &["rev-parse", "--verify", "--quiet", &spec])
        .map_err(|_| format!("`{rev}` does not name a commit in the source repository"))?;
    let sha = String::from_utf8_lossy(&out).trim().to_string();
    if sha.is_empty() {
        return Err(format!("`{rev}` does not name a commit in the source repository").into());
    }
    Ok(sha)
}

/// Every file in the commit's tree, recursively: mode, object id and path.
/// This is the enumeration half of "export a specific commit" — `ls-tree`
/// on the commit, not `ls-files` on the index, so a file that is staged but
/// not committed, or committed and then deleted on disk, is treated the way
/// the commit says rather than the way the checkout looks.
fn commit_tree(src_root: &Path, commit: &str) -> XtResult<Vec<TreeEntry>> {
    let out = git_stdout(src_root, &["ls-tree", "-r", "-z", commit])?;
    let text = String::from_utf8_lossy(&out);

    let mut entries = Vec::new();
    for record in text.split('\0').filter(|s| !s.is_empty()) {
        // `<mode> SP <type> SP <oid> TAB <path>`
        let (meta, path) = record
            .split_once('\t')
            .ok_or_else(|| format!("unparseable `git ls-tree` record: {record}"))?;
        let fields: Vec<&str> = meta.split_whitespace().collect();
        let [mode, _kind, oid] = fields[..] else {
            return Err(format!("unparseable `git ls-tree` record: {record}").into());
        };
        entries.push(TreeEntry {
            mode: mode.to_string(),
            oid: oid.to_string(),
            path: path.to_string(),
        });
    }

    if entries.is_empty() {
        return Err(format!("source commit {commit} has no files to export").into());
    }

    Ok(entries)
}
