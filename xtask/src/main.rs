// SPDX-License-Identifier: MIT

//! # xtask — phosphene dev automation (never shipped)
//!
//! ## Architecture
//!
//! Plain-Rust replacements for the shell glue CI would otherwise accrete
//! (product spec §10). One binary, three subcommands, no crate-internal
//! knowledge — this tool only drives `cargo` and reads its outputs, per the
//! plan's lane contract ("dev automation, CI, packaging — never crate
//! internals"):
//!
//! * `bench-baseline --class <name>` — run the criterion benches and store
//!   their medians as `benches/baselines/<class>.json`, the per-runner-class
//!   regression baseline of spec §10 / D-005.
//! * `bench-check --class <name>` — run the benches and compare against the
//!   stored baseline for that class; a >10% median regression fails (§10).
//!   With no stored baseline the run reports, writes its results to
//!   `target/bench-results/<class>.json` for seeding, and passes.
//! * `artifact-size [--target <triple>]` — build release binaries and report
//!   their sizes against NFR-P5's ≤15 MB target (clarification C7: measure
//!   and report honestly; the number is revised by decision, never gated
//!   quietly here).
//! * `scrub [--messages <rev-range>]` — the public-scrub check (PS-1 /
//!   D-082): tracked file content and paths, plus commit messages in the
//!   given range, against a private pattern list named by
//!   `PHOSPHENE_SCRUB_PATTERNS_FILE`.
//! * `tracker-ids` — internal-citation-ID hygiene check (SC-2): tracked
//!   file content, tree-wide, against a committed `LETTER-digits` shape.
//!   Unlike `scrub`, the shape is public syntax, not a secret.
//! * `export-snapshot --manifest <path> --out <dir> [--commit <rev>]` — the
//!   one-time public export mechanism (EXP-1 / D-085 §1): builds a brand new
//!   git repository at `<dir>` from the manifested files, read out of the
//!   source commit's own objects (default `HEAD`, never the working tree),
//!   gated on a clean `scrub` + `tracker-ids` re-run against the export
//!   itself.
//!
//! Run as `cargo run -p xtask -- <subcommand>`.

#![forbid(unsafe_code)]

mod bench;
mod export_snapshot;
mod scrub;
mod size;
mod tracker_ids;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

/// Everything here is best-effort tooling; a boxed error with context beats
/// a bespoke error enum nobody matches on (NFR-Q2: boring Rust).
pub type XtResult<T> = Result<T, Box<dyn Error>>;

fn usage() -> ExitCode {
    eprintln!(
        "usage: cargo run -p xtask -- <subcommand>\n\
         \n\
         subcommands:\n\
           bench-baseline --class <name> [--quick]   write benches/baselines/<name>.json\n\
           bench-check    --class <name> [--quick] [--advisory]\n\
                                                     compare against the stored baseline;\n\
                                                     >10% median regression fails (spec §10).\n\
                                                     --advisory (D-021, GitHub-hosted runners):\n\
                                                     print + record, never fail the build\n\
           artifact-size  [--target <triple>]        report release binary sizes vs NFR-P5 (C7)\n\
           scrub          [--messages <rev-range>]   public-scrub check (PS-1 / D-082)\n\
           tracker-ids                               internal-citation-ID hygiene check (SC-2)\n\
           export-snapshot --manifest <path> --out <dir> [--commit <rev>]\n\
                                                     one-time public export (EXP-1 / D-085 §1);\n\
                                                     --commit defaults to HEAD and is read from\n\
                                                     git's objects, never the working tree"
    );
    ExitCode::from(2)
}

/// Repo root = parent of this crate's manifest dir (`xtask/` lives at the
/// workspace root).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level below the workspace root")
        .to_path_buf()
}

/// Pull the value following a `--flag` out of the arg list.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `scrub` accepts exactly two forms: `scrub`, or `scrub --messages
/// <range>`. Anything else is a usage error (D-084) — a flag with no value,
/// a value that begins with `-` (so it can't be mistaken for a flag itself),
/// an unknown argument, or `--messages` given twice all reject rather than
/// silently falling through to the tree scan, which is exactly the fail-open
/// hole D-083 closed everywhere except the command line.
fn parse_scrub_args(rest: &[String]) -> Result<Option<String>, ()> {
    match rest {
        [] => Ok(None),
        [flag, range] if flag == "--messages" && !range.starts_with('-') => Ok(Some(range.clone())),
        _ => Err(()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        return usage();
    };
    let rest = &args[1..];
    let quick = rest.iter().any(|a| a == "--quick");
    let advisory = rest.iter().any(|a| a == "--advisory");

    if cmd == "scrub" {
        return match parse_scrub_args(rest) {
            Ok(range) => scrub::run(range.as_deref()),
            Err(()) => usage(),
        };
    }

    if cmd == "tracker-ids" {
        return if rest.is_empty() {
            tracker_ids::run()
        } else {
            usage()
        };
    }

    if cmd == "export-snapshot" {
        return export_snapshot::run(rest);
    }

    let outcome: XtResult<bool> = match cmd.as_str() {
        "bench-baseline" | "bench-check" => match flag_value(rest, "--class") {
            Some(class) => {
                if cmd == "bench-baseline" {
                    bench::baseline(&repo_root(), &class, quick).map(|()| true)
                } else {
                    bench::check(&repo_root(), &class, quick, advisory)
                }
            }
            None => {
                eprintln!("error: {cmd} requires --class <name>");
                return usage();
            }
        },
        "artifact-size" => {
            size::report(&repo_root(), flag_value(rest, "--target").as_deref()).map(|()| true)
        }
        _ => return usage(),
    };

    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
