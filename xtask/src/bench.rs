// SPDX-License-Identifier: MIT

//! Bench baseline + regression gate (product spec §10, D-005, D-012).
//!
//! Flow: run `cargo bench -p phosphene-benches`, harvest criterion's
//! per-bench `estimates.json` (median ns/iteration) and `benchmark.json`
//! (elements per iteration), then either store the result as the baseline
//! for a runner class or compare against the stored baseline and fail on a
//! median regression over 10%. Baselines are keyed by *runner class* (hardware
//! descriptor or CI image name — never a host name; this repo is public),
//! because a number measured on one machine class says nothing about
//! another.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};

use crate::XtResult;

/// §10: a regression is a measured median more than 10% above the baseline.
/// D-021 rules WHERE this gates: only on a fixed, known machine whose
/// baseline is meaningful. On GitHub-hosted runners the comparison is
/// advisory (`--advisory`) — measured run-to-run variance there is +20–36%
/// on identical code, so a wall-clock gate measures which machine the cloud
/// handed us, and widening the threshold past that variance would pass real
/// regressions instead. Record, print, upload; never fail the build there.
const REGRESSION_TOLERANCE: f64 = 0.10;

/// One bench's stored numbers: median wall time per iteration and how many
/// complex samples one iteration processes (for MS/s reporting).
#[derive(Debug, Clone, PartialEq)]
pub struct BenchResult {
    pub median_ns: f64,
    pub elements: Option<u64>,
}

pub type BenchSet = BTreeMap<String, BenchResult>;

/// Outcome of comparing one measured bench against its baseline entry.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    /// Within tolerance (includes any improvement).
    Ok { delta_pct: f64 },
    /// >10% slower than baseline — fails the gate.
    Regression { delta_pct: f64 },
    /// Present in the run but absent from the baseline.
    NotInBaseline,
    /// Present in the baseline but absent from the run.
    Missing,
}

/// Write the baseline for `class` from a fresh bench run.
pub fn baseline(root: &Path, class: &str, quick: bool) -> XtResult<()> {
    let measured = run_benches(root, quick)?;
    if measured.is_empty() {
        return Err("bench run produced no criterion results".into());
    }
    let path = baseline_path(root, class);
    fs::create_dir_all(path.parent().expect("baselines dir has a parent"))?;
    fs::write(&path, render_bench_set(class, &measured))?;
    println!("\nwrote baseline for class `{class}` -> {}", path.display());
    print_table(&measured);
    Ok(())
}

/// Run the benches and compare against the stored baseline for `class`.
/// Returns `Ok(false)` when the gate fails (regression), `Ok(true)` when it
/// passes — including the seeding case where no baseline exists yet. In
/// `advisory` mode (D-021: GitHub-hosted runners) the comparison is printed
/// and the results uploaded, but the outcome is always `Ok(true)`.
pub fn check(root: &Path, class: &str, quick: bool, advisory: bool) -> XtResult<bool> {
    let measured = run_benches(root, quick)?;
    if measured.is_empty() {
        return Err("bench run produced no criterion results".into());
    }
    print_table(&measured);

    // Always leave the measured numbers where CI can pick them up as an
    // artifact — that is how a new runner class gets its first baseline.
    let results_path = root
        .join("target/bench-results")
        .join(format!("{class}.json"));
    fs::create_dir_all(results_path.parent().expect("results dir has a parent"))?;
    fs::write(&results_path, render_bench_set(class, &measured))?;
    println!("measured results written to {}", results_path.display());

    let path = baseline_path(root, class);
    if !path.exists() {
        println!(
            "\nNOTE: no stored baseline for runner class `{class}` \
             ({} does not exist).\nSeed it by committing the measured results \
             file above as that baseline. Passing without a gate THIS RUN ONLY.",
            path.display()
        );
        return Ok(true);
    }

    let stored = parse_bench_set(&fs::read_to_string(&path)?)?;
    let limit_pct = REGRESSION_TOLERANCE * 100.0;
    let verdicts = compare(&stored, &measured, REGRESSION_TOLERANCE);
    let mut failed = false;
    println!(
        "\nbaseline `{class}` ({}), tolerance +{limit_pct:.0}%:",
        path.display()
    );
    for (id, verdict) in &verdicts {
        match verdict {
            Verdict::Ok { delta_pct } => {
                println!("  OK          {id}  ({delta_pct:+.1}% vs baseline)");
            }
            Verdict::Regression { delta_pct } => {
                failed = true;
                println!(
                    "  REGRESSION  {id}  ({delta_pct:+.1}% vs baseline, limit +{limit_pct:.0}%)"
                );
            }
            Verdict::NotInBaseline => {
                println!("  NEW         {id}  (not in baseline — commit an updated baseline)");
            }
            Verdict::Missing => {
                failed = true;
                println!("  MISSING     {id}  (in baseline but not measured)");
            }
        }
    }
    if failed {
        if advisory {
            println!(
                "\nADVISORY (D-021): regression observed vs the `{class}` baseline, but \
                 GitHub-hosted runner timings do not gate — measured machine-to-machine \
                 variance there exceeds the threshold. The authoritative >{limit_pct:.0}% \
                 gate runs against a fixed-hardware class. Not failing the build."
            );
            return Ok(true);
        }
        println!("\nFAIL: >{limit_pct:.0}% regression against the `{class}` baseline (spec §10).");
    } else {
        println!("\nPASS: within {limit_pct:.0}% of the `{class}` baseline.");
    }
    Ok(!failed)
}

fn baseline_path(root: &Path, class: &str) -> PathBuf {
    root.join("benches/baselines").join(format!("{class}.json"))
}

/// Compare every bench that appears in either set. Pure, so the gate logic
/// itself is unit-tested — a gate nobody has seen fail is not known to work.
pub fn compare(baseline: &BenchSet, measured: &BenchSet, tolerance: f64) -> Vec<(String, Verdict)> {
    let mut out = Vec::new();
    for (id, base) in baseline {
        match measured.get(id) {
            None => out.push((id.clone(), Verdict::Missing)),
            Some(m) => {
                let delta = (m.median_ns - base.median_ns) / base.median_ns;
                let verdict = if delta > tolerance {
                    Verdict::Regression {
                        delta_pct: delta * 100.0,
                    }
                } else {
                    Verdict::Ok {
                        delta_pct: delta * 100.0,
                    }
                };
                out.push((id.clone(), verdict));
            }
        }
    }
    for id in measured.keys() {
        if !baseline.contains_key(id) {
            out.push((id.clone(), Verdict::NotInBaseline));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Run the criterion benches and harvest their results. The criterion output
/// directory is cleared first so stale results from renamed or removed
/// benches cannot leak into the comparison.
fn run_benches(root: &Path, quick: bool) -> XtResult<BenchSet> {
    let criterion_dir = root.join("target/criterion");
    if criterion_dir.exists() {
        fs::remove_dir_all(&criterion_dir)?;
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root)
        .args(["bench", "-p", "phosphene-benches", "--bench", "dsp", "--"]);
    if quick {
        // CI profile: enough samples for a stable median, small enough to
        // keep the job in minutes. Baselines for a class must be recorded
        // with the same profile the class is checked with.
        cmd.args([
            "--sample-size",
            "20",
            "--measurement-time",
            "2",
            "--warm-up-time",
            "1",
        ]);
    }
    let status = cmd.status()?;
    if !status.success() {
        return Err(format!("cargo bench failed with {status}").into());
    }
    collect_criterion(&criterion_dir)
}

/// Walk `target/criterion` for `*/new/estimates.json` files (skipping the
/// aggregate `report` directories) and pull the median estimate plus the
/// throughput element count criterion recorded alongside it.
fn collect_criterion(dir: &Path) -> XtResult<BenchSet> {
    let mut set = BenchSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || path.file_name().is_some_and(|n| n == "report") {
                continue;
            }
            let estimates = path.join("new/estimates.json");
            if estimates.exists() {
                let (id, result) = read_one_bench(dir, &path)?;
                set.insert(id, result);
            } else {
                stack.push(path);
            }
        }
    }
    Ok(set)
}

fn read_one_bench(criterion_root: &Path, bench_dir: &Path) -> XtResult<(String, BenchResult)> {
    let estimates: Value =
        serde_json::from_str(&fs::read_to_string(bench_dir.join("new/estimates.json"))?)?;
    let median_ns = estimates["median"]["point_estimate"]
        .as_f64()
        .ok_or("estimates.json has no median.point_estimate")?;

    let benchmark: Value =
        serde_json::from_str(&fs::read_to_string(bench_dir.join("new/benchmark.json"))?)?;
    let id = benchmark["full_id"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| {
            bench_dir
                .strip_prefix(criterion_root)
                .unwrap_or(bench_dir)
                .to_string_lossy()
                .replace('\\', "/")
        });
    let elements = benchmark["throughput"]["Elements"].as_u64();
    Ok((
        id,
        BenchResult {
            median_ns,
            elements,
        },
    ))
}

/// Serialize a bench set as the stored-baseline JSON document.
fn render_bench_set(class: &str, set: &BenchSet) -> String {
    let mut benches = Map::new();
    for (id, r) in set {
        let mut entry = Map::new();
        entry.insert("median_ns".into(), json!(r.median_ns));
        if let Some(e) = r.elements {
            entry.insert("elements".into(), json!(e));
        }
        benches.insert(id.clone(), Value::Object(entry));
    }
    let doc = json!({
        "SPDX-License-Identifier": "MIT",
        "class": class,
        "generated_by": "cargo run -p xtask -- bench-baseline (spec §10 regression baseline)",
        "benches": Value::Object(benches),
    });
    let mut s = serde_json::to_string_pretty(&doc).expect("static JSON shape always serializes");
    s.push('\n');
    s
}

/// Parse a stored-baseline (or measured-results) JSON document.
pub fn parse_bench_set(text: &str) -> XtResult<BenchSet> {
    let doc: Value = serde_json::from_str(text)?;
    let benches = doc["benches"]
        .as_object()
        .ok_or("baseline JSON has no `benches` object")?;
    let mut set = BenchSet::new();
    for (id, entry) in benches {
        let median_ns = entry["median_ns"]
            .as_f64()
            .ok_or_else(|| format!("baseline entry `{id}` has no median_ns"))?;
        set.insert(
            id.clone(),
            BenchResult {
                median_ns,
                elements: entry["elements"].as_u64(),
            },
        );
    }
    Ok(set)
}

/// Markdown table of a bench set — the shape the README perf table and the
/// PR report use (Appendix B: the re-baselined numbers become both).
fn print_table(set: &BenchSet) {
    println!("\n| bench | median/iter | throughput |");
    println!("|---|---|---|");
    for (id, r) in set {
        let throughput = match r.elements {
            Some(e) => format!("{:.1} MS/s", e as f64 / r.median_ns * 1000.0),
            None => "—".into(),
        };
        println!("| `{id}` | {:.2} µs | {throughput} |", r.median_ns / 1000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(entries: &[(&str, f64)]) -> BenchSet {
        entries
            .iter()
            .map(|(id, ns)| {
                (
                    (*id).to_owned(),
                    BenchResult {
                        median_ns: *ns,
                        elements: Some(1024),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn within_tolerance_passes() {
        let verdicts = compare(
            &set(&[("e2e/1024", 1000.0)]),
            &set(&[("e2e/1024", 1099.0)]),
            REGRESSION_TOLERANCE,
        );
        assert!(matches!(verdicts[0].1, Verdict::Ok { .. }));
    }

    #[test]
    fn over_ten_percent_fails() {
        let verdicts = compare(
            &set(&[("e2e/1024", 1000.0)]),
            &set(&[("e2e/1024", 1101.0)]),
            REGRESSION_TOLERANCE,
        );
        assert!(matches!(verdicts[0].1, Verdict::Regression { .. }));
    }

    #[test]
    fn improvement_passes() {
        let verdicts = compare(
            &set(&[("e2e/1024", 1000.0)]),
            &set(&[("e2e/1024", 500.0)]),
            REGRESSION_TOLERANCE,
        );
        assert!(matches!(verdicts[0].1, Verdict::Ok { .. }));
    }

    #[test]
    fn missing_and_new_are_flagged() {
        let verdicts = compare(
            &set(&[("gone", 1.0)]),
            &set(&[("added", 1.0)]),
            REGRESSION_TOLERANCE,
        );
        assert_eq!(
            verdicts,
            vec![
                ("added".to_owned(), Verdict::NotInBaseline),
                ("gone".to_owned(), Verdict::Missing),
            ]
        );
    }

    #[test]
    fn baseline_roundtrips_through_json() {
        let original = set(&[("window_fft/512", 123.5), ("end_to_end/32768", 99000.0)]);
        let parsed = parse_bench_set(&render_bench_set("test-class", &original)).unwrap();
        assert_eq!(parsed, original);
    }
}
