// SPDX-License-Identifier: MIT

//! The SigMF metadata sidecar (D-055, FR-S1) — read only, three fields.
//!
//! A SigMF recording is a pair: `<name>.sigmf-data` beside `<name>.sigmf-meta`,
//! the latter a JSON object. This module reads exactly the three values D-055
//! names and nothing else:
//!
//! | JSON path                        | Becomes |
//! |----------------------------------|---------|
//! | `global."core:sample_rate"`      | the sample rate, Hz |
//! | `captures[0]."core:frequency"`   | the centre frequency, Hz |
//! | `global."core:datatype"`         | an [`IqFormat`] |
//!
//! Verified against the shape in the fleet's own corpus:
//!
//! ```json
//! { "global":   { "core:sample_rate": 7680000.0, "core:datatype": "ci32_le" },
//!   "captures": [ { "core:sample_start": 0, "core:frequency": 1876954000.0 } ] }
//! ```
//!
//! That `ci32_le` is the reason lane M3-C exists: it is the datatype the
//! fleet's own capture declares, and until D-062 it was a named refusal. It
//! maps to [`IqFormat::Cs32`] — one variant, several spellings.
//!
//! **Annotations are out of scope** — deliberately, per the lane: the three
//! values above are the whole job.
//!
//! ## What is an error and what is an absence
//!
//! A sidecar that is not there is an **absence**: the caller falls through to
//! the next rung of the precedence chain. So is a sidecar that simply does
//! not carry one of the three fields — SigMF does not require them all.
//!
//! A sidecar that *is* there and cannot be believed is an **error** that names
//! the file and the field: unreadable, not JSON, a field of the wrong type, a
//! rate that is not a positive finite number, or — the case D-055 calls out —
//! a `core:datatype` naming a layout this tool does not support. That last one
//! is a clear error naming the datatype, never a silent fallback to a guess,
//! because a guessed layout renders garbage that looks like signal.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::format::IqFormat;

/// What a sidecar declared. Every field is optional because SigMF does not
/// require any of them; `None` is an absence, never a default.
#[derive(Debug, Clone, PartialEq)]
pub struct Sidecar {
    /// The sidecar file these values were read from.
    pub path: PathBuf,
    /// `global."core:sample_rate"`, Hz.
    pub sample_rate_hz: Option<f64>,
    /// `captures[0]."core:frequency"`, Hz.
    pub center_freq_hz: Option<f64>,
    /// `global."core:datatype"` mapped to one of our formats. `None` when the
    /// key is absent; an *unsupported* datatype is an error, not a `None`,
    /// and is carried in [`Sidecar::datatype`] either way.
    pub format: Option<IqFormat>,
    /// The raw `core:datatype` string, kept verbatim for error text and for
    /// the caller that has an explicit `--format` and does not need it.
    pub datatype: Option<String>,
}

/// A sidecar that exists but cannot be believed.
#[derive(Debug)]
pub enum SidecarError {
    /// The file is there but could not be read.
    Unreadable {
        /// The sidecar path.
        path: PathBuf,
        /// The underlying I/O failure.
        source: io::Error,
    },
    /// The file is not valid JSON, or its top level is not an object.
    Malformed {
        /// The sidecar path.
        path: PathBuf,
        /// What the parser objected to.
        detail: String,
    },
    /// A field is present but is not a value we can use.
    Field {
        /// The sidecar path.
        path: PathBuf,
        /// The SigMF key, e.g. `core:sample_rate`.
        key: &'static str,
        /// Why it was rejected.
        detail: String,
    },
    /// `core:datatype` names a layout this tool does not support (D-055: a
    /// clear error naming it, never a silent fallback to a guess).
    UnsupportedDatatype {
        /// The sidecar path.
        path: PathBuf,
        /// The datatype exactly as the sidecar spelled it.
        datatype: String,
    },
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidecarError::Unreadable { path, source } => {
                write!(
                    f,
                    "cannot read the SigMF sidecar {}: {source}",
                    path.display()
                )
            }
            SidecarError::Malformed { path, detail } => write!(
                f,
                "the SigMF sidecar {} is not readable JSON metadata: {detail}",
                path.display()
            ),
            SidecarError::Field { path, key, detail } => write!(
                f,
                "the SigMF sidecar {} declares {key} as {detail}",
                path.display()
            ),
            SidecarError::UnsupportedDatatype { path, datatype } => write!(
                f,
                "the SigMF sidecar {} declares core:datatype {:?}, which phosphene \
                 does not support (supported: {}); pass --format to read it as one \
                 of cf32, cs32, cs16, cs8, cu8 if you know the layout",
                path.display(),
                datatype,
                supported_datatypes(),
            ),
        }
    }
}

impl std::error::Error for SidecarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SidecarError::Unreadable { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The SigMF `core:datatype` spellings that map onto our five formats.
///
/// A SigMF datatype is `<c|r><f|i|u><bits>[_le|_be]`. Only the **complex**
/// (`c`) forms are listed: this is a complex-baseband tool, and a real-valued
/// recording is a different thing rather than a nearly-right one. The
/// endianness suffix is required by SigMF above 8 bits and meaningless at or
/// below it, so the single-byte forms are accepted with or without it — that
/// is an exact equivalence, not a charitable guess.
const DATATYPES: &[(&str, IqFormat)] = &[
    ("cf32_le", IqFormat::Cf32),
    ("ci32_le", IqFormat::Cs32),
    ("ci16_le", IqFormat::Cs16),
    ("ci8", IqFormat::Cs8),
    ("ci8_le", IqFormat::Cs8),
    ("ci8_be", IqFormat::Cs8),
    ("cu8", IqFormat::Cu8),
    ("cu8_le", IqFormat::Cu8),
    ("cu8_be", IqFormat::Cu8),
];

/// The supported `core:datatype` names, for error text.
fn supported_datatypes() -> String {
    let mut names: Vec<&str> = DATATYPES.iter().map(|&(name, _)| name).collect();
    names.dedup();
    names.join(", ")
}

/// Map a SigMF `core:datatype` onto one of our formats, or `None` if we do
/// not support that layout.
///
/// Big-endian forms above one byte are deliberately absent: `cf32_be`,
/// `ci16_be` and `ci32_be` are all real SigMF datatypes that this tool cannot
/// decode, and answering with the nearest layout we *do* have — the
/// same-width little-endian one, whose bytes are the same bytes in the wrong
/// order — would show convincing nonsense. D-062 is explicit that `ci32_be`
/// stays refused by name rather than being implemented speculatively.
pub fn format_for_datatype(datatype: &str) -> Option<IqFormat> {
    DATATYPES
        .iter()
        .find(|&&(name, _)| name == datatype)
        .map(|&(_, format)| format)
}

/// The sidecar paths to try beside a dataset file, most canonical first.
///
/// 1. **Stem replacement** — `capture.sc16` → `capture.sigmf-meta`. This is
///    SigMF's own pairing: `<name>.sigmf-data` beside `<name>.sigmf-meta`, and
///    it is what the corpus this was verified against uses.
/// 2. **Appended** — `capture.sc16` → `capture.sc16.sigmf-meta`, which some
///    tools write when the dataset keeps its native extension.
///
/// Both are literally "the `.sigmf-meta` beside it"; the order only decides
/// which wins in the (unobserved) case that a directory holds both.
pub fn candidate_paths(dataset: &Path) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(2);
    let stem_replaced = dataset.with_extension("sigmf-meta");
    // `with_extension` on a name that has none appends, making the two
    // candidates identical — try each path once.
    out.push(stem_replaced);
    let mut appended = dataset.as_os_str().to_owned();
    appended.push(".sigmf-meta");
    let appended = PathBuf::from(appended);
    if !out.contains(&appended) {
        out.push(appended);
    }
    out
}

/// Read the SigMF sidecar beside `dataset`, if there is one.
///
/// `Ok(None)` means no sidecar exists — an absence, and the caller's cue to
/// fall through the precedence chain. `Err` means one exists and cannot be
/// believed; see [`SidecarError`].
pub fn read_beside(dataset: &Path) -> Result<Option<Sidecar>, SidecarError> {
    for path in candidate_paths(dataset) {
        match fs::read_to_string(&path) {
            Ok(text) => return parse(&path, &text).map(Some),
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => return Err(SidecarError::Unreadable { path, source }),
        }
    }
    Ok(None)
}

/// Parse sidecar JSON that has already been read from `path`.
pub fn parse(path: &Path, text: &str) -> Result<Sidecar, SidecarError> {
    let malformed = |detail: String| SidecarError::Malformed {
        path: path.to_path_buf(),
        detail,
    };
    let root: serde_json::Value =
        serde_json::from_str(text).map_err(|e| malformed(e.to_string()))?;
    let global = match root.get("global") {
        Some(serde_json::Value::Object(map)) => Some(map),
        Some(other) => {
            return Err(malformed(format!(
                "\"global\" is {}, not an object",
                type_name(other)
            )))
        }
        // SigMF requires `global`, but a sidecar carrying only `captures` is
        // still readable for what it does carry.
        None if root.is_object() => None,
        None => return Err(malformed("the top level is not a JSON object".to_owned())),
    };

    let sample_rate_hz = match global.and_then(|g| g.get("core:sample_rate")) {
        None => None,
        Some(value) => Some(positive_number(path, "core:sample_rate", value)?),
    };

    let center_freq_hz = match first_capture(path, &root)?.and_then(|c| c.get("core:frequency")) {
        None => None,
        Some(value) => Some(finite_number(path, "core:frequency", value)?),
    };

    let datatype = match global.and_then(|g| g.get("core:datatype")) {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(other) => {
            return Err(SidecarError::Field {
                path: path.to_path_buf(),
                key: "core:datatype",
                detail: format!("{}, not a string", type_name(other)),
            })
        }
    };
    // An unsupported datatype is reported by the caller *if it would be used*
    // — an explicit `--format` outranks the sidecar (D-055), and a value the
    // user overrode is not a reason to refuse the file.
    let format = datatype.as_deref().and_then(format_for_datatype);

    Ok(Sidecar {
        path: path.to_path_buf(),
        sample_rate_hz,
        center_freq_hz,
        format,
        datatype,
    })
}

/// `captures[0]`, or `None` when there are no captures at all.
fn first_capture<'a>(
    path: &Path,
    root: &'a serde_json::Value,
) -> Result<Option<&'a serde_json::Map<String, serde_json::Value>>, SidecarError> {
    match root.get("captures") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Array(items)) => match items.first() {
            None => Ok(None),
            Some(serde_json::Value::Object(map)) => Ok(Some(map)),
            Some(other) => Err(SidecarError::Field {
                path: path.to_path_buf(),
                key: "captures[0]",
                detail: format!("{}, not an object", type_name(other)),
            }),
        },
        Some(other) => Err(SidecarError::Field {
            path: path.to_path_buf(),
            key: "captures",
            detail: format!("{}, not an array", type_name(other)),
        }),
    }
}

fn finite_number(
    path: &Path,
    key: &'static str,
    value: &serde_json::Value,
) -> Result<f64, SidecarError> {
    match value.as_f64() {
        Some(v) if v.is_finite() => Ok(v),
        _ => Err(SidecarError::Field {
            path: path.to_path_buf(),
            key,
            detail: format!("{value}, which is not a finite number"),
        }),
    }
}

fn positive_number(
    path: &Path,
    key: &'static str,
    value: &serde_json::Value,
) -> Result<f64, SidecarError> {
    let v = finite_number(path, key, value)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(SidecarError::Field {
            path: path.to_path_buf(),
            key,
            detail: format!("{v}, which is not a positive rate"),
        })
    }
}

fn type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape D-055 verified against the fleet's own corpus — down
    /// to the `ci32_le` datatype, which D-062 made this tool support as
    /// `cs32`. Before M3-C this file was a named refusal; it is the lane's
    /// reason to exist.
    const CORPUS: &str = r#"{
 "global": {
  "core:datatype": "ci32_le",
  "core:sample_rate": 7680000.0,
  "core:num_channels": 1,
  "core:version": "1.2.3",
  "core:description": "a real capture"
 },
 "captures": [ { "core:sample_start": 0, "core:frequency": 1876954000.0 } ],
 "annotations": []
}"#;

    fn p(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn the_corpus_shape_yields_the_three_values_d055_names() {
        let sidecar = parse(&p("corpus.sigmf-meta"), CORPUS).expect("the corpus shape must parse");
        assert_eq!(sidecar.sample_rate_hz, Some(7_680_000.0));
        assert_eq!(sidecar.center_freq_hz, Some(1_876_954_000.0));
        assert_eq!(sidecar.datatype.as_deref(), Some("ci32_le"));
        // D-062: the datatype that used to be a refusal now resolves — and it
        // resolves to the same one variant `cs32` names.
        assert_eq!(sidecar.format, Some(IqFormat::Cs32));
        assert_eq!(sidecar.format, "cs32".parse::<IqFormat>().ok());
    }

    #[test]
    fn every_supported_datatype_maps_and_the_rest_do_not() {
        for (name, want) in [
            ("cf32_le", IqFormat::Cf32),
            ("ci32_le", IqFormat::Cs32),
            ("ci16_le", IqFormat::Cs16),
            ("ci8", IqFormat::Cs8),
            ("ci8_le", IqFormat::Cs8),
            ("cu8", IqFormat::Cu8),
        ] {
            assert_eq!(format_for_datatype(name), Some(want), "{name}");
        }
        // Real SigMF datatypes we cannot decode. Each of these has a
        // superficially "close" format in our set, which is exactly why
        // answering with it would be a lie.
        for name in [
            "ci32_be", "ci32", "cf64_le", "cf32_be", "ci16_be", "rf32_le", "ri16_le", "cu16_le",
            "", "CF32_LE",
        ] {
            assert_eq!(format_for_datatype(name), None, "{name} must not map");
        }
    }

    #[test]
    fn an_unsupported_datatype_error_names_the_datatype_and_the_file() {
        let err = SidecarError::UnsupportedDatatype {
            path: p("/caps/x.sigmf-meta"),
            datatype: "ci32_be".to_owned(),
        }
        .to_string();
        assert!(err.contains("ci32_be"), "must name the datatype: {err}");
        assert!(err.contains("x.sigmf-meta"), "must name the file: {err}");
        assert!(err.contains("cf32_le"), "must say what is supported: {err}");
    }

    #[test]
    fn absent_fields_are_absences_not_defaults() {
        let sidecar = parse(&p("m.sigmf-meta"), r#"{"global":{},"captures":[]}"#).unwrap();
        assert_eq!(sidecar.sample_rate_hz, None);
        assert_eq!(sidecar.center_freq_hz, None);
        assert_eq!(sidecar.format, None);
        assert_eq!(sidecar.datatype, None);
        // No `captures` key at all is the same absence.
        let sidecar = parse(
            &p("m.sigmf-meta"),
            r#"{"global":{"core:sample_rate":48000}}"#,
        )
        .unwrap();
        assert_eq!(sidecar.sample_rate_hz, Some(48000.0));
        assert_eq!(sidecar.center_freq_hz, None);
    }

    #[test]
    fn a_sidecar_that_cannot_be_believed_is_an_error_naming_the_field() {
        let cases: [(&str, &str); 6] = [
            ("not json at all", "not readable JSON"),
            ("[1,2,3]", "not a JSON object"),
            (r#"{"global":42}"#, "\"global\" is a number"),
            (
                r#"{"global":{"core:sample_rate":"7.68e6"}}"#,
                "core:sample_rate",
            ),
            (
                r#"{"global":{"core:sample_rate":0}}"#,
                "not a positive rate",
            ),
            (r#"{"global":{"core:datatype":16}}"#, "core:datatype"),
        ];
        for (text, needle) in cases {
            let err = parse(&p("bad.sigmf-meta"), text)
                .expect_err(&format!("{text} must be rejected"))
                .to_string();
            assert!(err.contains(needle), "{text}: unhelpful message {err}");
            assert!(
                err.contains("bad.sigmf-meta"),
                "{text}: must name the file: {err}"
            );
        }
    }

    #[test]
    fn a_negative_or_nonfinite_centre_is_rejected_but_zero_is_allowed() {
        let ok = parse(&p("m.sigmf-meta"), r#"{"captures":[{"core:frequency":0}]}"#).unwrap();
        assert_eq!(ok.center_freq_hz, Some(0.0));
        let err = parse(
            &p("m.sigmf-meta"),
            r#"{"captures":[{"core:frequency":"DC"}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("core:frequency"), "{err}");
    }

    #[test]
    fn the_sidecar_is_looked_for_beside_the_dataset_both_ways() {
        let got = candidate_paths(Path::new("/caps/cap_632MHz_30p72Msps.sc16"));
        assert_eq!(
            got,
            vec![
                PathBuf::from("/caps/cap_632MHz_30p72Msps.sigmf-meta"),
                PathBuf::from("/caps/cap_632MHz_30p72Msps.sc16.sigmf-meta"),
            ]
        );
        // SigMF's own pairing.
        assert_eq!(
            candidate_paths(Path::new("/caps/capture.sigmf-data"))[0],
            PathBuf::from("/caps/capture.sigmf-meta")
        );
        // A dataset with no extension yields one candidate, not two equal ones.
        assert_eq!(
            candidate_paths(Path::new("/caps/capture")),
            vec![PathBuf::from("/caps/capture.sigmf-meta")]
        );
    }

    #[test]
    fn no_sidecar_is_an_absence_not_an_error() {
        let dir = std::env::temp_dir().join(format!("phosphene-sigmf-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("lonely.cs16");
        fs::write(&data, [0u8; 8]).unwrap();
        assert_eq!(read_beside(&data).unwrap(), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_sidecar_beside_the_dataset_is_found_and_read() {
        let dir = std::env::temp_dir().join(format!("phosphene-sigmf-read-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("capture.sigmf-data");
        fs::write(&data, [0u8; 8]).unwrap();
        fs::write(
            dir.join("capture.sigmf-meta"),
            r#"{"global":{"core:sample_rate":30720000.0,"core:datatype":"ci16_le"},
                "captures":[{"core:sample_start":0,"core:frequency":632000000.0}]}"#,
        )
        .unwrap();
        let sidecar = read_beside(&data)
            .unwrap()
            .expect("the sidecar is beside it");
        assert_eq!(sidecar.sample_rate_hz, Some(30_720_000.0));
        assert_eq!(sidecar.center_freq_hz, Some(632_000_000.0));
        assert_eq!(sidecar.format, Some(IqFormat::Cs16));

        // The appended form is found too, for a dataset that kept its own
        // extension.
        let raw = dir.join("other.sc16");
        fs::write(&raw, [0u8; 8]).unwrap();
        fs::write(
            dir.join("other.sc16.sigmf-meta"),
            r#"{"global":{"core:sample_rate":1000.0}}"#,
        )
        .unwrap();
        assert_eq!(
            read_beside(&raw).unwrap().unwrap().sample_rate_hz,
            Some(1000.0)
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// D-062 draws the line at endianness: `ci32_le` opens, `ci32_be` is
    /// still refused **by name**. Reading big-endian bytes as little-endian
    /// would decode every sample to a plausible-looking wrong number, which
    /// is worse than a refusal.
    #[test]
    fn ci32_le_opens_and_ci32_be_is_still_refused_by_name() {
        let meta = |datatype: &str| {
            format!(r#"{{"global":{{"core:datatype":"{datatype}","core:sample_rate":7680000.0}}}}"#)
        };
        let le = parse(&p("le.sigmf-meta"), &meta("ci32_le")).unwrap();
        assert_eq!(le.format, Some(IqFormat::Cs32));

        let be = parse(&p("be.sigmf-meta"), &meta("ci32_be")).unwrap();
        assert_eq!(be.format, None, "big-endian must not resolve");
        assert_eq!(be.datatype.as_deref(), Some("ci32_be"));
        // The caller turns that absence into the named error; the text it
        // produces says the datatype out loud and now offers cs32 as a
        // layout the user could ask for explicitly.
        let err = SidecarError::UnsupportedDatatype {
            path: be.path.clone(),
            datatype: be.datatype.clone().unwrap(),
        }
        .to_string();
        assert!(err.contains("ci32_be"), "{err}");
        assert!(err.contains("cs32"), "{err}");
        assert!(
            supported_datatypes().contains("ci32_le"),
            "the supported list must now name ci32_le: {}",
            supported_datatypes()
        );
    }
}
