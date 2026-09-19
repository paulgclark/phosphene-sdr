// SPDX-License-Identifier: MIT

//! The capture tells us what it is (D-055): SigMF sidecar, filename
//! inference, and the precedence between them.
//!
//! The owner's goal, stated plainly: *"users should just be able to run with
//! a well-constructed filename and no other args."* The three values a replay
//! needs — sample rate, centre frequency, sample format — usually already
//! exist somewhere other than the command line, and this module is where they
//! are found.
//!
//! * [`sigmf`] — the `.sigmf-meta` sidecar beside the capture.
//! * [`capname`] — the owner's `cap_<freq><unit>_<rate><unit>sps` filename
//!   convention, and only that one.
//!
//! ## The precedence chain, which is an honesty rule and not a convenience
//!
//! ```text
//! explicit CLI flag  >  SigMF sidecar  >  filename inference  >  nothing
//! ```
//!
//! The last rung is not a default: with nothing to go on there is **no rate**,
//! and the display falls back to a normalised frequency axis with no Hz unit
//! anywhere on it (clarification C5, D-028 §2). A rate is never invented.
//!
//! ## Every value remembers where it came from
//!
//! A rate declared by a sidecar and a rate guessed from a filename are not the
//! same claim: a filename is a **hint**, a sidecar is a **declaration**, a
//! device readback is a **measurement**. So a resolved value is a
//! [`Sourced<T>`] — the number *and* its [`Provenance`] — rather than a bare
//! number that has forgotten how much it is worth.
//!
//! **Nothing renders the provenance yet.** D-058 moved `chrome.rs` to the
//! retune lane mid-flight and deferred the readout to a follow-up. The
//! distinction stays in the data regardless, because it is what the
//! precedence chain is *made of* — the ranking exists whether or not anything
//! is currently drawing it, and the follow-up lane then only has to render
//! what is already here.

pub mod capname;
pub mod sigmf;

use std::path::{Path, PathBuf};

use crate::format::IqFormat;
use sigmf::SidecarError;

/// How strong a claim a metadata value is (D-055), in descending order of
/// precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Provenance {
    /// The user typed it: `--rate`, `--center`, `--format`. Nothing outranks
    /// what was asked for explicitly.
    Flag,
    /// A **declaration**: a SigMF sidecar said so.
    Sidecar,
    /// A **hint**: inferred from the file's name — the owner's `cap_…`
    /// convention, or a recognised extension for the format. Right almost
    /// always, and never more than a guess about a file someone renamed.
    Filename,
}

impl Provenance {
    /// A one-word tag for a status readout: `set`, `sigmf`, `hint`.
    pub fn tag(self) -> &'static str {
        match self {
            Provenance::Flag => "set",
            Provenance::Sidecar => "sigmf",
            Provenance::Filename => "hint",
        }
    }

    /// The claim spelled out, for error and log text.
    pub fn describe(self) -> &'static str {
        match self {
            Provenance::Flag => "given on the command line",
            Provenance::Sidecar => "declared by the SigMF sidecar",
            Provenance::Filename => "inferred from the file name",
        }
    }
}

/// A value together with where it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sourced<T> {
    /// The value itself.
    pub value: T,
    /// Which rung of the precedence chain produced it.
    pub from: Provenance,
}

impl<T> Sourced<T> {
    /// Pair a value with its provenance.
    pub fn new(value: T, from: Provenance) -> Self {
        Sourced { value, from }
    }
}

/// What a capture turned out to say about itself. Every field is `Option`
/// because nothing here has a default: an absence stays an absence all the
/// way to the display (C5).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CaptureMetadata {
    /// Sample rate in Hz, and its provenance.
    pub sample_rate_hz: Option<Sourced<f64>>,
    /// Centre frequency in Hz, and its provenance.
    pub center_freq_hz: Option<Sourced<f64>>,
    /// Raw IQ layout, and its provenance.
    pub format: Option<Sourced<IqFormat>>,
    /// The SigMF sidecar that was read, if one was found — for the report,
    /// and so a caller can say *which* file made a declaration.
    pub sidecar: Option<PathBuf>,
}

impl CaptureMetadata {
    /// The resolved sample rate, provenance dropped.
    pub fn rate(&self) -> Option<f64> {
        self.sample_rate_hz.map(|s| s.value)
    }

    /// The resolved centre frequency, provenance dropped.
    pub fn center(&self) -> Option<f64> {
        self.center_freq_hz.map(|s| s.value)
    }

    /// The resolved IQ format, provenance dropped.
    pub fn format(&self) -> Option<IqFormat> {
        self.format.map(|s| s.value)
    }
}

/// Walk the precedence chain for one source and report what it says about
/// itself.
///
/// `path` is the capture file, or `None` for a source with no name to read —
/// stdin has neither a sidecar nor a filename, so only the flags apply.
///
/// Errors come from one place only: a SigMF sidecar that exists and cannot be
/// believed (see [`SidecarError`]). A filename that does not match the
/// convention is **silent** — it is a hint, and a hint that does not fire is
/// not an error.
///
/// The one subtlety worth stating: an unsupported `core:datatype` is an error
/// *when it would be used*. With an explicit `--format` the user has already
/// outranked the sidecar, so refusing the file would be refusing to do what
/// was asked; with no `--format`, the sidecar is the best claim available and
/// a layout we cannot decode has to be said out loud rather than guessed past.
pub fn resolve(
    path: Option<&Path>,
    flag_rate: Option<f64>,
    flag_center: Option<f64>,
    flag_format: Option<IqFormat>,
) -> Result<CaptureMetadata, SidecarError> {
    let sidecar = match path {
        Some(path) => sigmf::read_beside(path)?,
        None => None,
    };
    let inferred = path.and_then(capname::parse_path);
    let extension = path
        .and_then(Path::extension)
        .and_then(|e| e.to_str())
        .and_then(IqFormat::from_extension);

    // Each value takes the first rung that has something to say. Written as
    // one ordered list per value so the chain is readable as the rule it is,
    // rather than as nested conditionals that have to be re-derived.
    let sample_rate_hz = first(&[
        flag_rate.map(|v| Sourced::new(v, Provenance::Flag)),
        sidecar
            .as_ref()
            .and_then(|s| s.sample_rate_hz)
            .map(|v| Sourced::new(v, Provenance::Sidecar)),
        inferred.map(|c| Sourced::new(c.sample_rate_hz, Provenance::Filename)),
    ]);
    let center_freq_hz = first(&[
        flag_center.map(|v| Sourced::new(v, Provenance::Flag)),
        sidecar
            .as_ref()
            .and_then(|s| s.center_freq_hz)
            .map(|v| Sourced::new(v, Provenance::Sidecar)),
        inferred.map(|c| Sourced::new(c.center_freq_hz, Provenance::Filename)),
    ]);
    let format = first(&[
        flag_format.map(|v| Sourced::new(v, Provenance::Flag)),
        sidecar
            .as_ref()
            .and_then(|s| s.format)
            .map(|v| Sourced::new(v, Provenance::Sidecar)),
        // The extension is part of the file's name, so it ranks with the
        // filename convention rather than beside a declaration.
        extension.map(|v| Sourced::new(v, Provenance::Filename)),
    ]);

    // D-055: a datatype we do not support is a clear error naming it — but
    // only when nothing outranked it.
    if flag_format.is_none() {
        if let Some(s) = &sidecar {
            if let (None, Some(datatype)) = (s.format, s.datatype.as_deref()) {
                return Err(SidecarError::UnsupportedDatatype {
                    path: s.path.clone(),
                    datatype: datatype.to_owned(),
                });
            }
        }
    }

    Ok(CaptureMetadata {
        sample_rate_hz,
        center_freq_hz,
        format,
        sidecar: sidecar.map(|s| s.path),
    })
}

/// The first rung of a precedence chain that has a value.
fn first<T: Copy>(chain: &[Option<Sourced<T>>]) -> Option<Sourced<T>> {
    chain.iter().find_map(|slot| *slot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A scratch directory holding a capture and (optionally) its sidecar.
    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(tag: &str) -> Fixture {
            let dir = std::env::temp_dir().join(format!(
                "phosphene-meta-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            fs::create_dir_all(&dir).expect("scratch dir");
            Fixture { dir }
        }

        fn capture(&self, name: &str) -> PathBuf {
            let path = self.dir.join(name);
            fs::write(&path, [0u8; 16]).expect("write the capture");
            path
        }

        fn sidecar(&self, name: &str, json: &str) {
            fs::write(self.dir.join(name), json).expect("write the sidecar");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.dir).ok();
        }
    }

    const DECLARED: &str = r#"{"global":{"core:sample_rate":7680000.0,"core:datatype":"ci16_le"},
        "captures":[{"core:sample_start":0,"core:frequency":1876954000.0}]}"#;

    /// Rung 3: a well-constructed filename and nothing else — the owner's
    /// actual request.
    #[test]
    fn a_conventional_filename_alone_supplies_all_three_values() {
        let fx = Fixture::new("name");
        let cap = fx.capture("cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16");
        let meta = resolve(Some(&cap), None, None, None).unwrap();
        assert_eq!(meta.rate(), Some(30.72e6));
        assert_eq!(meta.center(), Some(1.985e9));
        assert_eq!(meta.format(), Some(IqFormat::Cs16));
        // All three are hints, and the data says so.
        for from in [
            meta.sample_rate_hz.unwrap().from,
            meta.center_freq_hz.unwrap().from,
            meta.format.unwrap().from,
        ] {
            assert_eq!(from, Provenance::Filename);
        }
        assert_eq!(meta.sidecar, None);
    }

    /// Rung 2 beats rung 3: a sidecar is a declaration, a filename a hint.
    #[test]
    fn a_sidecar_outranks_the_filename_and_says_so() {
        let fx = Fixture::new("sidecar");
        // The name says 632 MHz / 30.72 Msps; the sidecar disagrees on both.
        let cap = fx.capture("cap_632MHz_30p72Msps.sc16");
        fx.sidecar("cap_632MHz_30p72Msps.sigmf-meta", DECLARED);
        let meta = resolve(Some(&cap), None, None, None).unwrap();
        assert_eq!(meta.rate(), Some(7.68e6));
        assert_eq!(meta.center(), Some(1_876_954_000.0));
        assert_eq!(meta.sample_rate_hz.unwrap().from, Provenance::Sidecar);
        assert_eq!(meta.center_freq_hz.unwrap().from, Provenance::Sidecar);
        assert_eq!(meta.format.unwrap().from, Provenance::Sidecar);
        assert!(meta.sidecar.is_some());
    }

    /// Rung 1 beats everything, including a sidecar that disagrees.
    #[test]
    fn an_explicit_flag_beats_a_sidecar_that_disagrees() {
        let fx = Fixture::new("flag");
        let cap = fx.capture("cap_632MHz_30p72Msps.sc16");
        fx.sidecar("cap_632MHz_30p72Msps.sigmf-meta", DECLARED);
        let meta = resolve(Some(&cap), Some(2.4e6), Some(100e6), Some(IqFormat::Cu8)).unwrap();
        assert_eq!(meta.rate(), Some(2.4e6));
        assert_eq!(meta.center(), Some(100e6));
        assert_eq!(meta.format(), Some(IqFormat::Cu8));
        for from in [
            meta.sample_rate_hz.unwrap().from,
            meta.center_freq_hz.unwrap().from,
            meta.format.unwrap().from,
        ] {
            assert_eq!(from, Provenance::Flag);
        }
    }

    /// The chain is per value, not all-or-nothing: one explicit flag does not
    /// discard what the sidecar knows about the others.
    #[test]
    fn the_chain_is_walked_per_value() {
        let fx = Fixture::new("mixed");
        let cap = fx.capture("cap_632MHz_30p72Msps.sc16");
        fx.sidecar("cap_632MHz_30p72Msps.sigmf-meta", DECLARED);
        let meta = resolve(Some(&cap), Some(2.4e6), None, None).unwrap();
        assert_eq!(
            meta.sample_rate_hz,
            Some(Sourced::new(2.4e6, Provenance::Flag))
        );
        assert_eq!(
            meta.center_freq_hz,
            Some(Sourced::new(1_876_954_000.0, Provenance::Sidecar))
        );
    }

    /// Rung 4 is not a default. Nothing to go on means nothing — the C5
    /// normalised axis, never an invented rate.
    #[test]
    fn nothing_to_go_on_stays_nothing() {
        let fx = Fixture::new("bare");
        let cap = fx.capture("b13_751.cs16");
        let meta = resolve(Some(&cap), None, None, None).unwrap();
        assert_eq!(meta.rate(), None);
        assert_eq!(meta.center(), None);
        // The extension is still a hint about the layout — that much of the
        // name IS recognised, and it says nothing about the rate.
        assert_eq!(
            meta.format,
            Some(Sourced::new(IqFormat::Cs16, Provenance::Filename))
        );
    }

    /// A source with no name at all: stdin has neither rung 2 nor rung 3.
    #[test]
    fn a_nameless_source_has_only_the_flags() {
        let meta = resolve(None, Some(48e3), None, Some(IqFormat::Cu8)).unwrap();
        assert_eq!(meta.rate(), Some(48e3));
        assert_eq!(meta.center(), None);
        assert_eq!(meta.format(), Some(IqFormat::Cu8));
        assert_eq!(
            resolve(None, None, None, None).unwrap(),
            CaptureMetadata::default()
        );
    }

    /// D-055: a datatype we cannot decode is a clear error naming it. Since
    /// D-062 that is `ci32_be`, not `ci32_le` — the little-endian form is the
    /// one the fleet's corpus declares and this milestone added.
    #[test]
    fn an_unsupported_sidecar_datatype_is_a_named_error() {
        let fx = Fixture::new("ci32");
        let cap = fx.capture("capture.sigmf-data");
        fx.sidecar(
            "capture.sigmf-meta",
            r#"{"global":{"core:sample_rate":7680000.0,"core:datatype":"ci32_be"},
                "captures":[{"core:frequency":1876954000.0}]}"#,
        );
        let err = resolve(Some(&cap), None, None, None)
            .expect_err("ci32_be must not be guessed past")
            .to_string();
        assert!(err.contains("ci32_be"), "must name the datatype: {err}");
        assert!(
            err.contains("capture.sigmf-meta"),
            "must name the file: {err}"
        );

        // …but an explicit --format outranks the sidecar, so the same file
        // opens: the user has already said what the layout is.
        let meta = resolve(Some(&cap), None, None, Some(IqFormat::Cs16)).unwrap();
        assert_eq!(meta.format(), Some(IqFormat::Cs16));
        assert_eq!(meta.rate(), Some(7.68e6));
        assert_eq!(meta.sample_rate_hz.unwrap().from, Provenance::Sidecar);
    }

    /// D-062, through the whole precedence chain: the corpus capture — a
    /// `.sigmf-data` file with no extension anyone could read a layout from —
    /// resolves all three values from its sidecar's declaration alone.
    #[test]
    fn the_corpus_ci32_le_sidecar_resolves_to_cs32() {
        let fx = Fixture::new("ci32le");
        let cap = fx.capture("1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data");
        fx.sidecar(
            "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta",
            r#"{"global":{"core:sample_rate":7680000.0,"core:datatype":"ci32_le"},
                "captures":[{"core:sample_start":0,"core:frequency":1876954000.0}]}"#,
        );
        let meta = resolve(Some(&cap), None, None, None).expect("ci32_le must open");
        assert_eq!(meta.format(), Some(IqFormat::Cs32));
        assert_eq!(meta.format.unwrap().from, Provenance::Sidecar);
        assert_eq!(meta.rate(), Some(7.68e6));
        assert_eq!(meta.center(), Some(1_876_954_000.0));
    }

    /// A malformed *name* is silent; a malformed *sidecar* is not. The two
    /// are different kinds of thing: a hint that does not fire is not an
    /// error, but a declaration that cannot be read is.
    #[test]
    fn a_bad_name_falls_through_silently_and_a_bad_sidecar_does_not() {
        let fx = Fixture::new("silence");
        for name in ["gnb_n3.cs16", "cap_.cs16", "cap_1p9p85GHz_30p72Msps.cs16"] {
            let cap = fx.capture(name);
            let meta = resolve(Some(&cap), None, None, None)
                .unwrap_or_else(|e| panic!("{name} must fall through silently, got: {e}"));
            assert_eq!(meta.rate(), None, "{name}");
            assert_eq!(meta.center(), None, "{name}");
        }
        let cap = fx.capture("broken.cs16");
        fx.sidecar("broken.sigmf-meta", "{ this is not json");
        assert!(resolve(Some(&cap), None, None, None).is_err());
    }

    #[test]
    fn provenance_ranks_and_tags_as_the_precedence_chain_reads() {
        assert!(Provenance::Flag < Provenance::Sidecar);
        assert!(Provenance::Sidecar < Provenance::Filename);
        assert_eq!(Provenance::Flag.tag(), "set");
        assert_eq!(Provenance::Sidecar.tag(), "sigmf");
        assert_eq!(Provenance::Filename.tag(), "hint");
        for p in [Provenance::Flag, Provenance::Sidecar, Provenance::Filename] {
            assert!(!p.describe().is_empty());
        }
    }
}
