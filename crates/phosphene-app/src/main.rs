// SPDX-License-Identifier: MIT

//! # phosphene — real-time spectrum analyzer
//!
//! ## Architecture
//!
//! The binary crate wires the workspace together and owns nothing clever
//! (§8.1): CLI parsing here, the source→ring→FFT→display pipeline in
//! [`pipeline`], rendering in `phosphene-render`, DSP in `phosphene-core`,
//! sources in `phosphene-sources`.
//!
//! Two entry points share one render path: a winit window ([`window`]) that
//! consumes a real [`SampleSource`](phosphene_sources::SampleSource) through
//! the batch pipeline, and an offscreen PNG renderer ([`headless`]). With no
//! `--source`, headless deliberately keeps the deterministic local scene
//! ([`synth`]) so CI's golden output depends only on the rasterizer; with
//! a real `--source` (`file:<path>`, `stdin`, or `soapy`) it drives the
//! same batch pipeline the window consumes and renders that capture to the
//! PNG (D-041 / M2-A).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use phosphene_sources::metadata::{self, CaptureMetadata};
use phosphene_sources::{
    FileSourceConfig, Input, IqFormat, ReplayPace, SigGenConfig, SoapySourceConfig, SourceDesc,
};

#[cfg(feature = "analyze")]
mod analyze;
mod headless;
mod pipeline;
mod synth;
mod window;

/// phosphene — fosphor-style real-time spectrum analyzer.
#[derive(Parser, Debug)]
#[command(name = "phosphene", version, about)]
struct Cli {
    /// Render offscreen instead of opening a window, then write a PNG.
    /// Renders the selected --source; without one, the deterministic
    /// synthetic scene (the CI golden path).
    #[arg(long)]
    headless: bool,

    /// Number of frames to render in headless mode.
    #[arg(long, default_value_t = 240)]
    frames: u32,

    /// Output PNG path for headless mode.
    #[arg(long, default_value = "phosphene-headless.png")]
    out: PathBuf,

    /// Render width in pixels (headless mode).
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Render height in pixels (headless mode).
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// FFT size (power of two, 512-32768). Also the batch size: the block of
    /// input samples feeding one FFT (D-013).
    #[arg(long, default_value_t = 1024)]
    fft: usize,

    /// Input source: `siggen` (the built-in demo signal generator, default),
    /// `file:<path>` (raw IQ replay), `stdin` (raw IQ from a pipe), or
    /// `soapy[:device-args]` (a SoapySDR device, e.g. soapy:driver=uhd —
    /// needs a build with the phosphene-sources `soapy` feature).
    #[arg(long, default_value = "siggen")]
    source: String,

    /// SoapySDR device arguments for a soapy source, e.g. driver=uhd or
    /// driver=uhd,serial=XYZ — the same thing as the inline form
    /// `--source soapy:<args>`; give the arguments once, either way.
    #[arg(long)]
    device_args: Option<String>,

    /// Overall RX gain for a soapy source in dB, applied through Soapy's
    /// API. Omitted: the device keeps its own default gain.
    #[arg(long)]
    gain: Option<f64>,

    /// Raw IQ sample format of a file/stdin source: cf32, cs32, cs16, cs8,
    /// or cu8 (sc16 is an alias of cs16; ci32 and sc32 are aliases of cs32).
    /// Omitted, it is taken from the capture itself (D-055): a SigMF
    /// sidecar's core:datatype, else a recognised extension
    /// (.cf32/.cfile/.cs32/.ci32/.sc32/.cs16/.sc16/.cs8/.cu8).
    #[arg(long)]
    format: Option<String>,

    /// Sample rate in Hz. For a file/stdin source it enables throttled
    /// real-time replay and labels the frequency axis. Omitted, it is taken
    /// from the capture itself (D-055): a SigMF sidecar, else the
    /// `cap_<freq><unit>_<rate><unit>sps` file-name convention. With nothing
    /// to go on, replay runs unthrottled and the axis is labelled in
    /// normalised frequency — a rate is never invented (clarification C5).
    /// For a soapy source it is asked of the device; the displayed rate is
    /// always the device's own readback.
    #[arg(long)]
    rate: Option<f64>,

    /// Center frequency in Hz. For a file/stdin source it labels the axis;
    /// omitted, it is taken from the capture itself on the same D-055 chain
    /// as --rate (sidecar, then file name), and with nothing to go on the
    /// labels are relative Hz (FR-C3). For a soapy source the device is
    /// tuned to it.
    #[arg(long)]
    center: Option<f64>,

    /// Replay pace for a throttled file/stdin source (requires --rate):
    /// 1 is real time, >1 fast-forward, <1 slow motion.
    #[arg(long)]
    pace: Option<f64>,

    /// Rewind a regular file at end-of-file and keep replaying.
    #[arg(long = "loop")]
    loop_replay: bool,

    /// Replay as fast as the consumer accepts even when --rate is given
    /// (the rate then only labels the axis).
    #[arg(long)]
    unthrottled: bool,

    /// Enable the live Signal Inspector (FR-AU1): detection, tracking and
    /// measurement, shown as always-on in-place annotations. Only present in
    /// a binary built with `--features analyze`; the pure v1 display is
    /// always available untouched.
    #[cfg(feature = "analyze")]
    #[arg(long)]
    analyze: bool,

    /// The waterfall/persistence colormap to render with (FR-D8): p7
    /// (default), inferno, viridis, or turbo. The windowed view cycles
    /// these live with the `C` key (D-009); headless has no keyboard, so
    /// this is how a headless capture picks one — a look review across all
    /// four takes one `--headless` run per map.
    #[arg(long, default_value = "p7")]
    colormap: String,
}

/// Parses `--colormap`'s value, naming every valid choice on failure
/// (§8.4: a usage error must name the offending value).
fn parse_colormap(name: &str) -> Result<phosphene_render::Colormap, String> {
    phosphene_render::Colormap::ALL
        .into_iter()
        .find(|m| m.name() == name)
        .ok_or_else(|| {
            let valid: Vec<_> = phosphene_render::Colormap::ALL
                .iter()
                .map(|m| m.name())
                .collect();
            format!("--colormap {name:?} is not one of {}", valid.join(", "))
        })
}

/// What the windowed pipeline should consume, resolved from the CLI.
#[derive(Debug)]
struct SourceSetup {
    desc: SourceDesc,
    /// True when the source delivers under real-time pacing, so a full ring
    /// sheds whole batches (NFR-P3); false for pull-paced sources
    /// (clarification C5: "as fast as the consumer accepts").
    paced: bool,
}

/// Resolve `--source` and its companion flags into a [`SourceSetup`], with
/// §8.4-grade errors: every rejection names the flag and the offending value.
fn build_source(cli: &Cli) -> Result<SourceSetup, String> {
    let input = match cli.source.as_str() {
        "siggen" => {
            let irrelevant: &[(&str, bool)] = &[
                ("--format", cli.format.is_some()),
                ("--rate", cli.rate.is_some()),
                ("--center", cli.center.is_some()),
                ("--pace", cli.pace.is_some()),
                ("--loop", cli.loop_replay),
                ("--unthrottled", cli.unthrottled),
                ("--device-args", cli.device_args.is_some()),
                ("--gain", cli.gain.is_some()),
            ];
            if let Some((flag, _)) = irrelevant.iter().find(|(_, given)| *given) {
                return Err(format!(
                    "{flag} does not apply to the siggen demo scene, which \
                     defines its own rate and content"
                ));
            }
            return Ok(SourceSetup {
                desc: SourceDesc::SigGen(SigGenConfig::demo()),
                paced: false,
            });
        }
        s if s == "soapy" || s.starts_with("soapy:") => {
            let replay_only: &[(&str, bool)] = &[
                ("--format", cli.format.is_some()),
                ("--pace", cli.pace.is_some()),
                ("--loop", cli.loop_replay),
                ("--unthrottled", cli.unthrottled),
            ];
            if let Some((flag, _)) = replay_only.iter().find(|(_, given)| *given) {
                return Err(format!(
                    "{flag} only applies to file/stdin replay; a soapy device \
                     delivers live cf32 at its own pace"
                ));
            }
            let device_args = match (s.strip_prefix("soapy:"), cli.device_args.as_deref()) {
                (Some(_), Some(_)) => {
                    return Err(
                        "--source soapy:<args> and --device-args conflict: give the \
                         device arguments once, either way"
                            .to_owned(),
                    )
                }
                (Some(""), None) => {
                    return Err("--source soapy: is missing its device arguments (use \
                         soapy:driver=uhd,… or plain soapy to match any device)"
                        .to_owned())
                }
                (Some(inline), None) => inline,
                (None, Some(flag_args)) => flag_args,
                (None, None) => "",
            };
            return Ok(SourceSetup {
                desc: SourceDesc::Soapy(SoapySourceConfig {
                    device_args: device_args.to_owned(),
                    sample_rate_hz: cli.rate,
                    center_freq_hz: cli.center,
                    gain_db: cli.gain,
                }),
                // A radio delivers under real time: a full ring sheds whole
                // batches, counted (NFR-P3/D-013).
                paced: true,
            });
        }
        "stdin" => {
            if cli.loop_replay {
                return Err(
                    "--loop needs a rewindable regular file; stdin cannot rewind".to_owned(),
                );
            }
            Input::Stdin
        }
        s => match s.strip_prefix("file:") {
            Some("") => {
                return Err("--source file: is missing its path (use file:<path>)".to_owned())
            }
            Some(path) => Input::Path(PathBuf::from(path)),
            None => {
                return Err(format!(
                    "unknown --source {s:?}: expected siggen, stdin, file:<path>, \
                     or soapy[:device-args]"
                ))
            }
        },
    };

    if cli.device_args.is_some() {
        return Err("--device-args only applies to a soapy source".to_owned());
    }
    if cli.gain.is_some() {
        return Err(
            "--gain only applies to a soapy source; a file/stdin replay has no \
             hardware gain to set"
                .to_owned(),
        );
    }

    // D-055's precedence chain, walked once: what the flags said, then the
    // SigMF sidecar, then the file name, then nothing. Everything below
    // reads the resolved answer rather than `cli.rate` / `cli.center`, which
    // is what makes "a well-constructed filename and no other args" behave
    // exactly as typing the flags would.
    let meta = resolve_capture_metadata(cli, &input)?;
    let format = meta
        .format()
        .ok_or_else(|| format_required_message(&input))?;
    let pace = match (cli.unthrottled, cli.pace) {
        (true, Some(_)) => {
            return Err(
                "--pace and --unthrottled conflict: pick a throttled pace or none".to_owned(),
            )
        }
        (true, None) => ReplayPace::Unthrottled,
        (false, Some(factor)) => {
            if meta.rate().is_none() {
                return Err(format!(
                    "--pace {factor} needs a sample rate: pacing is relative to \
                     the stream's real-time rate, and a rate is never invented \
                     (clarification C5). Give --rate, or replay a capture whose \
                     SigMF sidecar or file name states it (D-055)"
                ));
            }
            ReplayPace::Factor(factor)
        }
        (false, None) => ReplayPace::RealTime,
    };

    Ok(SourceSetup {
        // Throttled replay arrives under time pressure like a live radio, so
        // the ring-full policy is NFR-P3's shed-whole-batches; without a rate
        // (or with --unthrottled) the consumer itself is the pace (C5). A
        // rate the capture declared is a rate: an inferred one paces replay
        // exactly as a typed one does, which is the whole point of D-055.
        paced: meta.rate().is_some() && !cli.unthrottled,
        desc: SourceDesc::File(FileSourceConfig {
            input,
            format,
            sample_rate_hz: meta.rate(),
            center_freq_hz: meta.center(),
            pace,
            loop_replay: cli.loop_replay,
        }),
    })
}

/// Ask the capture what it is (D-055): walk `explicit flag > SigMF sidecar >
/// file name > nothing` for the sample rate, the centre frequency and the IQ
/// format, all at once.
///
/// `stdin` has no name to read, so only the flags apply to it. The only
/// failures here are an explicitly malformed `--format` and a SigMF sidecar
/// that exists and cannot be believed — a file name that does not match the
/// convention is silent, because a hint that does not fire is not an error.
fn resolve_capture_metadata(cli: &Cli, input: &Input) -> Result<CaptureMetadata, String> {
    let flag_format = cli
        .format
        .as_deref()
        .map(|name| {
            name.parse::<IqFormat>()
                .map_err(|e| format!("--format: {e}"))
        })
        .transpose()?;
    let path = match input {
        Input::Path(path) => Some(path.as_path()),
        Input::Stdin => None,
    };
    metadata::resolve(path, cli.rate, cli.center, flag_format).map_err(|e| e.to_string())
}

/// §8.4-grade message for the one thing that has no honest fallback: nothing,
/// anywhere, said what the bytes are.
fn format_required_message(input: &Input) -> String {
    format!(
        "--format is required for {}: raw IQ has no self-describing header, and \
         guessing would silently show garbage (expected cf32, cs32, cs16, cs8, or \
         cu8). Nothing declared the layout — no SigMF sidecar with a core:datatype, \
         and no recognised extension \
         (.cf32/.cfile/.cs32/.ci32/.sc32/.cs16/.sc16/.cs8/.cu8)",
        match input {
            Input::Stdin => "stdin".to_owned(),
            Input::Path(p) => format!("{} (extension not recognised)", p.display()),
        }
    )
}

/// Whether a headless run renders the built-in synthetic scene rather than
/// a real source: exactly when `--source` resolved to the default siggen
/// demo. This is M2-A's invariant seam (D-041/D-014): no `--source` ⇒ the
/// existing deterministic scene, unchanged — CI's goldens consume it.
fn headless_uses_synthetic(setup: &SourceSetup) -> bool {
    matches!(setup.desc, SourceDesc::SigGen(_))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // FR-AU1: the flag only exists in a binary built with `--features
    // analyze` (Cli::analyze is itself cfg-gated); everywhere else in this
    // function reads this plain bool so no other call site needs a #[cfg].
    #[cfg(feature = "analyze")]
    let analyze = cli.analyze;
    #[cfg(not(feature = "analyze"))]
    let analyze = false;
    // `--source` and its companion flags resolve identically in both modes
    // (M2-A build step 1: every flag M1-E landed applies unchanged).
    let result = parse_colormap(&cli.colormap).and_then(|colormap| {
        build_source(&cli).and_then(|setup| {
            if !cli.headless {
                window::run(cli.fft, setup, analyze, colormap)
            } else if headless_uses_synthetic(&setup) {
                // No --source: the deterministic synthetic scene, byte-for-
                // byte as before — the D-014 golden path.
                headless::run(
                    cli.width, cli.height, cli.frames, cli.fft, &cli.out, analyze, colormap,
                )
            } else {
                headless::run_source(
                    cli.width, cli.height, cli.frames, cli.fft, &cli.out, setup, analyze, colormap,
                )
            }
        })
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("phosphene: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("phosphene").chain(args.iter().copied())).unwrap()
    }

    #[test]
    fn zero_arguments_defaults_to_the_siggen_demo() {
        let setup = build_source(&cli(&[])).unwrap();
        assert!(matches!(setup.desc, SourceDesc::SigGen(_)));
        // Pull-paced: the generator runs as fast as the consumer accepts, so
        // it can never honestly drop.
        assert!(!setup.paced);
    }

    #[test]
    fn file_source_with_rate_is_time_paced() {
        let setup = build_source(&cli(&["--source", "file:cap.cs8", "--rate", "2400000"])).unwrap();
        assert!(setup.paced);
        let SourceDesc::File(config) = setup.desc else {
            panic!("expected a file source");
        };
        assert_eq!(config.format, IqFormat::Cs8);
        assert_eq!(config.sample_rate_hz, Some(2_400_000.0));
        assert_eq!(config.pace, ReplayPace::RealTime);
    }

    #[test]
    fn file_source_without_rate_is_pull_paced_and_rateless() {
        // Clarification C5: no --rate → unthrottled replay, no invented rate.
        let setup = build_source(&cli(&["--source", "file:cap.cf32"])).unwrap();
        assert!(!setup.paced);
        let SourceDesc::File(config) = setup.desc else {
            panic!("expected a file source");
        };
        assert_eq!(config.sample_rate_hz, None);
    }

    #[test]
    fn unthrottled_with_rate_keeps_the_rate_for_the_axis_only() {
        let setup = build_source(&cli(&[
            "--source",
            "file:cap.cu8",
            "--rate",
            "1000000",
            "--unthrottled",
        ]))
        .unwrap();
        assert!(!setup.paced);
        let SourceDesc::File(config) = setup.desc else {
            panic!("expected a file source");
        };
        assert_eq!(config.pace, ReplayPace::Unthrottled);
        assert_eq!(config.sample_rate_hz, Some(1_000_000.0));
    }

    #[test]
    fn stdin_requires_an_explicit_format() {
        let err = build_source(&cli(&["--source", "stdin"])).unwrap_err();
        assert!(err.contains("--format"), "unhelpful: {err}");
        let setup = build_source(&cli(&["--source", "stdin", "--format", "cu8"])).unwrap();
        let SourceDesc::File(config) = setup.desc else {
            panic!("expected a stdin source");
        };
        assert_eq!(config.input, Input::Stdin);
        assert_eq!(config.format, IqFormat::Cu8);
    }

    // ————————————————————————————————————————————————————————————————
    // D-055 — the capture tells us what it is.
    // ————————————————————————————————————————————————————————————————

    /// A scratch directory of captures and sidecars, removed on drop.
    struct Corpus {
        dir: PathBuf,
    }

    impl Corpus {
        fn new(tag: &str) -> Corpus {
            let dir = std::env::temp_dir().join(format!(
                "phosphene-m3a-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Corpus { dir }
        }

        /// Write `bytes` under `name` and return the `file:<path>` argument.
        fn capture(&self, name: &str, bytes: &[u8]) -> String {
            let path = self.dir.join(name);
            std::fs::write(&path, bytes).expect("write the capture");
            format!("file:{}", path.display())
        }

        fn sidecar(&self, name: &str, json: &str) {
            std::fs::write(self.dir.join(name), json).expect("write the sidecar");
        }
    }

    impl Drop for Corpus {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    /// The committed excerpt of the **real** SDRangel `ci32_le` recording that
    /// motivated M3-C, and its own sidecar — bytes phosphene did not write.
    /// Provenance and regeneration: `phosphene-sources/tests/corpus/README.md`.
    fn corpus_ci32_le_fixture() -> (Vec<u8>, String) {
        const DIR: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../phosphene-sources/tests/corpus"
        );
        let data = std::fs::read(format!("{DIR}/sdrangel-pluto-ci32le-head.sigmf-data"))
            .expect("the committed ci32_le corpus prefix");
        let meta = std::fs::read_to_string(format!("{DIR}/sdrangel-pluto-ci32le-head.sigmf-meta"))
            .expect("the recording's own sidecar");
        (data, meta)
    }

    /// A steady full-quadrature tone encoded through the production encoder
    /// for `format` — so the bytes on disk are exactly what the decoder under
    /// test is expected to read back.
    fn tone_bytes(format: IqFormat, n: usize, batches: usize) -> Vec<u8> {
        let samples: Vec<phosphene_sources::Complex<f32>> = (0..n * batches)
            .map(|i| {
                let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
                let (im, re) = phase.sin_cos();
                phosphene_sources::Complex::new(re * 0.5, im * 0.5)
            })
            .collect();
        let mut bytes = Vec::new();
        phosphene_sources::format::encode_append(format, &samples, &mut bytes);
        bytes
    }

    fn file_config(setup: SourceSetup) -> FileSourceConfig {
        match setup.desc {
            SourceDesc::File(config) => config,
            other => panic!("expected a file source, got {other:?}"),
        }
    }

    /// The owner's request at the CLI boundary: a well-constructed filename
    /// and **no other args** yields all three values, and the replay is paced
    /// exactly as it would be had `--rate` been typed.
    #[test]
    fn a_conventional_filename_alone_supplies_rate_center_and_format() {
        let corpus = Corpus::new("name");
        let arg = corpus.capture(
            "cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16",
            &[0u8; 16],
        );
        let setup = build_source(&cli(&["--source", &arg])).unwrap();
        // An inferred rate paces replay like a typed one — that is the point.
        assert!(setup.paced);
        let config = file_config(setup);
        assert_eq!(config.sample_rate_hz, Some(30.72e6));
        assert_eq!(config.center_freq_hz, Some(1.985e9));
        // `.sc16` is an alias, so the format is the very same Cs16.
        assert_eq!(config.format, IqFormat::Cs16);
        assert_eq!(config.pace, ReplayPace::RealTime);
    }

    /// The corpus's own variety, including the `MHz` case that a hard-coded
    /// GHz scale would render a thousand times too wide, and the tails that
    /// are ignored rather than parsed.
    #[test]
    fn the_corpus_filename_shapes_all_resolve_through_the_cli() {
        let corpus = Corpus::new("shapes");
        let cases: [(&str, f64, f64); 3] = [
            ("cap_632MHz_30p72Msps.sc16", 632e6, 30.72e6),
            (
                "cap_1p985GHz_30p72Msps_arfcn397000_gscn4961_20260609-143210-PDT.sc16",
                1.985e9,
                30.72e6,
            ),
            ("cap_634p54MHz_7p68Msps.cs16", 634.54e6, 7.68e6),
        ];
        for (name, center, rate) in cases {
            let arg = corpus.capture(name, &[0u8; 16]);
            let config = file_config(build_source(&cli(&["--source", &arg])).unwrap());
            assert_eq!(config.center_freq_hz, Some(center), "{name}");
            assert_eq!(config.sample_rate_hz, Some(rate), "{name}");
        }
    }

    /// Rung 2 outranks rung 3, and rung 1 outranks both — asserted through
    /// the production `build_source`, over real files on disk.
    #[test]
    fn precedence_is_flag_then_sidecar_then_filename_then_nothing() {
        let corpus = Corpus::new("precedence");
        // The name says 632 MHz / 30.72 Msps; the sidecar disagrees on both.
        let arg = corpus.capture("cap_632MHz_30p72Msps.sc16", &[0u8; 16]);
        corpus.sidecar(
            "cap_632MHz_30p72Msps.sigmf-meta",
            r#"{"global":{"core:sample_rate":7680000.0,"core:datatype":"ci16_le"},
                "captures":[{"core:sample_start":0,"core:frequency":1876954000.0}]}"#,
        );

        // Rung 2 beats rung 3: a declaration beats a hint.
        let config = file_config(build_source(&cli(&["--source", &arg])).unwrap());
        assert_eq!(config.sample_rate_hz, Some(7.68e6));
        assert_eq!(config.center_freq_hz, Some(1_876_954_000.0));

        // Rung 1 beats rung 2: an explicit --rate beats a sidecar that
        // disagrees, and does not disturb what the sidecar knows about the
        // centre.
        let config =
            file_config(build_source(&cli(&["--source", &arg, "--rate", "2400000"])).unwrap());
        assert_eq!(config.sample_rate_hz, Some(2_400_000.0));
        assert_eq!(config.center_freq_hz, Some(1_876_954_000.0));

        // …and all the way down, flags beat everything.
        let config = file_config(
            build_source(&cli(&[
                "--source", &arg, "--rate", "2400000", "--center", "100e6", "--format", "cu8",
            ]))
            .unwrap(),
        );
        assert_eq!(config.sample_rate_hz, Some(2_400_000.0));
        assert_eq!(config.center_freq_hz, Some(100e6));
        assert_eq!(config.format, IqFormat::Cu8);

        // Rung 4 is nothing at all: a name that is not the convention and no
        // sidecar leaves the rate absent, and an absent rate is pull-paced.
        let bare = corpus.capture("gnb_n3.cs16", &[0u8; 16]);
        let setup = build_source(&cli(&["--source", &bare])).unwrap();
        assert!(!setup.paced);
        let config = file_config(setup);
        assert_eq!(config.sample_rate_hz, None);
        assert_eq!(config.center_freq_hz, None);
    }

    /// D-055's two named negatives plus a malformed name: each falls through
    /// **silently** — no error, no half-guessed pair — and stdin, which has no
    /// name at all, is unaffected by any of it.
    #[test]
    fn an_unmatched_name_falls_through_silently() {
        let corpus = Corpus::new("negatives");
        for name in [
            "b13_751.cs16",
            "gnb_n3.cs16",
            "cap_1p9p85GHz_30p72Msps.cs16",
            "cap_c634p54M_s7p68M.sc16",
        ] {
            let arg = corpus.capture(name, &[0u8; 16]);
            let setup = build_source(&cli(&["--source", &arg]))
                .unwrap_or_else(|e| panic!("{name} must fall through silently, got: {e}"));
            let config = file_config(setup);
            assert_eq!(config.sample_rate_hz, None, "{name}");
            assert_eq!(config.center_freq_hz, None, "{name}");
        }
        let config =
            file_config(build_source(&cli(&["--source", "stdin", "--format", "cs16"])).unwrap());
        assert_eq!(config.sample_rate_hz, None);
    }

    /// An inferred rate is a rate: `--pace` no longer needs `--rate` typed
    /// out when the capture already said what its rate is.
    #[test]
    fn pace_accepts_a_rate_the_capture_declared() {
        let corpus = Corpus::new("pace");
        let arg = corpus.capture("cap_632MHz_30p72Msps.sc16", &[0u8; 16]);
        let config = file_config(build_source(&cli(&["--source", &arg, "--pace", "2"])).unwrap());
        assert_eq!(config.pace, ReplayPace::Factor(2.0));
        // …and with nothing declaring one, the refusal still says so.
        let bare = corpus.capture("gnb_n3.cs16", &[0u8; 16]);
        let err = build_source(&cli(&["--source", &bare, "--pace", "2"])).unwrap_err();
        assert!(err.contains("--rate"), "unhelpful: {err}");
    }

    /// D-055: a `core:datatype` we cannot decode is a clear error naming it,
    /// never a silent fallback to the nearest layout we do have — and an
    /// explicit `--format`, which outranks the sidecar, opens the file anyway.
    ///
    /// `ci32_be` is the case D-062 deliberately left refused: its bytes are
    /// the bytes of a `ci32_le` sample in the other order, so the layout we
    /// *do* have would decode every sample to a plausible wrong number.
    #[test]
    fn an_unsupported_sidecar_datatype_is_refused_by_name() {
        let corpus = Corpus::new("ci32be");
        let arg = corpus.capture("capture.sigmf-data", &[0u8; 16]);
        corpus.sidecar(
            "capture.sigmf-meta",
            r#"{"global":{"core:sample_rate":7680000.0,"core:datatype":"ci32_be"},
                "captures":[{"core:frequency":1876954000.0}]}"#,
        );
        let err = build_source(&cli(&["--source", &arg])).unwrap_err();
        assert!(err.contains("ci32_be"), "must name the datatype: {err}");
        assert!(
            err.contains("capture.sigmf-meta"),
            "must name the file: {err}"
        );

        let config =
            file_config(build_source(&cli(&["--source", &arg, "--format", "cs16"])).unwrap());
        assert_eq!(config.format, IqFormat::Cs16);
        // The rate the sidecar declared still applies: overriding the layout
        // does not throw away the rest of the declaration.
        assert_eq!(config.sample_rate_hz, Some(7.68e6));
    }

    /// A sidecar with no `.sigmf-data` name still supplies the format, which
    /// is the only way an extension-less capture opens with no `--format`.
    #[test]
    fn a_sidecar_datatype_supplies_a_format_no_extension_could() {
        let corpus = Corpus::new("datatype");
        let arg = corpus.capture("capture.raw", &[0u8; 16]);
        corpus.sidecar(
            "capture.sigmf-meta",
            r#"{"global":{"core:datatype":"cu8"}}"#,
        );
        let config = file_config(build_source(&cli(&["--source", &arg])).unwrap());
        assert_eq!(config.format, IqFormat::Cu8);
    }

    /// **M3-C's reason to exist, at the CLI boundary** (D-062): the fleet's
    /// own capture declares `core:datatype: "ci32_le"`, and until this lane
    /// the honest answer was a named refusal. The same sidecar now opens —
    /// as `cs32`, at the 7.68 MS/s and 1876.954 MHz it declares.
    #[test]
    fn the_corpus_ci32_le_sidecar_opens_instead_of_refusing() {
        let corpus = Corpus::new("ci32le");
        // The corpus file's own name and declaration, verbatim.
        let arg = corpus.capture(
            "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data",
            &tone_bytes(IqFormat::Cs32, 512, 2),
        );
        corpus.sidecar(
            "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta",
            r#"{"global":{"core:author":"SDRangel","core:datatype":"ci32_le",
                          "core:sample_rate":7680000.0,"core:version":"1.2.3"},
                "captures":[{"core:sample_start":0,"core:frequency":1876954000.0}],
                "annotations":[]}"#,
        );
        // No --format, no --rate, no --center: the capture says all three.
        let setup = build_source(&cli(&["--source", &arg])).unwrap();
        assert!(setup.paced, "a declared rate paces replay");
        let config = file_config(setup);
        assert_eq!(config.format, IqFormat::Cs32);
        assert_eq!(config.sample_rate_hz, Some(7.68e6));
        assert_eq!(config.center_freq_hz, Some(1_876_954_000.0));
    }

    /// Every spelling of the one layout resolves to the one variant (D-062):
    /// `--format cs32`, `--format ci32`, a `.sc32` extension and a sidecar's
    /// `ci32_le` all arrive at `IqFormat::Cs32` — and `ci32_be` still does
    /// not arrive anywhere.
    #[test]
    fn every_cs32_spelling_resolves_to_the_one_variant() {
        let corpus = Corpus::new("spellings");
        let raw = corpus.capture("capture.raw", &[0u8; 16]);
        for spelling in ["cs32", "ci32", "sc32"] {
            let config =
                file_config(build_source(&cli(&["--source", &raw, "--format", spelling])).unwrap());
            assert_eq!(config.format, IqFormat::Cs32, "--format {spelling}");
        }
        for name in ["capture.cs32", "capture.sc32", "capture.ci32"] {
            let arg = corpus.capture(name, &[0u8; 16]);
            let config = file_config(build_source(&cli(&["--source", &arg])).unwrap());
            assert_eq!(config.format, IqFormat::Cs32, "{name}");
        }
        let declared = corpus.capture("declared.sigmf-data", &[0u8; 16]);
        corpus.sidecar(
            "declared.sigmf-meta",
            r#"{"global":{"core:datatype":"ci32_le"}}"#,
        );
        let config = file_config(build_source(&cli(&["--source", &declared])).unwrap());
        assert_eq!(config.format, IqFormat::Cs32);
        // And the refusal that stays a refusal, by name.
        let err = build_source(&cli(&["--source", &raw, "--format", "ci32_be"])).unwrap_err();
        assert!(err.contains("ci32_be"), "must name what was asked: {err}");
        assert!(err.contains("cs32"), "must say what is supported: {err}");
    }

    /// **The corpus capture rendered end to end** (D-031, the owner's actual
    /// reason for M3-C): the real `ci32_le` sidecar's declaration — 7.68 MS/s
    /// at 1876.954 MHz — drives the production headless frame loop and lands
    /// on the frequency axis, with no flags typed at all.
    ///
    /// The bytes and the sidecar are **the recording's own**, not a tone we
    /// encoded under its name: a `cs32` round trip through our own encoder
    /// could not have failed for the reason this format exists, which is that
    /// a real `ci32_le` file in the corpus would not open.
    #[test]
    fn the_corpus_ci32_le_capture_renders_its_declared_axis() {
        use phosphene_render::axis;
        use phosphene_render::layout::DisplayParams;

        let (w, h, n) = (640u32, 360u32, 512usize);
        let corpus = Corpus::new("ci32-e2e");
        let (data, sidecar) = corpus_ci32_le_fixture();
        let arg = corpus.capture(
            "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data",
            &data,
        );
        corpus.sidecar(
            "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta",
            &sidecar,
        );
        let out = corpus.dir.join("ci32.png");

        let setup = build_source(&cli(&["--headless", "--source", &arg])).unwrap();
        let outcome = crate::headless::run_source_frames(
            w,
            h,
            20,
            n,
            &out,
            setup,
            false,
            phosphene_render::Colormap::P7,
        )
        .expect("the headless loop must run over the ci32_le capture");
        assert!(outcome.frames_rendered > 0, "nothing rendered");

        let expected = DisplayParams {
            sample_rate: Some(7.68e6),
            center: Some(1_876_954_000.0),
            ..DisplayParams::default()
        };
        for tick in axis::freq_ticks(&expected) {
            assert!(
                outcome.label_texts.contains(&tick.label),
                "the axis must be labelled {:?} (from the sidecar alone); labels: {:?}",
                tick.label,
                outcome.label_texts
            );
        }
        assert!(
            outcome.label_texts.iter().any(|t| t == "7.68M"),
            "the span must read 7.68M; labels: {:?}",
            outcome.label_texts
        );
        for t in &outcome.label_texts {
            assert!(
                !t.to_ascii_lowercase().contains("norm"),
                "a declared rate must not render a normalised label: {t:?}"
            );
        }
    }

    /// **The owner's request, stated as a test** (D-031: through the
    /// production path, end to end).
    ///
    /// A capture named to the convention, opened with **no `--rate`, no
    /// `--center` and no `--format`**, drives the real headless frame loop —
    /// the same `build_source` → `SourceDesc` → pipeline → chrome chain the
    /// window runs — and the frequency axis it renders is the absolute one
    /// the file name states: ticks around 1.985 GHz over a 30.72 MHz span,
    /// and not one normalised label anywhere.
    #[test]
    fn a_conventional_filename_alone_renders_the_right_absolute_axis() {
        use phosphene_render::axis;
        use phosphene_render::layout::DisplayParams;

        let (w, h, n) = (640u32, 360u32, 512usize);
        let corpus = Corpus::new("e2e");
        let arg = corpus.capture(
            "cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16",
            &tone_bytes(IqFormat::Cs16, n, 60),
        );
        let out = corpus.dir.join("axis.png");

        // No --rate. No --center. No --format.
        let setup = build_source(&cli(&["--headless", "--source", &arg])).unwrap();
        let outcome = crate::headless::run_source_frames(
            w,
            h,
            20,
            n,
            &out,
            setup,
            false,
            phosphene_render::Colormap::P7,
        )
        .expect("the headless loop must run over the named capture");
        assert!(outcome.frames_rendered > 0, "nothing rendered");

        // The labels an absolute axis at this centre and span produces —
        // computed by the same `axis` mapping the chrome uses, so this
        // asserts the *values reached the display*, not a formatting detail.
        let expected = DisplayParams {
            sample_rate: Some(30.72e6),
            center: Some(1.985e9),
            ..DisplayParams::default()
        };
        for tick in axis::freq_ticks(&expected) {
            assert!(
                outcome.label_texts.contains(&tick.label),
                "the axis must be labelled {:?} (from the file name alone); labels: {:?}",
                tick.label,
                outcome.label_texts
            );
        }
        // The span readout tells the same story as the axis.
        assert!(
            outcome.label_texts.iter().any(|t| t == "30.72M"),
            "the span must read 30.72M; labels: {:?}",
            outcome.label_texts
        );
        // And nothing anywhere fell back to the rate-less axis.
        for t in &outcome.label_texts {
            assert!(
                !t.to_ascii_lowercase().contains("norm"),
                "a declared rate must not render a normalised label: {t:?}"
            );
        }
    }

    /// The other end of the same chain (C5 / D-028 §2, the assertion M1-C
    /// already uses): a capture whose name is **not** the convention and has
    /// no sidecar keeps the normalised axis, with no `Hz` anywhere. Inference
    /// that fired here would be the failure D-055 exists to prevent.
    #[test]
    fn an_unmatched_name_still_renders_a_normalised_axis_with_no_hz() {
        let (w, h, n) = (640u32, 360u32, 512usize);
        let corpus = Corpus::new("e2e-norm");
        let arg = corpus.capture("gnb_n3.cs16", &tone_bytes(IqFormat::Cs16, n, 20));
        let out = corpus.dir.join("norm.png");

        let setup = build_source(&cli(&["--headless", "--source", &arg])).unwrap();
        let outcome = crate::headless::run_source_frames(
            w,
            h,
            20,
            n,
            &out,
            setup,
            false,
            phosphene_render::Colormap::P7,
        )
        .expect("the headless loop must run over the unmatched capture");
        assert!(outcome.frames_rendered > 0, "nothing rendered");
        assert!(
            outcome.label_texts.iter().any(|t| t == "±0.5 NORM"),
            "the span must say normalised; labels: {:?}",
            outcome.label_texts
        );
        for t in &outcome.label_texts {
            assert!(
                !t.to_ascii_lowercase().contains("hz"),
                "a capture nothing declared a rate for rendered a Hz label (C5): {t:?}"
            );
        }
    }

    #[test]
    fn unrecognised_extension_requires_format() {
        let err = build_source(&cli(&["--source", "file:capture.bin"])).unwrap_err();
        assert!(err.contains("--format"), "unhelpful: {err}");
        assert!(err.contains("capture.bin"), "unhelpful: {err}");
    }

    #[test]
    fn pace_without_rate_is_rejected_with_the_reason() {
        let err = build_source(&cli(&["--source", "file:cap.cf32", "--pace", "2"])).unwrap_err();
        assert!(err.contains("--rate"), "unhelpful: {err}");
    }

    #[test]
    fn pace_and_unthrottled_conflict() {
        let err = build_source(&cli(&[
            "--source",
            "file:cap.cf32",
            "--rate",
            "1e6",
            "--pace",
            "2",
            "--unthrottled",
        ]))
        .unwrap_err();
        assert!(err.contains("--pace") && err.contains("--unthrottled"));
    }

    #[test]
    fn siggen_rejects_file_only_flags() {
        for args in [
            &["--rate", "1e6"][..],
            &["--format", "cu8"][..],
            &["--loop"][..],
        ] {
            let err = build_source(&cli(args)).unwrap_err();
            assert!(err.contains("siggen"), "unhelpful: {err}");
        }
    }

    #[test]
    fn stdin_rejects_loop() {
        let err =
            build_source(&cli(&["--source", "stdin", "--format", "cu8", "--loop"])).unwrap_err();
        assert!(err.contains("rewind"), "unhelpful: {err}");
    }

    /// M2-A's invariant seam (D-041/D-014): with no `--source`, a headless
    /// run dispatches to the untouched synthetic renderer — the CI golden
    /// path — and never to the source-driven loop. An explicit
    /// `--source siggen` is the same resolution, so it takes the same path.
    #[test]
    fn headless_without_source_renders_the_synthetic_scene() {
        for args in [
            &["--headless"][..],
            &["--headless", "--source", "siggen"][..],
        ] {
            let setup = build_source(&cli(args)).unwrap();
            assert!(
                headless_uses_synthetic(&setup),
                "no --source must keep the deterministic golden path ({args:?})"
            );
        }
    }

    /// D-041: a file or stdin source selects the source-driven headless loop.
    #[test]
    fn headless_with_a_real_source_renders_that_source() {
        for args in [
            &["--headless", "--source", "file:cap.cf32"][..],
            &["--headless", "--source", "stdin", "--format", "cu8"][..],
            // A live radio straight to a PNG falls out of the uniform
            // source abstraction — deliberately not special-cased.
            &["--headless", "--source", "soapy"][..],
        ] {
            let setup = build_source(&cli(args)).unwrap();
            assert!(
                !headless_uses_synthetic(&setup),
                "a real source must not fall back to the synthetic scene ({args:?})"
            );
        }
    }

    #[test]
    fn unknown_source_names_the_valid_forms() {
        let err = build_source(&cli(&["--source", "hackrf"])).unwrap_err();
        assert!(err.contains("hackrf") && err.contains("file:<path>"));
        let err = build_source(&cli(&["--source", "file:"])).unwrap_err();
        assert!(err.contains("path"), "unhelpful: {err}");
    }

    #[test]
    fn soapy_source_maps_the_full_request_and_is_time_paced() {
        let setup = build_source(&cli(&[
            "--source",
            "soapy:driver=uhd,serial=EXAMPLE1",
            "--rate",
            "2048000",
            "--center",
            "100e6",
            "--gain",
            "40",
        ]))
        .unwrap();
        // A radio delivers under real time — ring-full sheds counted batches.
        assert!(setup.paced);
        let SourceDesc::Soapy(config) = setup.desc else {
            panic!("expected a soapy source");
        };
        assert_eq!(config.device_args, "driver=uhd,serial=EXAMPLE1");
        assert_eq!(config.sample_rate_hz, Some(2_048_000.0));
        assert_eq!(config.center_freq_hz, Some(100e6));
        assert_eq!(config.gain_db, Some(40.0));
    }

    #[test]
    fn bare_soapy_matches_any_device_with_device_defaults() {
        let setup = build_source(&cli(&["--source", "soapy"])).unwrap();
        let SourceDesc::Soapy(config) = setup.desc else {
            panic!("expected a soapy source");
        };
        assert_eq!(config.device_args, "");
        assert_eq!(config.sample_rate_hz, None);
        assert_eq!(config.center_freq_hz, None);
        assert_eq!(config.gain_db, None);
    }

    #[test]
    fn device_args_flag_is_the_inline_form() {
        let setup = build_source(&cli(&[
            "--source",
            "soapy",
            "--device-args",
            "driver=hackrf",
        ]))
        .unwrap();
        let SourceDesc::Soapy(config) = setup.desc else {
            panic!("expected a soapy source");
        };
        assert_eq!(config.device_args, "driver=hackrf");

        let err = build_source(&cli(&[
            "--source",
            "soapy:driver=uhd",
            "--device-args",
            "driver=hackrf",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--device-args") && err.contains("once"),
            "unhelpful: {err}"
        );

        let err = build_source(&cli(&["--source", "soapy:"])).unwrap_err();
        assert!(err.contains("device arguments"), "unhelpful: {err}");
    }

    #[test]
    fn soapy_rejects_replay_only_flags() {
        for args in [
            &["--source", "soapy", "--format", "cu8"][..],
            &["--source", "soapy", "--pace", "2"][..],
            &["--source", "soapy", "--loop"][..],
            &["--source", "soapy", "--unthrottled"][..],
        ] {
            let err = build_source(&cli(args)).unwrap_err();
            assert!(err.contains("file/stdin"), "unhelpful: {err}");
        }
    }

    #[test]
    fn soapy_only_flags_are_rejected_elsewhere() {
        let err = build_source(&cli(&["--source", "file:cap.cf32", "--gain", "20"])).unwrap_err();
        assert!(
            err.contains("--gain") && err.contains("soapy"),
            "unhelpful: {err}"
        );
        let err = build_source(&cli(&[
            "--source",
            "stdin",
            "--format",
            "cu8",
            "--device-args",
            "driver=uhd",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--device-args") && err.contains("soapy"),
            "unhelpful: {err}"
        );
        let err = build_source(&cli(&["--gain", "20"])).unwrap_err();
        assert!(err.contains("siggen"), "unhelpful: {err}");
    }

    /// The S1-A production-path hardware seal: `--source soapy` opens a real
    /// device through the exact chain windowed mode runs — CLI parse →
    /// [`SourceDesc`] → [`pipeline::open_source`] → [`pipeline::Pipeline`] —
    /// and live spectra with honest FR-D11 figures come out the other end.
    /// A backend that opens a device and streams zeros fails the trace
    /// assertion (D-031/D-032/D-033).
    ///
    /// Skips (passing) on a build without the `soapy` feature and on a
    /// machine without a radio, so CI stays green; on the capture nodes it
    /// runs for real.
    ///
    /// **This test needs the CPU to itself (FL-3/D-103).** It is real-time
    /// hardware I/O: sharing a thread pool with CPU-heavy tests (the
    /// `window` render suite) starves its source/compute threads, and the
    /// ring sheds batches it would otherwise keep up with — a test-harness
    /// artifact, not a product defect (confirmed on real hardware: 10/10
    /// isolated runs at 100% processed, failing every run under forced CPU
    /// oversubscription instead). CI runs it alone, its own invocation,
    /// skipped from the bundled run beside it
    /// (`.github/workflows/ci.yml`); see
    /// `ci_soapy_hardware_test_runs_in_its_own_invocation_on_every_soapy_leg`
    /// below, which fails if that separation is ever lost.
    #[test]
    fn soapy_source_runs_end_to_end_through_the_production_path() {
        use phosphene_core::{WindowKind, DBFS_FLOOR};
        use std::time::{Duration, Instant};

        let setup = build_source(&cli(&[
            "--source", "soapy", "--rate", "2048000", "--center", "100e6", "--gain", "40",
        ]))
        .unwrap();
        let paced = setup.paced;
        let source = match crate::pipeline::open_source(&setup.desc) {
            Ok(source) => source,
            Err(e)
                if e.contains("no SoapySDR support")
                    || e.contains("no SoapySDR devices")
                    || e.contains("no SoapySDR radio") =>
            {
                eprintln!("SKIP: {e}");
                return;
            }
            Err(e) => panic!("--source soapy failed to open: {e}"),
        };

        // Metadata is device readback (C5), visible through the same seam
        // the window reads.
        let rate = source
            .meta()
            .sample_rate_hz
            .expect("hardware knows its rate");
        assert!(
            (rate - 2_048_000.0).abs() / 2_048_000.0 < 0.01,
            "readback rate {rate}"
        );

        let pipeline = crate::pipeline::Pipeline::start(
            source,
            crate::pipeline::PipelineConfig {
                fft_size: 1024,
                window: WindowKind::Hann,
                paced,
                backpressure: false,
                db_bottom: -120.0,
                db_top: 0.0,
            },
        )
        .expect("start the pipeline over the soapy source");

        // ≥ 0.25 s of real capture resolved into FFTs.
        let deadline = Instant::now() + Duration::from_secs(10);
        while pipeline.health().batches_completed < 500 {
            if let Some(e) = pipeline.source_error() {
                panic!("source failed mid-stream: {e}");
            }
            assert!(Instant::now() < deadline, "no spectra after 10 s");
            std::thread::sleep(Duration::from_millis(10));
        }

        let snap = pipeline.health();
        assert!(
            snap.processed_pct >= 99.0,
            "pipeline sheds at a rate it should sustain: {snap:?}"
        );
        let mut trace = Vec::new();
        assert!(pipeline.copy_trace(&mut trace), "no live trace published");
        assert!(
            trace.iter().any(|&bin| bin > DBFS_FLOOR + 10.0),
            "trace is at the floor — the device is streaming but nothing real arrives"
        );
    }

    /// FL-3 diagnostic instrument: the same real-time-paced `RingSink` path
    /// the hardware test above exercises (`PipelineConfig::paced == true`),
    /// driven by a looped file replay instead of a radio, so the throughput
    /// question — does the pipeline sustain 2.048 MS/s at FFT 1024 while the
    /// CPU is oversubscribed? — can be asked on any machine, hardware or not
    /// (`specs/001-rtsa-v1/lanes/fl-3-soapy-throughput-flake.md`).
    ///
    /// The contention is manufactured directly (busy-spin threads at 4x the
    /// visible core count) rather than borrowed from ambient load, so the
    /// result is deterministic on an otherwise-idle machine: this is what
    /// let the diagnosis be confirmed on a 16-core workstation where the
    /// ordinary suite never oversubscribes enough to reproduce it.
    ///
    /// `#[ignore]`: running it is the whole point of a CPU-contention test,
    /// so it must never join the ordinary suite it is diagnosing — that
    /// would be exactly the flake this lane exists to explain. Run
    /// explicitly: `cargo test --bin phosphene -- --ignored
    /// paced_stand_in_sheds_under_forced_cpu_oversubscription --nocapture`.
    #[test]
    #[ignore = "manufactures CPU oversubscription on purpose — run explicitly, never in the default suite"]
    fn paced_stand_in_sheds_under_forced_cpu_oversubscription() {
        use phosphene_core::WindowKind;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let corpus = Corpus::new("fl3-stand-in");
        let sps = 2_048_000.0f32;
        let n_samples: usize = 1 << 16;
        let mut bytes = Vec::with_capacity(n_samples * 8);
        for i in 0..n_samples {
            let phase = 2.0 * std::f32::consts::PI * 100_000.0 * (i as f32) / sps;
            bytes.extend_from_slice(&phase.cos().to_le_bytes());
            bytes.extend_from_slice(&phase.sin().to_le_bytes());
        }
        let arg = corpus.capture("fl3-tone.cf32", &bytes);
        let setup = build_source(&cli(&[
            "--source", &arg, "--rate", "2048000", "--center", "100e6", "--loop",
        ]))
        .unwrap();
        assert!(
            setup.paced,
            "the stand-in must be time-paced like a radio (D-013) or it proves nothing"
        );

        // Deterministic, machine-independent contention: oversubscribe every
        // visible core 4x with busy-spin threads for the run's duration — the
        // in-process analogue of `cargo test`'s own thread pool running the
        // CPU-heavy `window` render tests beside this one (FL-3's hypothesis).
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let stop = Arc::new(AtomicBool::new(false));
        let hogs: Vec<_> = (0..cores * 4)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        for i in 0..10_000u64 {
                            std::hint::black_box(i);
                        }
                    }
                })
            })
            .collect();

        let source = crate::pipeline::open_source(&setup.desc).expect("open the paced stand-in");
        let pipeline = crate::pipeline::Pipeline::start(
            source,
            crate::pipeline::PipelineConfig {
                fft_size: 1024,
                window: WindowKind::Hann,
                paced: setup.paced,
                backpressure: false,
                db_bottom: -120.0,
                db_top: 0.0,
            },
        )
        .expect("start the pipeline over the paced stand-in");

        let deadline = Instant::now() + Duration::from_secs(20);
        let snap = loop {
            let snap = pipeline.health();
            if snap.batches_completed >= 500 {
                break snap;
            }
            assert!(Instant::now() < deadline, "no spectra after 20 s");
            std::thread::sleep(Duration::from_millis(10));
        };

        stop.store(true, Ordering::Relaxed);
        for h in hogs {
            let _ = h.join();
        }

        assert!(
            snap.processed_pct < 99.0,
            "expected forced CPU oversubscription to reproduce shedding, honestly \
             reported (D-013); got {snap:?} instead — either the mechanism no \
             longer holds or this machine's scheduler did not starve the \
             pipeline threads this run"
        );
    }

    /// FL-3/D-103's negative control: the fix is a pure CI-scheduling change
    /// (run `soapy_source_runs_end_to_end_through_the_production_path` in
    /// its own invocation instead of sharing a thread pool with the
    /// CPU-heavy `window` tests), which has no product code to revert and
    /// so no behavioural regression could ever fail deterministically on an
    /// unloaded machine. This structural check stands in for that: it reads
    /// the CI workflow itself and asserts the separation is really there,
    /// on every leg that compiles the `soapy` feature. Deleting the `--skip`
    /// clause, or the hardware test's own separate invocation, on any such
    /// leg fails this test — on any machine, unloaded, since it never runs
    /// the hardware path itself.
    #[test]
    fn ci_soapy_hardware_test_runs_in_its_own_invocation_on_every_soapy_leg() {
        const CI_YAML: &str = include_str!("../../../.github/workflows/ci.yml");
        const HW_TEST: &str = "soapy_source_runs_end_to_end_through_the_production_path";
        const SOAPY_FEATURE: &str = "--features phosphene-sources/soapy";
        let skip_clause = format!("--skip {HW_TEST}");

        // Scope to the `test:` job only (the `lint`/`bench`/`artifact-size`
        // jobs have their own, unrelated `matrix.name` entries that would
        // otherwise collide with leg names below).
        let job_start = CI_YAML
            .find("\n  test:\n")
            .expect("ci.yml has no `test:` job — has it been restructured?");
        let job_end = CI_YAML[job_start..]
            .find("\n  bench:\n")
            .map(|off| job_start + off)
            .expect("ci.yml has no `bench:` job after `test:` — has it been restructured?");
        let test_job = &CI_YAML[job_start..job_end];

        // Every leg the `test:` job's matrix names, in the order they
        // appear (D-052/D-101 doc comments describe each one).
        let legs = ["linux-x11", "linux-wayland", "macos-arm64"];
        let mut starts: Vec<(&str, usize)> = legs
            .iter()
            .map(|&leg| {
                let marker = format!("name: {leg}\n");
                let at = test_job.find(&marker).unwrap_or_else(|| {
                    panic!("leg {leg:?} not found in ci.yml's `test:` job matrix")
                });
                (leg, at)
            })
            .collect();
        starts.sort_by_key(|&(_, at)| at);

        let mut soapy_legs_checked = 0;
        for (i, &(leg, start)) in starts.iter().enumerate() {
            let end = starts
                .get(i + 1)
                .map(|&(_, at)| at)
                .unwrap_or(test_job.len());
            let block = &test_job[start..end];
            if !block.contains(SOAPY_FEATURE) {
                // A leg with no soapy feature step is deliberately skipped
                // here (D-052).
                continue;
            }
            soapy_legs_checked += 1;
            assert!(
                block.contains(&skip_clause),
                "leg {leg:?} compiles the soapy feature but its bundled `cargo test` \
                 run does not `--skip` the hardware test (FL-3/D-103)"
            );
            let occurrences = block.matches(HW_TEST).count();
            assert_eq!(
                occurrences, 2,
                "leg {leg:?}: expected the hardware test named exactly twice in its \
                 soapy-feature commands (once skipped in the bundled run, once alone \
                 in its own invocation) — found {occurrences} (FL-3/D-103)"
            );
        }
        assert!(
            soapy_legs_checked > 0,
            "no leg in ci.yml's `test:` job compiles the soapy feature at all — \
             this check would pass vacuously, which means D-052's CI coverage \
             regressed"
        );
    }
}
