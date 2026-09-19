// SPDX-License-Identifier: MIT

//! The file / stdin / FIFO source (FR-S1, lane M1-E) — replay of raw IQ
//! captures in any of the four [`IqFormat`]s.
//!
//! ## Behaviour contract
//!
//! * **Normalisation** happens at ingest via [`crate::format`]: whatever the
//!   native format, the sink sees full-scale-normalised `cf32`, so the D-006
//!   dBFS convention holds — a full-scale tone in any format reads 0.0 dBFS
//!   through `phosphene-core`.
//! * **Throttled real-time replay is the default** when a sample rate is
//!   known, and it paces **within** the data, not around read sizes: samples
//!   go out in blocks bounded by [`PACE_BLOCK_MAX_S`] of stream time, each
//!   pushed only once the stream time of its last sample has elapsed — the
//!   file arrives the way a live radio would deliver it, never running ahead
//!   of the rate. **Clarification C5:** absent a rate, replay is
//!   **unthrottled** and [`SourceMeta::sample_rate_hz`] is honestly `None`
//!   (the HUD then labels the axis in normalised frequency); a rate is
//!   **never invented**. Absent a center, `center_freq_hz` is `None` (FR-C3:
//!   relative-Hz labels). Fast-forward is a pace [`ReplayPace::Factor`] > 1;
//!   [`ReplayPace::Unthrottled`] ignores the rate for pacing while still
//!   reporting it in the metadata.
//! * **Truncation is normal, not a panic.** A partial sample at EOF (or
//!   anywhere a read boundary falls mid-sample) is carried across reads;
//!   whatever partial sample remains at EOF is discarded and counted in
//!   [`FileSourceStats::trailing_bytes`] so the app can report it honestly.
//! * **`--loop`** ([`FileSourceConfig::loop_replay`]) rewinds a regular file
//!   at EOF and keeps streaming — seamlessly: the decoded stream continues
//!   without a gap, and pacing carries across the wrap. Only regular files can
//!   loop (stdin, FIFOs and arbitrary readers cannot rewind; rejected at
//!   construction). A loop pass that yields no complete samples ends the
//!   stream instead of spinning.
//! * **Stop/resume resumes PACING, not just delivery**: [`SinkFlow::Stop`]
//!   returns from `stream`; streaming again resumes exactly where the source
//!   left off, carried partial sample included, and the resumed session is
//!   itself paced from its own start — a consumer that pauses gets no burst
//!   of unpaced samples on resume, and a paused stream does not sprint to
//!   catch up on wall-clock time spent stopped.
//!
//! Licensing pattern (LC-2, spec §4.5): pure Rust standard library, no
//! external code at all — trivially Pattern A, like the signal generator.

use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use phosphene_core::Complex;

use crate::format::{decode_append, IqFormat};
use crate::source::{
    Control, ControlCaps, SampleSink, SampleSource, SinkFlow, SourceDesc, SourceError, SourceMeta,
};

/// Where the raw IQ bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// The process's standard input (which may itself be a pipe or FIFO).
    Stdin,
    /// A filesystem path — a regular file or a FIFO.
    Path(PathBuf),
}

/// Replay pacing (FR-S1: throttled real-time replay by default, fast-forward
/// available).
///
/// All pacing needs a known sample rate to pace against; per clarification C5,
/// any pace combined with an absent [`FileSourceConfig::sample_rate_hz`]
/// replays unthrottled — a rate is never invented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplayPace {
    /// Real-time replay — the default.
    RealTime,
    /// Replay at `factor` × real time (must be finite and positive; > 1 is
    /// fast-forward, < 1 is slow motion).
    Factor(f64),
    /// As fast as the consumer accepts.
    Unthrottled,
}

/// Configuration for a [`FileSource`]. The CLI layer parses `--sdr file:…`,
/// `--format`, `--rate`, `--center`, `--loop` into this; the backend never
/// parses command lines.
#[derive(Debug, Clone, PartialEq)]
pub struct FileSourceConfig {
    /// Where the bytes come from.
    pub input: Input,
    /// The native sample format of the bytes.
    pub format: IqFormat,
    /// Sample rate in Hz, if the user supplied one. `None` replays
    /// unthrottled with a normalised-frequency axis (clarification C5).
    pub sample_rate_hz: Option<f64>,
    /// Center frequency in Hz, if the user supplied one. `None` labels the
    /// axis in relative Hz (FR-C3).
    pub center_freq_hz: Option<f64>,
    /// Replay pacing. Only paces when `sample_rate_hz` is known.
    pub pace: ReplayPace,
    /// Rewind at EOF and keep streaming (`--loop`). Regular files only.
    pub loop_replay: bool,
}

impl FileSourceConfig {
    /// A default-shaped config: real-time pace, no loop, no rate or center
    /// (i.e. unthrottled with normalised axes until the user says otherwise).
    pub fn new(input: Input, format: IqFormat) -> Self {
        FileSourceConfig {
            input,
            format,
            sample_rate_hz: None,
            center_freq_hz: None,
            pace: ReplayPace::RealTime,
            loop_replay: false,
        }
    }
}

/// Honest accounting of what a [`FileSource`] has done so far — the app
/// surfaces these instead of guessing (lane M1-E: truncation is reported, not
/// panicked over).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileSourceStats {
    /// Complete samples delivered to sinks, across all streaming sessions.
    pub samples_delivered: u64,
    /// Bytes of partial trailing samples discarded at EOF, cumulative across
    /// loop passes. Non-zero means the input was truncated mid-sample.
    pub trailing_bytes: u64,
    /// Completed rewinds under [`FileSourceConfig::loop_replay`].
    pub loops_completed: u64,
}

enum Reader {
    File(fs::File),
    Stdin(std::io::Stdin),
    Boxed(Box<dyn Read + Send>),
}

impl Reader {
    fn kind(&self) -> &'static str {
        match self {
            Reader::File(_) => "file",
            Reader::Stdin(_) => "stdin",
            Reader::Boxed(_) => "reader",
        }
    }
}

/// The file / stdin source. See the [module docs](self) for the behaviour
/// contract.
pub struct FileSource {
    config: FileSourceConfig,
    reader: Reader,
    /// Undecoded bytes carried between reads and across stream sessions —
    /// always shorter than one sample.
    leftover: Vec<u8>,
    /// Complete samples delivered since the last rewind (guards `--loop`
    /// against spinning on an input too short for a single sample).
    samples_since_rewind: u64,
    stats: FileSourceStats,
}

impl fmt::Debug for FileSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSource")
            .field("config", &self.config)
            .field("reader", &self.reader.kind())
            .field("stats", &self.stats)
            .finish()
    }
}

/// Bytes per read from the underlying input.
const READ_CHUNK_BYTES: usize = 64 * 1024;

/// The stated burstiness bound of throttled replay: when pacing, samples are
/// delivered in blocks of at most this much stream time (never fewer than one
/// sample), each pushed only once the stream time of its **last** sample has
/// elapsed. Within any streaming session, samples delivered by session time
/// `t` therefore never exceed `rate × t`; delivery may lag the schedule (sink
/// latency, OS sleep overshoot) but never leads it.
pub const PACE_BLOCK_MAX_S: f64 = 0.010;

fn validate(config: &FileSourceConfig) -> Result<(), SourceError> {
    if let Some(rate) = config.sample_rate_hz {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(SourceError::InvalidConfig(format!(
                "sample rate must be finite and positive, got {rate} Hz"
            )));
        }
    }
    if let Some(center) = config.center_freq_hz {
        if !center.is_finite() {
            return Err(SourceError::InvalidConfig(format!(
                "center frequency must be finite, got {center} Hz"
            )));
        }
    }
    if let ReplayPace::Factor(f) = config.pace {
        if !f.is_finite() || f <= 0.0 {
            return Err(SourceError::InvalidConfig(format!(
                "replay pace factor must be finite and positive, got {f}"
            )));
        }
    }
    // The pacing arithmetic must be TOTAL: individually valid factors can
    // still multiply to an unusable effective rate (overflow to infinity,
    // underflow to zero, or a rate so slow that a single sample's schedule
    // slot is not a representable `Duration`). Reject the product, not just
    // the factors, and name the offending values.
    let throttled = match (config.sample_rate_hz, config.pace) {
        (Some(rate), ReplayPace::RealTime) => Some((rate, 1.0)),
        (Some(rate), ReplayPace::Factor(f)) => Some((rate, f)),
        (Some(_), ReplayPace::Unthrottled) | (None, _) => None,
    };
    if let Some((rate, factor)) = throttled {
        let sps = rate * factor;
        if !sps.is_finite() || sps <= 0.0 || Duration::try_from_secs_f64(1.0 / sps).is_err() {
            return Err(SourceError::InvalidConfig(format!(
                "sample rate {rate:e} Hz × pace factor {factor} gives an effective replay \
                 rate of {sps:e} samples/s, which cannot be paced; the product must be \
                 finite, positive, and slow enough per sample to schedule"
            )));
        }
    }
    Ok(())
}

impl FileSource {
    /// Open the configured input.
    ///
    /// Fails with [`SourceError::InvalidConfig`] on a bad rate/center/pace or
    /// on `loop_replay` over anything but a regular file, and with
    /// [`SourceError::Io`] if the path cannot be opened.
    pub fn new(config: FileSourceConfig) -> Result<Self, SourceError> {
        validate(&config)?;
        let reader = match &config.input {
            Input::Stdin => {
                if config.loop_replay {
                    return Err(SourceError::InvalidConfig(
                        "loop replay needs a rewindable regular file; stdin cannot rewind"
                            .to_string(),
                    ));
                }
                Reader::Stdin(std::io::stdin())
            }
            Input::Path(path) => {
                if config.loop_replay {
                    // fs::metadata (unlike opening a FIFO) never blocks.
                    let meta = fs::metadata(path).map_err(|e| {
                        SourceError::Io(format!("cannot stat {}: {e}", path.display()))
                    })?;
                    if !meta.is_file() {
                        return Err(SourceError::InvalidConfig(format!(
                            "loop replay needs a rewindable regular file, and {} is not one \
                             (FIFO or special file?)",
                            path.display()
                        )));
                    }
                }
                let file = fs::File::open(path)
                    .map_err(|e| SourceError::Io(format!("cannot open {}: {e}", path.display())))?;
                Reader::File(file)
            }
        };
        Ok(FileSource {
            config,
            reader,
            leftover: Vec::new(),
            samples_since_rewind: 0,
            stats: FileSourceStats::default(),
        })
    }

    /// Build a source over an arbitrary reader (a pipe from a subprocess, an
    /// in-memory buffer in tests). `config.input` is used only for labelling;
    /// the bytes come from `reader`. Arbitrary readers cannot rewind, so
    /// `loop_replay` is rejected.
    pub fn from_reader(
        reader: Box<dyn Read + Send>,
        config: FileSourceConfig,
    ) -> Result<Self, SourceError> {
        validate(&config)?;
        if config.loop_replay {
            return Err(SourceError::InvalidConfig(
                "loop replay needs a rewindable regular file; an arbitrary reader cannot rewind"
                    .to_string(),
            ));
        }
        Ok(FileSource {
            config,
            reader: Reader::Boxed(reader),
            leftover: Vec::new(),
            samples_since_rewind: 0,
            stats: FileSourceStats::default(),
        })
    }

    /// The configuration this source was opened with.
    pub fn config(&self) -> &FileSourceConfig {
        &self.config
    }

    /// What has happened so far — delivery, truncation, loop accounting.
    pub fn stats(&self) -> FileSourceStats {
        self.stats
    }

    /// What the source is called in errors, e.g. `file capture.cs8` / `stdin`.
    fn describe(&self) -> String {
        match &self.config.input {
            Input::Stdin => "stdin".to_string(),
            Input::Path(path) => format!("file {}", path.display()),
        }
    }

    fn rewind(&mut self) -> Result<(), SourceError> {
        match &mut self.reader {
            Reader::File(f) => f
                .seek(SeekFrom::Start(0))
                .map(|_| ())
                .map_err(|e| SourceError::Io(format!("cannot rewind {}: {e}", self.describe()))),
            // Unreachable: loop_replay is only accepted over regular files.
            _ => Err(SourceError::InvalidConfig(format!(
                "{} cannot rewind",
                self.describe()
            ))),
        }
    }

    /// Effective pacing in samples per second, or `None` for unthrottled.
    /// Clarification C5: no rate → no throttle, never an invented rate.
    fn pace_sps(&self) -> Option<f64> {
        match (self.config.sample_rate_hz, self.config.pace) {
            (Some(rate), ReplayPace::RealTime) => Some(rate),
            (Some(rate), ReplayPace::Factor(f)) => Some(rate * f),
            (Some(_), ReplayPace::Unthrottled) | (None, _) => None,
        }
    }
}

impl SampleSource for FileSource {
    fn open(desc: &SourceDesc) -> Result<Self, SourceError> {
        match desc {
            SourceDesc::File(config) => FileSource::new(config.clone()),
            #[allow(unreachable_patterns)]
            _ => Err(SourceError::WrongBackend { backend: "file" }),
        }
    }

    fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
        sink.meta_changed(&self.meta());

        // When throttled, delivery is sliced into blocks of at most
        // [`PACE_BLOCK_MAX_S`] of stream time (never fewer than one sample),
        // and each block waits for its schedule slot BEFORE it is pushed — so
        // pacing happens within the data, not around whatever the read size
        // happened to be, and a sink that stops mid-replay never causes a
        // pacing debt: the next session paces from its own start.
        // Saturating throughout: a huge-but-finite rate casts to usize::MAX,
        // and the block size need never exceed one read anyway.
        let bps = self.config.format.bytes_per_sample();
        let pace = self.pace_sps().map(|sps| {
            let block_samples = ((sps * PACE_BLOCK_MAX_S) as usize).max(1);
            (sps, block_samples.saturating_mul(bps))
        });
        let pace_start = Instant::now();
        let mut paced_samples: u64 = 0;

        let mut bytes = std::mem::take(&mut self.leftover);
        let mut chunk = vec![0u8; READ_CHUNK_BYTES];
        let mut samples: Vec<Complex<f32>> = Vec::new();

        loop {
            // Deliver every whole sample currently buffered — one bounded,
            // schedule-slotted block at a time when throttled, all at once
            // when not.
            loop {
                let cap = match pace {
                    Some((_, block_bytes)) => bytes.len().min(block_bytes),
                    None => bytes.len(),
                };
                samples.clear();
                let consumed = decode_append(self.config.format, &bytes[..cap], &mut samples);
                if samples.is_empty() {
                    break;
                }
                bytes.drain(..consumed);
                let delivered = samples.len() as u64;
                if let Some((sps, _)) = pace {
                    // Live-radio schedule: a block is due once the stream
                    // time of its LAST sample has elapsed, so within a
                    // session delivery never runs ahead of the rate —
                    // samples delivered by session time t ≤ rate·t.
                    paced_samples += delivered;
                    // Validation guarantees a representable per-sample slot;
                    // saturating to Duration::MAX keeps even the cumulative
                    // arithmetic total (an unreachable ~10^19 s of stream
                    // time) rather than panicking.
                    let due = Duration::try_from_secs_f64(paced_samples as f64 / sps)
                        .unwrap_or(Duration::MAX);
                    let elapsed = pace_start.elapsed();
                    if due > elapsed {
                        std::thread::sleep(due - elapsed);
                    }
                }
                self.stats.samples_delivered += delivered;
                self.samples_since_rewind += delivered;
                if sink.push(&samples) == SinkFlow::Stop {
                    self.leftover = bytes;
                    return Ok(());
                }
            }

            let n = loop {
                match self.reader_read(&mut chunk) {
                    Ok(n) => break n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        self.leftover = bytes;
                        return Err(SourceError::Io(format!(
                            "read error on {}: {e}",
                            self.describe()
                        )));
                    }
                }
            };

            if n == 0 {
                // EOF. A partial trailing sample is a normal condition: count
                // it honestly and never concatenate it across a rewind.
                self.stats.trailing_bytes += bytes.len() as u64;
                bytes.clear();
                if self.config.loop_replay && self.samples_since_rewind > 0 {
                    self.rewind()?;
                    self.stats.loops_completed += 1;
                    self.samples_since_rewind = 0;
                    continue;
                }
                return Ok(());
            }

            bytes.extend_from_slice(&chunk[..n]);
        }
    }

    fn caps(&self) -> ControlCaps {
        ControlCaps::NONE
    }

    fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
        Err(SourceError::UnsupportedControl {
            backend: "file",
            control: ctl.name(),
        })
    }

    fn meta(&self) -> SourceMeta {
        let (label, provenance) = match &self.config.input {
            Input::Stdin => (
                "stdin".to_string(),
                format!("stdin ({})", self.config.format),
            ),
            Input::Path(path) => (
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string()),
                format!("file {} ({})", path.display(), self.config.format),
            ),
        };
        SourceMeta {
            sample_rate_hz: self.config.sample_rate_hz,
            center_freq_hz: self.config.center_freq_hz,
            label,
            provenance,
        }
    }
}

impl FileSource {
    fn reader_read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match &mut self.reader {
            Reader::File(f) => f.read(buf),
            Reader::Stdin(s) => s.read(buf),
            Reader::Boxed(r) => r.read(buf),
        }
    }
}
