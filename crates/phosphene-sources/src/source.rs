// SPDX-License-Identifier: MIT

//! The [`SampleSource`] seam (product spec §8.2) and its supporting types.
//!
//! ## Design constraints this shape answers
//!
//! * **Out-of-process first (D-015).** The highest-priority real backends
//!   reach GPL device libraries via SoapySDR runtime modules (Pattern C) or
//!   subprocess capture (Pattern B) — never in-process linking of the device
//!   library. The trait therefore assumes nothing about callbacks or library
//!   ownership: a source is simply something that blocks and delivers `cf32`
//!   into a [`SampleSink`] until told to stop.
//! * **Sweep-ready (spec §5.3).** A future retune-and-stitch sweep must not
//!   require re-architecting this seam. Two provisions cover it: metadata is
//!   delivered *in-band* via [`SampleSink::meta_changed`], so a stream is a
//!   sequence of segments each under known (center, rate); and streaming is
//!   restartable — [`SinkFlow::Stop`] returns control to the caller, which may
//!   [`SampleSource::set`] a new center frequency and stream again. Sweep is
//!   then either backend-internal retuning (segments announced via
//!   `meta_changed`) or an external retune loop; neither needs a new trait.
//! * **Ingest normalisation.** Backends convert their native sample format
//!   (`cu8`, `cs8`, `cs16`, …) to full-scale-normalised `cf32` **at ingest**
//!   (spec §8.2), so downstream code sees one format and the D-006 dBFS
//!   convention holds regardless of converter width. [`SourceMeta`] records
//!   the provenance so the UI can say what the native format was.
//!
//! ## Threading contract, and live control (D-056/D-058)
//!
//! [`SampleSource::stream`] is a blocking call intended to own a dedicated
//! source thread (spec §8.3); the sink is the app's ring producer. Because
//! `stream` and [`SampleSource::set`] both take `&mut self`, live device
//! control during streaming cannot go through `set` — the source is inside
//! `stream`, holding itself mutably. `set` therefore configures a source
//! **between** streaming sessions, and live control travels the mechanism
//! this module's doc comment used to defer to "the first controllable
//! backend (M3)": a [`control_channel`].
//!
//! [`SampleSource::control_handle`] hands out a [`ControlHandle`] before
//! streaming starts and keeps the [`ControlInbox`]; the backend drains that
//! inbox **inside its own stream loop**, where it does hold `&mut self`, and
//! answers each [`ControlRequest`] with the metadata it **read back from the
//! device** (clarification C5) or with the device's own error. So there is
//! exactly one place a control is applied — the same code `set` runs — and
//! exactly one truth about where the radio is (**D-058**: two entry points,
//! one mechanism; the UI shows where the radio *is*, never where it was
//! asked to go).
//!
//! Sources that cannot be controlled live implement nothing: the default
//! [`SampleSource::control_handle`] returns `None`, and their `caps()`
//! already reports `tune: false`, so the app refuses the control before any
//! channel is involved.

use std::fmt;
use std::sync::mpsc;
use std::time::Duration;

use phosphene_core::Complex;

#[cfg(feature = "file")]
use crate::file::FileSourceConfig;
#[cfg(feature = "siggen")]
use crate::siggen::SigGenConfig;
use crate::soapy::SoapySourceConfig;

/// Metadata describing what a source is delivering (spec §8.2: rate, center,
/// label, format provenance).
///
/// `None` fields are *honestly absent*, never invented (clarification C5): a
/// missing rate means the display labels frequency in normalised units and
/// replay is unthrottled; a missing center means the axis is labelled in
/// relative Hz (FR-C3).
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMeta {
    /// Sample rate in Hz, if known.
    pub sample_rate_hz: Option<f64>,
    /// Center (tuned) frequency in Hz, if known.
    pub center_freq_hz: Option<f64>,
    /// Short human-readable name for the HUD / source picker.
    pub label: String,
    /// Where the samples came from and their native format before ingest
    /// conversion to `cf32` (e.g. `"file capture.cs8 (cs8)"`).
    pub provenance: String,
}

/// The centre-frequency range a device reports it can reach, Hz — its own
/// answer, never a guess (**D-058**: a slider needs bounds, and inventing
/// them would promise frequencies the radio cannot reach).
///
/// A device that advertises several disjoint ranges is summarised by their
/// hull: `min_hz` is the lowest bound reported and `max_hz` the highest.
/// The hull can therefore span a gap the hardware does not cover — which is
/// why it bounds the *interaction* only and never the truth: the device
/// remains the arbiter, a retune it refuses surfaces as a real error, and
/// the display follows the readback (C5), never the request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneRange {
    /// Lowest centre frequency the device reported, Hz.
    pub min_hz: f64,
    /// Highest centre frequency the device reported, Hz.
    pub max_hz: f64,
}

impl TuneRange {
    /// The hull of the ranges a device reported, or `None` when it reported
    /// none or reported unusable numbers — "no range" is stated honestly and
    /// never replaced with a plausible default.
    pub fn hull(ranges: impl IntoIterator<Item = (f64, f64)>) -> Option<TuneRange> {
        let mut hull: Option<TuneRange> = None;
        for (min_hz, max_hz) in ranges {
            if !min_hz.is_finite() || !max_hz.is_finite() || max_hz <= min_hz || min_hz < 0.0 {
                continue;
            }
            hull = Some(match hull {
                None => TuneRange { min_hz, max_hz },
                Some(h) => TuneRange {
                    min_hz: h.min_hz.min(min_hz),
                    max_hz: h.max_hz.max(max_hz),
                },
            });
        }
        hull
    }

    /// Whether `hz` falls inside the reported hull.
    pub fn contains(&self, hz: f64) -> bool {
        hz >= self.min_hz && hz <= self.max_hz
    }
}

/// Which controls the active source supports (FR-S7): everything `false`
/// means a read-only source whose metadata is display-only.
///
/// `Eq` is deliberately absent: [`tune_range`](Self::tune_range) carries the
/// device's own floating-point bounds (D-058), and there is no sensible
/// total equality over those.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ControlCaps {
    /// Center frequency can be set.
    pub tune: bool,
    /// Sample rate can be set.
    pub rate: bool,
    /// Gain can be set.
    pub gain: bool,
    /// Antenna port can be selected.
    pub antenna: bool,
    /// The centre-frequency range the device reports (D-058). `None` when
    /// the source cannot tune at all, or when a tunable device reports no
    /// usable range — the UI then offers no bounded control rather than one
    /// bounded by a guess.
    pub tune_range: Option<TuneRange>,
}

impl ControlCaps {
    /// A source with no controls at all (pipes, files, the signal generator).
    pub const NONE: ControlCaps = ControlCaps {
        tune: false,
        rate: false,
        gain: false,
        antenna: false,
        tune_range: None,
    };

    /// True if any control is supported.
    pub fn any(&self) -> bool {
        self.tune || self.rate || self.gain || self.antenna
    }
}

/// A control command for a source that advertises the matching capability.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Control {
    /// Tune to a new center frequency (Hz).
    CenterFreqHz(f64),
    /// Change the sample rate (Hz).
    SampleRateHz(f64),
    /// Set overall gain (dB); per-stage gains arrive with the backends that
    /// have them (M3).
    GainDb(f64),
    /// Select an antenna port by name.
    Antenna(String),
}

impl Control {
    /// Stable name of the control, for error messages and logs.
    pub fn name(&self) -> &'static str {
        match self {
            Control::CenterFreqHz(_) => "center frequency",
            Control::SampleRateHz(_) => "sample rate",
            Control::GainDb(_) => "gain",
            Control::Antenna(_) => "antenna",
        }
    }
}

/// Which source to open, with its backend-specific configuration.
///
/// One variant per backend; each is added (behind its cargo feature) by the
/// lane that implements the backend. The CLI/app layer parses `--sdr …` into
/// this; backends never parse command lines.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SourceDesc {
    /// The built-in deterministic signal generator (demo mode + test fixture).
    #[cfg(feature = "siggen")]
    SigGen(SigGenConfig),
    /// Raw IQ replay from a file, FIFO, or stdin (FR-S1, lane M1-E).
    #[cfg(feature = "file")]
    File(FileSourceConfig),
    /// A SoapySDR device (FR-S5, lane S1-A) — spec §4.5 Pattern C. The
    /// descriptor is unconditional (its config is pure data) so every build's
    /// CLI can represent the request; the device backend itself is behind
    /// the `soapy` feature, off by default.
    Soapy(SoapySourceConfig),
}

/// Errors from opening, configuring, or controlling a source.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SourceError {
    /// The backend rejected its configuration; the message says exactly which
    /// field and why (spec §8.4: errors are specific and actionable).
    InvalidConfig(String),
    /// [`SampleSource::open`] was handed a [`SourceDesc`] for a different
    /// backend.
    WrongBackend {
        /// The backend that was asked to open the descriptor.
        backend: &'static str,
    },
    /// An I/O failure opening, reading, or rewinding the underlying input;
    /// the message names the path/stream and the operation that failed.
    Io(String),
    /// Device discovery found nothing to open; the message names exactly
    /// what was searched (lane S1-A: zero devices is a clear error, never a
    /// panic and never a silent hang).
    NoDevice(String),
    /// A live control could not be delivered to a streaming source, or the
    /// source did not answer within [`CONTROL_TIMEOUT`] (D-056/D-058). The
    /// message says which — never a silent no-op that would leave the
    /// display claiming a frequency the radio is not on.
    ControlUnavailable(String),
    /// The tuner was mutated and the result could **not** be read back
    /// (**D-060**). The device moved; where it moved to is unknown, so the
    /// centre frequency this program displays can no longer be substantiated
    /// and the stream ends rather than assert one. The message names both
    /// facts — the control was applied, the readback failed — because either
    /// alone reads as the wrong story.
    TuneUnverified(String),
    /// The source does not support the requested control — consult
    /// [`SampleSource::caps`] first.
    UnsupportedControl {
        /// The backend that rejected the control.
        backend: &'static str,
        /// [`Control::name`] of the rejected control.
        control: &'static str,
    },
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceError::InvalidConfig(msg) => write!(f, "invalid source configuration: {msg}"),
            SourceError::Io(msg) => write!(f, "source I/O error: {msg}"),
            SourceError::NoDevice(msg) => write!(f, "{msg}"),
            SourceError::ControlUnavailable(msg) => write!(f, "{msg}"),
            SourceError::TuneUnverified(msg) => write!(f, "{msg}"),
            SourceError::WrongBackend { backend } => {
                write!(f, "source descriptor is not for the {backend} backend")
            }
            SourceError::UnsupportedControl { backend, control } => {
                write!(f, "the {backend} source does not support setting {control}")
            }
        }
    }
}

impl std::error::Error for SourceError {}

/// Flow-control verdict returned by a sink for each delivered block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkFlow {
    /// Keep streaming.
    Continue,
    /// Stop streaming; [`SampleSource::stream`] returns `Ok(())`.
    Stop,
}

/// Where a streaming source delivers its samples — in the running app this is
/// the SPSC ring producer feeding the compute pool (spec §8.3); in tests it is
/// whatever the test wants to capture.
pub trait SampleSink {
    /// Deliver one block of consecutive `cf32` samples under the most recently
    /// announced [`SourceMeta`]. Block sizes are backend-chosen and carry no
    /// meaning; only the concatenated stream does.
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow;

    /// The stream's metadata changed (retune, rate change). Called before the
    /// first `push` of a streaming session and again whenever meta changes
    /// mid-stream — the in-band segment boundary a future sweep mode stitches
    /// on (spec §5.3).
    fn meta_changed(&mut self, meta: &SourceMeta) {
        let _ = meta;
    }

    /// The device reported that `samples` were lost upstream, before delivery
    /// (hardware overflow). Per **D-048** these count as **received and
    /// dropped**: the app's accounting sink folds them into the D-013
    /// figures so `PROC %` falls and the drop counter rises exactly as for a
    /// locally shed batch — a sample lost in the radio and a sample shed in
    /// our ring answer the same user question ("is this display showing me
    /// everything?").
    ///
    /// **Gated on evidence (D-050):** a source may call this only with a
    /// count the device *trustworthily reported* — never one inferred from
    /// an unvalidated side channel. No backend can currently produce such a
    /// count (SoapySDR's read path surfaces no timestamp-validity flag), so
    /// today's overflow reporting goes through [`device_overflow`] instead;
    /// this path stands ready for a binding that can supply real numbers.
    /// Default: ignore (test sinks that do not track accounting).
    ///
    /// [`device_overflow`]: SampleSink::device_overflow
    fn device_lost(&mut self, samples: u64) {
        let _ = samples;
    }

    /// The device reported one overflow **event**: it lost an *unknown*
    /// amount of data upstream (D-050). Events are counted and surfaced —
    /// "the device lost data, amount unknown" is information — but they are
    /// **never** converted into a sample or batch quantity: `PROC %` and
    /// `DROPS` keep describing only what the pipeline actually received and
    /// shed. Default: ignore.
    fn device_overflow(&mut self) {}

    /// The tuner was mutated and the result could **not** be read back
    /// (**D-060**) — the hardware has moved somewhere this program cannot
    /// name.
    ///
    /// This is *not* [`meta_changed`]: there is no new metadata to announce,
    /// which is exactly the trap. A sink that keys its reset off "did the
    /// centre change?" is silently told "no" when the thing that failed was
    /// the very act of finding out, and it then goes on drawing old-centre
    /// history under an axis the radio has left.
    ///
    /// So this call says the one thing that *is* known: **every
    /// frequency-dependent accumulator is stale from this instant** —
    /// persistence, max hold, waterfall — and the displayed centre is no
    /// longer a claim this program is entitled to make. An accounting sink
    /// wipes them and drops the centre from its published metadata; the
    /// stream then ends with [`SourceError::TuneUnverified`], because a
    /// spectrum display whose frequency axis cannot be substantiated is not a
    /// degraded instrument, it is a misleading one.
    ///
    /// Default: ignore (test sinks that hold no accumulators).
    ///
    /// [`meta_changed`]: SampleSink::meta_changed
    fn tune_unverified(&mut self) {}
}

/// How long a caller waits for a streaming source to answer a live control
/// before giving up (D-056/D-058).
///
/// The backend answers from inside its stream loop, so the worst case is one
/// blocking device read plus the device's own tune time — a Soapy read
/// timeout is 200 ms and a `set_frequency` on a USB radio is tens of
/// milliseconds. Two seconds is roughly an order of magnitude of headroom
/// over that, and giving up is itself an honest outcome: a caller that
/// waited forever on a wedged driver would freeze the display, and one that
/// assumed success would assert a frequency the radio never reached.
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);

/// What a source answers a live control with: the metadata it **read back
/// from the device** after applying the control (clarification C5 — the
/// device's own answer, never an echo of the request), or the error the
/// device gave.
pub type ControlReply = Result<SourceMeta, SourceError>;

/// One live-control request waiting in a backend's [`ControlInbox`].
///
/// The backend applies the control and **must** answer, with
/// [`ControlRequest::answer`]; dropping a request unanswered fails the
/// caller's wait with a "did not answer" error rather than hanging it.
#[derive(Debug)]
pub struct ControlRequest {
    control: Control,
    reply: mpsc::Sender<ControlReply>,
}

impl ControlRequest {
    /// The control to apply.
    pub fn control(&self) -> &Control {
        &self.control
    }

    /// Answer the waiting caller with the device readback, or the device's
    /// error. A caller that has already given up is not an error here — the
    /// send is allowed to fail silently.
    pub fn answer(self, reply: ControlReply) {
        let _ = self.reply.send(reply);
    }
}

/// The backend's half of a [`control_channel`]: drained inside the stream
/// loop, where the backend holds `&mut self` and can actually touch the
/// device.
#[derive(Debug)]
pub struct ControlInbox {
    rx: mpsc::Receiver<ControlRequest>,
}

impl ControlInbox {
    /// The next pending request, or `None` when the inbox is empty (or every
    /// handle has been dropped). Never blocks — the stream loop must keep
    /// servicing the device.
    pub fn try_recv(&self) -> Option<ControlRequest> {
        self.rx.try_recv().ok()
    }
}

/// The caller's half of a [`control_channel`]: `Send + Sync + Clone`, so the
/// app can hold it on the display thread while the source streams on its own
/// (spec §8.3: "device control commands flow back over a channel").
#[derive(Debug, Clone)]
pub struct ControlHandle {
    tx: mpsc::Sender<ControlRequest>,
}

impl ControlHandle {
    /// Apply `control` to the streaming source and wait for its answer.
    ///
    /// Returns the source's metadata **as read back from the device** — the
    /// value the display must follow (C5/D-058) — or a real error. Blocks
    /// for at most [`CONTROL_TIMEOUT`]; a keypress-rate call, never on a
    /// per-frame path.
    pub fn set(&self, control: Control) -> ControlReply {
        let name = control.name();
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(ControlRequest { control, reply: tx })
            .map_err(|_| {
                SourceError::ControlUnavailable(format!(
                    "cannot set the {name}: the source is not streaming"
                ))
            })?;
        rx.recv_timeout(CONTROL_TIMEOUT).map_err(|_| {
            SourceError::ControlUnavailable(format!(
                "the source did not answer the {name} control within {} s — \
                 the device may be wedged or unplugged",
                CONTROL_TIMEOUT.as_secs()
            ))
        })?
    }
}

/// Build a live-control channel: the [`ControlHandle`] the app keeps and the
/// [`ControlInbox`] the backend drains inside its stream loop.
pub fn control_channel() -> (ControlHandle, ControlInbox) {
    let (tx, rx) = mpsc::channel();
    (ControlHandle { tx }, ControlInbox { rx })
}

/// What applying one live control actually did to the device — the three
/// states **D-060** names, kept apart because the middle one used to be
/// indistinguishable from the first.
///
/// The distinction is not pedantry. A guard that asks *"did anything
/// change?"* answers "no" in both the first and the third case, and only one
/// of those is honest.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlOutcome {
    /// The device refused. Nothing outside this program moved, the readback
    /// still describes the hardware, and the display stays exactly where it
    /// was — an honest state.
    Rejected(SourceError),
    /// Applied, and confirmed against the device's own readback (C5) — an
    /// honest state.
    Applied,
    /// **Applied but unverifiable**: the mutation succeeded and the readback
    /// failed, so the hardware has moved somewhere this program cannot name.
    /// Fatal — see [`SourceError::TuneUnverified`].
    Unverified(SourceError),
}

/// How many times a readback is attempted before a control is declared
/// unverifiable: the first read plus **one** retry.
///
/// A readback is a cheap register read on a bus that has just been made to
/// retune, so a single transient failure is plausible enough to be worth
/// asking twice. Twice is where it stops: retrying is not a substitute for
/// the rule (D-060), it only keeps the rule from firing on noise.
const READBACK_ATTEMPTS: usize = 2;

/// Apply one control to a device and **verify it by reading the device
/// back** — the D-060 ordering, in one place, for every backend that mutates
/// hardware.
///
/// `mutate` performs the device-side change; `read_back` re-reads the
/// device's own answer into wherever the backend keeps its metadata. The
/// order matters and the failure path matters more:
///
/// * `mutate` fails → [`ControlOutcome::Rejected`]. The device never moved,
///   so the caller's error is the device's own and nothing needs resetting.
/// * `mutate` succeeds, `read_back` succeeds → [`ControlOutcome::Applied`].
/// * `mutate` succeeds, `read_back` fails (twice) →
///   [`ControlOutcome::Unverified`]. **The hardware has moved and this
///   program can no longer say where.** The old value is not a claim it is
///   entitled to keep making, so this is a hard failure carrying a message
///   that names both facts, never a quiet `Err` that leaves the caller's
///   metadata sitting at the pre-tune value.
///
/// `device` names the hardware for the message; `control` names what was
/// applied.
pub fn apply_and_verify(
    device: &str,
    control: &Control,
    mutate: impl FnOnce() -> Result<(), SourceError>,
    mut read_back: impl FnMut() -> Result<(), SourceError>,
) -> ControlOutcome {
    if let Err(e) = mutate() {
        return ControlOutcome::Rejected(e);
    }
    let mut failures: Vec<String> = Vec::new();
    for _ in 0..READBACK_ATTEMPTS {
        match read_back() {
            Ok(()) => return ControlOutcome::Applied,
            Err(e) => failures.push(e.to_string()),
        }
    }
    ControlOutcome::Unverified(SourceError::TuneUnverified(format!(
        "{device}: the {} was applied to the device but could not be read \
         back ({}) — the radio has moved and this program can no longer say \
         where, so the displayed center frequency cannot be substantiated. \
         Ending the stream rather than asserting one.",
        control.name(),
        failures.join("; then on retry: ")
    )))
}

/// Announce one [`ControlOutcome`] in band, answer the caller, and say
/// whether the stream may continue — the second half of the **D-060**
/// contract, shared by every backend that drains a [`ControlInbox`].
///
/// `before` and `after` are the source's metadata either side of the
/// attempt; a control that moved the device's own answer is announced with
/// [`SampleSink::meta_changed`] at exactly the sample it starts being true
/// of (spec §5.3's segment boundary), which is also where D-056's
/// accumulator reset belongs.
///
/// Returns `Err` only for [`ControlOutcome::Unverified`], and the caller
/// **must** end the stream with it. Before that error is returned the sink
/// has been told, via [`SampleSink::tune_unverified`], that every
/// frequency-dependent accumulator is stale — D-060 rule 1: their contents
/// went stale the moment the hardware moved, whatever the readback did.
pub fn settle_control(
    outcome: ControlOutcome,
    request: ControlRequest,
    sink: &mut dyn SampleSink,
    before: &SourceMeta,
    after: &SourceMeta,
) -> Result<(), SourceError> {
    match outcome {
        ControlOutcome::Unverified(err) => {
            sink.tune_unverified();
            request.answer(Err(err.clone()));
            Err(err)
        }
        ControlOutcome::Rejected(err) => {
            // A refusal can still have nudged the device (a clamp, a partial
            // apply); if the readback moved, the display follows it.
            if after != before {
                sink.meta_changed(after);
            }
            request.answer(Err(err));
            Ok(())
        }
        ControlOutcome::Applied => {
            if after != before {
                sink.meta_changed(after);
            }
            request.answer(Ok(after.clone()));
            Ok(())
        }
    }
}

/// A source of complex baseband samples — the seam every backend implements
/// (product spec §8.2).
pub trait SampleSource {
    /// Open the source described by `desc`.
    ///
    /// Returns [`SourceError::WrongBackend`] if `desc` is for a different
    /// backend, [`SourceError::InvalidConfig`] if the configuration is
    /// rejected.
    fn open(desc: &SourceDesc) -> Result<Self, SourceError>
    where
        Self: Sized;

    /// Stream samples into `sink` until it returns [`SinkFlow::Stop`] (then
    /// `Ok(())`), the source is exhausted (files — also `Ok(())`), or the
    /// source fails.
    ///
    /// Blocking; owns the calling thread for the duration (spec §8.3). Must
    /// announce the current metadata via [`SampleSink::meta_changed`] before
    /// the first [`SampleSink::push`]. Streaming again after a stop is
    /// permitted and resumes where the source left off (live sources have no
    /// rewind; a stopped-then-restarted stream is simply a new segment).
    fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError>;

    /// Which controls this source supports (FR-S7). Static per source.
    fn caps(&self) -> ControlCaps;

    /// Apply a control command between streaming sessions.
    ///
    /// Sources must reject controls they did not advertise in
    /// [`caps`](Self::caps) with [`SourceError::UnsupportedControl`].
    ///
    /// This is **not** the live path: a streaming source is inside
    /// [`stream`](Self::stream) and cannot be called here. Live control goes
    /// through [`control_handle`](Self::control_handle), which routes to the
    /// same application code (D-058: one mechanism).
    fn set(&mut self, ctl: Control) -> Result<(), SourceError>;

    /// Hand out a [`ControlHandle`] for controlling this source **while it
    /// streams** (D-056/D-058), keeping the [`ControlInbox`] to drain inside
    /// [`stream`](Self::stream).
    ///
    /// Called once, before streaming starts. The default is `None`: a source
    /// with no live control implements nothing, and its
    /// [`caps`](Self::caps) already says so.
    fn control_handle(&mut self) -> Option<ControlHandle> {
        None
    }

    /// Current metadata (rate, center, label, provenance).
    fn meta(&self) -> SourceMeta;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_caps_none_reports_no_capability() {
        assert!(!ControlCaps::NONE.any());
        assert!(ControlCaps {
            gain: true,
            ..ControlCaps::NONE
        }
        .any());
    }

    #[test]
    fn error_messages_name_the_problem() {
        let e = SourceError::InvalidConfig("sample rate must be positive".into());
        assert!(e.to_string().contains("sample rate"));
        let e = SourceError::UnsupportedControl {
            backend: "siggen",
            control: Control::GainDb(10.0).name(),
        };
        assert!(e.to_string().contains("siggen"));
        assert!(e.to_string().contains("gain"));
    }

    /// D-058: the range is the device's own answer, summarised without
    /// inventing anything. Nothing usable in means nothing out — "no range"
    /// is a fact the UI is entitled to hear.
    #[test]
    fn tune_range_hull_summarises_only_what_was_reported() {
        let one = TuneRange::hull([(70e6, 6e9)]).unwrap();
        assert_eq!((one.min_hz, one.max_hz), (70e6, 6e9));
        assert!(one.contains(751e6));
        assert!(!one.contains(10e6));

        // Several reported ranges collapse to their hull.
        let two = TuneRange::hull([(24e6, 1.75e9), (1.75e9, 6e9)]).unwrap();
        assert_eq!((two.min_hz, two.max_hz), (24e6, 6e9));

        // Unusable numbers are dropped, never repaired into a plausible one.
        assert_eq!(TuneRange::hull([]), None);
        assert_eq!(TuneRange::hull([(f64::NAN, 6e9)]), None);
        assert_eq!(TuneRange::hull([(6e9, 70e6)]), None);
        assert_eq!(TuneRange::hull([(0.0, 0.0)]), None);
    }

    /// A sink that records the two things D-060 cares about: whether it was
    /// told its accumulators are stale, and what metadata it was handed.
    #[derive(Default)]
    struct NotedSink {
        stale: usize,
        announced: Vec<SourceMeta>,
    }

    impl SampleSink for NotedSink {
        fn push(&mut self, _: &[Complex<f32>]) -> SinkFlow {
            SinkFlow::Continue
        }
        fn meta_changed(&mut self, meta: &SourceMeta) {
            self.announced.push(meta.clone());
        }
        fn tune_unverified(&mut self) {
            self.stale += 1;
        }
    }

    fn radio_meta(center_hz: Option<f64>) -> SourceMeta {
        SourceMeta {
            sample_rate_hz: Some(2.048e6),
            center_freq_hz: center_hz,
            label: "radio".into(),
            provenance: "radio (cf32)".into(),
        }
    }

    /// **D-060, the ordering itself.** The three states a live control can
    /// leave a radio in are three different answers, and the middle one is
    /// the whole point: a mutation that succeeded followed by a readback
    /// that failed is NOT "nothing happened".
    #[test]
    fn a_tune_that_cannot_be_read_back_is_its_own_outcome() {
        let tune = Control::CenterFreqHz(751e6);

        // 1. The device refused: nothing moved, and the readback is never
        //    even attempted — there is nothing new to read.
        let mut reads = 0;
        let refused = apply_and_verify(
            "radio",
            &tune,
            || Err(SourceError::InvalidConfig("out of range".into())),
            || {
                reads += 1;
                Ok(())
            },
        );
        assert!(matches!(refused, ControlOutcome::Rejected(_)));
        assert_eq!(reads, 0, "a refused control must not claim a readback");

        // 2. Applied and confirmed.
        assert_eq!(
            apply_and_verify("radio", &tune, || Ok(()), || Ok(())),
            ControlOutcome::Applied
        );

        // 3. Applied, and the readback failed. One retry (D-060 §3), then
        //    the hard failure — naming BOTH facts, because either alone
        //    reads as the wrong story.
        let mut attempts = 0;
        let unverified = apply_and_verify(
            "B200mini",
            &tune,
            || Ok(()),
            || {
                attempts += 1;
                Err(SourceError::Io("USB read failed".into()))
            },
        );
        assert_eq!(attempts, 2, "one readback retry, then the rule fires");
        let ControlOutcome::Unverified(SourceError::TuneUnverified(msg)) = unverified else {
            panic!("a readback failure after a successful mutation is not a rejection");
        };
        assert!(msg.contains("B200mini"), "{msg}");
        assert!(msg.contains("applied"), "{msg}");
        assert!(msg.contains("read"), "{msg}");
        assert!(msg.contains("USB read failed"), "{msg}");

        // A transient failure is not the rule firing: the retry is allowed
        // to succeed.
        let mut attempts = 0;
        let flaky = apply_and_verify(
            "radio",
            &tune,
            || Ok(()),
            || {
                attempts += 1;
                if attempts == 1 {
                    Err(SourceError::Io("transient".into()))
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(flaky, ControlOutcome::Applied);
    }

    /// **D-060's settlement.** The unverified case must tell the sink its
    /// accumulators are stale, answer the caller with the error, and hand
    /// the stream a reason to end — all three, because the bug was that a
    /// guard keyed off "did the metadata change?" fired none of them.
    #[test]
    fn settling_an_unverified_tune_stales_the_sink_and_ends_the_stream() {
        let before = radio_meta(Some(100e6));
        // The readback failed, so `after` is exactly `before`: this is the
        // "nothing changed" the old guard believed.
        let after = before.clone();
        let (handle, inbox) = control_channel();
        let caller = std::thread::spawn(move || handle.set(Control::CenterFreqHz(751e6)));
        let request = loop {
            if let Some(request) = inbox.try_recv() {
                break request;
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        let mut sink = NotedSink::default();
        let verdict = settle_control(
            ControlOutcome::Unverified(SourceError::TuneUnverified("applied, unread".into())),
            request,
            &mut sink,
            &before,
            &after,
        );

        assert_eq!(
            sink.stale, 1,
            "the accumulators must be told they are stale"
        );
        assert!(
            sink.announced.is_empty(),
            "there is no verified metadata to announce"
        );
        assert!(
            matches!(verdict, Err(SourceError::TuneUnverified(_))),
            "the stream must end: {verdict:?}"
        );
        assert!(matches!(
            caller.join().unwrap(),
            Err(SourceError::TuneUnverified(_))
        ));
    }

    /// The honest states settle honestly: a refusal leaves the accumulators
    /// alone (nothing moved), and a success announces the readback.
    #[test]
    fn settling_the_honest_outcomes_leaves_the_accumulators_alone() {
        for (outcome, want_announced) in [
            (
                ControlOutcome::Rejected(SourceError::InvalidConfig("no".into())),
                false,
            ),
            (ControlOutcome::Applied, true),
        ] {
            let before = radio_meta(Some(100e6));
            let after = radio_meta(Some(if want_announced { 751e6 } else { 100e6 }));
            let (handle, inbox) = control_channel();
            let caller = std::thread::spawn(move || handle.set(Control::CenterFreqHz(751e6)));
            let request = loop {
                if let Some(request) = inbox.try_recv() {
                    break request;
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            let mut sink = NotedSink::default();
            let verdict = settle_control(outcome, request, &mut sink, &before, &after);
            assert!(verdict.is_ok(), "only D-060's case ends the stream");
            assert_eq!(sink.stale, 0);
            assert_eq!(sink.announced.len(), usize::from(want_announced));
            assert_eq!(caller.join().unwrap().is_ok(), want_announced);
        }
    }

    /// The live-control seam (D-056/D-058): a request reaches the backend's
    /// inbox, the backend answers with the **device readback**, and the
    /// caller gets that back — not an echo of what it asked for.
    #[test]
    fn control_channel_carries_the_readback_back_to_the_caller() {
        let (handle, inbox) = control_channel();
        let backend = std::thread::spawn(move || {
            let request = loop {
                if let Some(request) = inbox.try_recv() {
                    break request;
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            assert_eq!(request.control(), &Control::CenterFreqHz(751_010_000.0));
            // A real tuner snaps to its own grid; the answer says where it
            // actually went (C5).
            request.answer(Ok(SourceMeta {
                sample_rate_hz: Some(2.048e6),
                center_freq_hz: Some(751_000_000.0),
                label: "fake".into(),
                provenance: "fake".into(),
            }));
        });
        let meta = handle.set(Control::CenterFreqHz(751_010_000.0)).unwrap();
        assert_eq!(meta.center_freq_hz, Some(751_000_000.0));
        backend.join().unwrap();
    }

    /// A refusal comes back as a refusal — never as a silent `Ok` that would
    /// leave a display asserting a frequency the radio is not on.
    #[test]
    fn control_channel_carries_a_refusal_and_a_dead_backend() {
        let (handle, inbox) = control_channel();
        let backend = std::thread::spawn(move || loop {
            if let Some(request) = inbox.try_recv() {
                request.answer(Err(SourceError::InvalidConfig(
                    "device rejected center frequency 1 Hz".into(),
                )));
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        });
        let err = handle.set(Control::CenterFreqHz(1.0)).unwrap_err();
        assert!(err.to_string().contains("rejected"), "unhelpful: {err}");
        backend.join().unwrap();

        // The backend is gone: the caller is told so rather than hanging or
        // being told it worked.
        let err = handle.set(Control::CenterFreqHz(751e6)).unwrap_err();
        assert!(
            err.to_string().contains("not streaming"),
            "unhelpful: {err}"
        );
    }

    /// The default `control_handle` is `None`: a source with no live control
    /// implements nothing, and its caps already say so.
    #[test]
    fn sources_without_live_control_hand_out_no_handle() {
        struct Inert;
        impl SampleSource for Inert {
            fn open(_: &SourceDesc) -> Result<Self, SourceError> {
                Ok(Inert)
            }
            fn stream(&mut self, _: &mut dyn SampleSink) -> Result<(), SourceError> {
                Ok(())
            }
            fn caps(&self) -> ControlCaps {
                ControlCaps::NONE
            }
            fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
                Err(SourceError::UnsupportedControl {
                    backend: "inert",
                    control: ctl.name(),
                })
            }
            fn meta(&self) -> SourceMeta {
                SourceMeta {
                    sample_rate_hz: None,
                    center_freq_hz: None,
                    label: "inert".into(),
                    provenance: "inert".into(),
                }
            }
        }
        let mut inert = Inert;
        assert!(inert.control_handle().is_none());
        assert!(!inert.caps().tune);
        assert!(inert.caps().tune_range.is_none());
    }
}
