// SPDX-License-Identifier: MIT

//! The batch pipeline: source → ring buffer → FFT → accumulators → snapshots
//! (spec §7.0/§8.3, lane M1-D / D-027).
//!
//! ## Shape
//!
//! * A **source thread** runs [`SampleSource::stream`], delivering `cf32`
//!   blocks into a [`RingSink`]: a [`Batcher`] assembles whole N-sample
//!   batches (D-013: a batch is the block of samples feeding one FFT) and
//!   writes each complete batch into a preallocated SPSC ring sized for
//!   ≥ 250 ms at the configured rate (NFR-P3).
//! * A **compute thread** pops whole batches, runs the §7.2 spectrum, feeds
//!   the §7.4 [`LiveTrace`], and publishes the trace snapshot the render
//!   thread copies out at its own cadence (§7.0: the two rates are fully
//!   decoupled).
//! * [`PipelineHealth`] (phosphene-core) is the single accounting both
//!   threads record into and the HUD reads from — the FR-D11 figures cannot
//!   disagree with each other because there is only one set of records.
//!
//! ## One time base for the rolling window (D-028)
//!
//! D-013's percentage is a ratio over **one window on one clock**: a batch
//! is attributed to the same instant in the numerator and the denominator —
//! its **receipt time** at the producer. A dropped batch is received and
//! shed in the same moment; a completed batch's receipt stamp travels with
//! it through a second SPSC ring (one `f64` per batch, preallocated,
//! written before the samples commit so the consumer always finds it), and
//! `record_completed` is called with that stamp once the FFT has actually
//! run. Without this, a batch received at `T` would enter the numerator at
//! `T + δ` while its shed sibling entered the denominator at `T` — and the
//! ring NFR-P3 sizes at ≥ 250 ms makes δ a quarter of the 1 s window
//! precisely when the HUD matters.
//!
//! ## The degradation contract (NFR-P3, D-013)
//!
//! Flow control depends on who paces the stream:
//!
//! * **Time-paced sources** ([`PipelineConfig::paced`] — throttled file
//!   replay now, live radios later): by default cannot be told to wait, so
//!   when the ring is full the producer sheds **whole batches**, counts
//!   them, and moves on — never a partial batch, never decimation within
//!   one. The exception is [`PipelineConfig::backpressure`] (D-125
//!   Amendment 4 item 3): a file-backed pipeline sets both `paced` and
//!   `backpressure`, and a full ring then waits for room instead of
//!   shedding, same as a pull-paced source below — a file falls behind
//!   real time rather than losing samples.
//! * **Pull-paced sources** (the signal generator, unthrottled replay per
//!   clarification C5's "as fast as the consumer accepts") are paced by
//!   consumption itself: the producer waits for ring space, so input can
//!   never outpace compute and the drop count is honestly zero.
//!
//! ## Hot-path allocation (§7.7)
//!
//! Everything is preallocated at start-up: the ring, the batcher staging, the
//! per-drain spectra buffer, the trace buffers. The per-batch path performs
//! ring reads/writes, `copy_from_slice`, FFT into preallocated scratch, and
//! mutex-guarded counter updates — no allocation. The one deliberate
//! exception is a *reconfiguration*: for a source with no known rate the
//! live-trace EMA interval is measured rather than invented (C5), and the
//! trace is rebuilt on the rare occasions the measured cadence shifts by more
//! than 2× — an event, not a per-batch cost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use phosphene_core::{
    min_ring_samples, Batcher, Complex, HealthSnapshot, LiveTrace, MaxHold, MaxHoldMode,
    PersistenceHistogram, PipelineHealth, SpectrumAnalyzer, WindowKind, DBFS_FLOOR, DEFAULT_LEVELS,
    DEFAULT_TAU_LIVE,
};
use phosphene_render::{
    derive_cadence, RowAggregation, RowAggregator, RowCadence, RowRing, RowTimeBase, WaterfallMode,
};
use phosphene_sources::source::{ControlHandle, TuneRange};
use phosphene_sources::{
    Control, ControlCaps, FileSource, SampleSink, SampleSource, SigGen, SinkFlow, SourceDesc,
    SourceMeta,
};

/// Rolling window for the FR-D11 figures, seconds. Display smoothing only —
/// the D-013 definitions of the numbers do not depend on it.
pub const HEALTH_WINDOW_S: f64 = 1.0;

/// Ring capacity when the source declares no sample rate (clarification C5:
/// a rate is never invented, so NFR-P3's 250 ms cannot be computed; such
/// sources are pull-paced anyway, making the ring pure flow control).
const UNRATED_RING_SAMPLES: usize = 1 << 20;

/// Most batches folded into the accumulators per compute-thread wake-up —
/// one §7.0 "K" worth of work per pass, so the closed-form §7.4 batch blend
/// is exercised and the trace lock is taken per drain, not per FFT.
const MAX_BATCHES_PER_DRAIN: usize = 16;

/// Sleep when a thread has nothing to do (ring empty / ring full under
/// backpressure). Well under a display frame, so added latency is invisible
/// against NFR-P4's 50 ms budget.
const IDLE_SLEEP: Duration = Duration::from_micros(300);

/// How the pipeline consumes a source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineConfig {
    /// FFT size N — also the batch size (D-013).
    pub fft_size: usize,
    /// Analysis window.
    pub window: WindowKind,
    /// `true` for sources delivering under real-time pacing (ring-full sheds
    /// whole batches, counted — NFR-P3); `false` for pull-paced sources
    /// (producer waits — clarification C5). See the module docs.
    pub paced: bool,
    /// D-125 Amendment 4 item 3, the owner's ruling: "if reading from a live
    /// source, shed the count; if reading from a file — lag behind." `true`
    /// for a file-backed source, and it reaches into **both** rings between
    /// source and analysis, not just the last one:
    ///
    /// * The compute thread, in [`fold_run`], waits for the analysis worker
    ///   to make room in the bounded `frame_ring` before folding a new
    ///   spectrum into it, rather than silently overwriting an unconsumed
    ///   one.
    /// * [`RingSink::push`] (the source thread) also waits for room in the
    ///   *sample* ring instead of shedding a full one (`paced`'s own
    ///   default), because that same compute thread is the one draining it
    ///   — while it is stalled in the frame-ring wait above, an unmodified
    ///   `paced` sink would shed whole raw sample batches before they ever
    ///   became frames, defeating the frame-ring fix one layer upstream
    ///   (the fix-round-1 bug this doc comment now describes correctly).
    ///
    /// Either way the file falls behind real time, but every frame it
    /// produces is eventually analysed. `false` (every other source, live
    /// radios included) keeps today's behaviour at both layers: ring-full
    /// sheds, counted, at the sample ring (`paced`, NFR-P3), and
    /// `frame_ring` always accepts the newest frame immediately, oldest
    /// silently overwritten when analysis cannot keep up (NFR-A5's honest
    /// shed). A rated file source is `paced: true, backpressure: true` at
    /// once: `paced` shapes the sample ring's *default* (shed), and
    /// `backpressure` overrides that default at both rings to "wait".
    pub backpressure: bool,
    /// Bottom of the displayed dB range at start-up, dBFS — the §7.3
    /// histogram accumulates against `[db_bottom, db_top]`, the display's own
    /// window. When the display moves it at runtime (M1-C's keys), the frame
    /// read ([`Pipeline::intensity_frame`]) tracks it.
    pub db_bottom: f32,
    /// Top of the displayed dB range (the reference level) at start-up, dBFS.
    pub db_top: f32,
}

/// Ring capacity in samples for a source: NFR-P3's ≥ 250 ms **at the
/// configured rate** when one is declared, with a floor of a few batches so
/// tiny rates still pipeline; a fixed preallocation for rate-less sources.
pub fn ring_capacity_samples(
    sample_rate_hz: Option<f64>,
    fft_size: usize,
) -> Result<usize, String> {
    let floor = fft_size.saturating_mul(8);
    match sample_rate_hz {
        Some(rate) => {
            let min =
                min_ring_samples(rate).map_err(|e| format!("cannot size the ring buffer: {e}"))?;
            Ok(min.max(floor))
        }
        None => Ok(UNRATED_RING_SAMPLES.max(floor)),
    }
}

/// Open the backend a [`SourceDesc`] names, boxed for the source thread.
pub fn open_source(desc: &SourceDesc) -> Result<Box<dyn SampleSource + Send>, String> {
    match desc {
        SourceDesc::SigGen(config) => Ok(Box::new(
            SigGen::new(config.clone()).map_err(|e| e.to_string())?,
        )),
        SourceDesc::File(config) => Ok(Box::new(
            FileSource::new(config.clone()).map_err(|e| e.to_string())?,
        )),
        // Always-compiled seam (lane S1-A): without the `soapy` feature this
        // is the §8.4 rebuild-flag error, never a mystery failure.
        SourceDesc::Soapy(config) => {
            phosphene_sources::soapy::open(config).map_err(|e| e.to_string())
        }
        // SourceDesc is #[non_exhaustive]: backends land with their lanes.
        _ => Err("this build has no backend for that source".to_owned()),
    }
}

/// The §7.8 compute tiers. Only the scalar CPU tier exists today; when
/// simd/gpu land, [`ComputeBackend::select`] becomes the runtime probe of
/// §7.8 step 4 and everything downstream (the HUD included) follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// cpu-basic (§7.8 tier 1): rustfft, scalar — always present.
    Cpu,
}

impl ComputeBackend {
    /// Probe and select the compute path. Trivial while only one tier
    /// exists, but this is the single selection point: the HUD label comes
    /// from what was selected here, so it can never contradict the probe.
    pub fn select() -> ComputeBackend {
        ComputeBackend::Cpu
    }

    /// The FR-D11 HUD label for this path (`cpu` / `cpu-simd` / `gpu`).
    pub fn label(self) -> &'static str {
        match self {
            ComputeBackend::Cpu => "cpu",
        }
    }
}

/// The running pipeline. Dropping it stops both threads.
pub struct Pipeline {
    shared: Arc<Shared>,
    compute: Option<JoinHandle<()>>,
    source: Option<JoinHandle<()>>,
    fft_size: usize,
    backend: ComputeBackend,
    /// What the source said it can control (FR-S7), captured once before the
    /// source moved to its own thread — `caps()` is static per source.
    caps: ControlCaps,
    /// The live-control channel to the streaming source (D-056/D-058), or
    /// `None` for a source with no live control. The **only** way this
    /// process reaches `SampleSource::set` while streaming.
    control: Option<ControlHandle>,
    /// The source's label, so a refused control can name what refused it
    /// (§8.4).
    label: String,
    /// AI-1's render seam (D-017): always present, regardless of the
    /// `analyze` feature or the `--analyze` flag — an unused, empty feed is
    /// indistinguishable from v1 to anything that reads it (AC-11), and
    /// callers never need to feature-gate the `chrome::draw` call that takes
    /// it.
    annotation_feed: Arc<phosphene_render::AnnotationFeed>,
    /// Runtime state of the live analysis worker (FR-AU1, NFR-A5) — spawned
    /// unconditionally when this binary is compiled with `analyze`, inactive
    /// (and therefore producing nothing) until [`Pipeline::set_analyze_active`].
    #[cfg(feature = "analyze")]
    analysis: Arc<crate::analyze::AnalysisHandle>,
    #[cfg(feature = "analyze")]
    analysis_worker: Option<JoinHandle<()>>,
}

struct Shared {
    /// Time origin for the whole pipeline: every health record and snapshot
    /// uses seconds since this instant.
    start: Instant,
    stop: AtomicBool,
    health: Mutex<PipelineHealth>,
    trace: Mutex<TraceBuf>,
    /// The §7.3 persistence histogram (FR-D1 — D-033: M1-H's reason to
    /// exist), preallocated at start-up (§7.7). The compute thread folds
    /// every completed spectrum in (one lock per drain); the frame loop
    /// ticks and reads at display cadence. Never held together with another
    /// lock — see the locking note on [`compute_loop`].
    histo: Mutex<PersistenceHistogram>,
    /// The §7.5 max hold (FR-D3 — D-034: the render input and draw path now
    /// exist, and M1-I wires the feature end to end), preallocated at
    /// start-up (§7.7). Accumulated beside the histogram on the compute
    /// thread; the frame loop applies the display's D-007 mode, ticks the
    /// decay against the live trace and reads at display cadence
    /// ([`Pipeline::max_hold_frame`]). Same locking discipline as `histo`.
    max_hold: Mutex<MaxHold>,
    /// The §7.4 live-trace averaging τ the display has requested (FR-D2
    /// "averaging time configurable" — M1-J, the fourth dead control of
    /// D-036). The [`LiveTrace`] itself lives on the compute thread, which
    /// adopts this value at the top of each drain — a keypress-rate write on
    /// one side, an equality check on the other.
    live_tau: Mutex<f32>,
    meta: Mutex<SourceMeta>,
    /// `Some` once the source thread finished: `Ok` on a clean end of stream,
    /// `Err` with the source's own message on failure (§8.4).
    source_done: Mutex<Option<Result<(), String>>>,
    /// Device-reported overflow EVENTS (D-050): "the device lost an unknown
    /// amount", counted per event and never converted into a sample or
    /// batch quantity. The D-043 HUD's `ovfl` field reads it via
    /// [`Pipeline::device_overflow_events`].
    device_overflows: AtomicU64,
    /// The §7.6 waterfall feed (D-051): the aggregator that folds **every**
    /// completed spectrum — on the compute thread, beside the histogram and
    /// max hold — plus the pending-row ring the display drains once per
    /// frame for one batched upload (D-047). Same locking discipline as
    /// `histo`: one lock per drain, never held with another.
    wf: Mutex<WfShared>,
    /// Batches shed since the waterfall's last fold (producer-side drops and
    /// D-048 device-reported losses). The compute thread converts them into
    /// aggregator gap time — a shed batch is signal time with no data, and
    /// §7.6 renders it as floor rather than silently splicing the axis.
    wf_dropped: AtomicU64,
    /// Metadata announced by a **retune** and not yet true of the samples the
    /// compute thread is folding (D-056). Each entry is `(epoch, meta)`: the
    /// source thread stamps the epoch onto every batch it accepts from that
    /// instant on, and the compute thread publishes the meta — and resets the
    /// frequency-dependent accumulators — at exactly the batch where the
    /// epoch changes. This is why the axis never labels old-centre history
    /// with a new centre: the label and the data move together, in band.
    /// A `Vec` because retunes are keypress-rate events; it holds one entry
    /// for the fraction of a second a retune is in flight.
    pending_meta: Mutex<Vec<(u64, SourceMeta)>>,
    /// Bumped once per applied retune reset, by the compute thread. The frame
    /// loop watches it to clear the render-side waterfall history that the
    /// pipeline cannot reach (`window::consume_waterfall_reset`) — the
    /// consumer half of D-056's reset.
    retune_gen: AtomicU64,
    /// The retune epoch that must be applied **without waiting for a batch
    /// behind it** (**D-060**), or 0 for "none pending".
    ///
    /// The ordinary reset is sequenced by the samples themselves: a retune
    /// stamps a new epoch onto every batch accepted from that instant, and
    /// the compute thread resets when it crosses the boundary — exactly
    /// where the new centre starts being true. That mechanism needs a batch
    /// under the new epoch to exist.
    ///
    /// After a tune that could not be read back, no such batch is ever
    /// coming: the stream is ending, and the samples still in flight belong
    /// to a centre the radio has already left. So the reset is announced
    /// here instead and the compute thread applies it on its next pass,
    /// whether or not anything arrives.
    forced_epoch: AtomicU64,
    /// AI-1's `SpectrumTap` backing store (nextgen spec §2.2): the most
    /// recent completed spectra, filled beside `histo`/`max_hold`/`wf` in
    /// [`fold_run`]. Deliberately not reset by a retune the way those three
    /// are (D-056): its bounded capacity means stale, pre-retune frames age
    /// out within one analysis cycle on their own, and the analysis worker
    /// that reads it discards its own stale state on the same
    /// `retune_gen` signal those accumulators' consumers already watch.
    #[cfg(feature = "analyze")]
    frame_ring: Mutex<FrameRing>,
    /// D-125 Amendment 4 item 3: the live analysis worker's own handle —
    /// the same `Arc` [`Pipeline::analysis`] holds — shared here so
    /// [`fold_run`]'s file-source backpressure can read `frames_analysed()`
    /// to decide whether pushing into `frame_ring` must wait.
    #[cfg(feature = "analyze")]
    analysis: Arc<crate::analyze::AnalysisHandle>,
    /// D-125 Amendment 4 item 3: `true` for a file-backed pipeline —
    /// [`PipelineConfig::backpressure`], copied in at construction.
    #[cfg(feature = "analyze")]
    backpressure: bool,
    /// D-125 Amendment 4 item 3: total frames ever folded into `frame_ring`
    /// since the pipeline started (reset alongside the other retune-reset
    /// accumulators in [`apply_retune`], so it stays on the same baseline
    /// as `analysis`'s own retune-reset counters). Compared against
    /// `analysis.frames_analysed()` in [`fold_run`] to decide whether a
    /// file-backed pipeline's next push must wait.
    #[cfg(feature = "analyze")]
    frames_pushed_to_ring: AtomicU64,
    /// Batches actually folded into the display accumulators (histogram, max
    /// hold, waterfall, `frame_ring`) so far — D-123 (FL-5). Bumped once per
    /// [`fold_run`] call, by however many spectra that run folded, **after**
    /// every accumulator it touches has been updated and its lock released.
    /// A caller that waits for this to reach a count and then reads one of
    /// those accumulators is waiting on the condition it asserts, not on
    /// `health().batches_completed` (bumped earlier, when a batch's FFT
    /// completes but before `fold_run` has folded it — D-013's own
    /// accounting, untouched here) as a proxy for it (D-094). Deliberately
    /// not part of [`HealthSnapshot`]: it describes the display
    /// accumulators' own state, not the honesty HUD's contract.
    batches_folded: AtomicU64,
    /// D-123 (FL-5) test-only reproduction hook. When set, the compute
    /// thread pauses in [`fold_run`]'s call site — after `record_completed`
    /// has counted a drain's batches but before that drain is folded —
    /// reproducing the tau-decay flake's mechanism deterministically instead
    /// of relying on the microseconds-wide natural race (see
    /// [`fl5_test_fold_delay`]'s doc comment). The field, and therefore the
    /// delay, exists only in test builds.
    #[cfg(test)]
    test_fold_delay: AtomicBool,
}

/// What travels alongside each batch in the stamp ring: the D-028 receipt
/// instant, and the **retune epoch** the batch was received under (D-056).
///
/// The epoch is what makes the accumulator reset exact rather than
/// approximate. A retune is announced on the source thread while whole
/// batches at the *old* centre are still in the ring; wall-clock coordination
/// would either wipe history the new centre had already produced or leave
/// old-centre spectra to be folded in after the wipe. Tagging each batch
/// instead puts the boundary between two adjacent batches on the one thread
/// that folds them, so no spectrum at the old centre can survive the reset
/// and none at the new centre is thrown away.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BatchTag {
    /// Receipt instant at the producer, seconds since the pipeline start
    /// (D-028: one time base for numerator and denominator).
    t_receipt: f64,
    /// Retune generation this batch was received under (D-056).
    epoch: u64,
}

/// The compute-side half of the waterfall feed (D-051).
struct WfShared {
    agg: RowAggregator,
    ring: RowRing,
    /// Time one spectrum advances the aggregator, in the cadence's own unit
    /// (D-054): `fft_size / sample_rate` seconds for a rated source, exactly
    /// 1.0 for a rate-less one (its clock is the FFT itself).
    dt: f32,
}

/// AI-1 (nextgen spec §2.2): a bounded circular history of completed power
/// spectra, filled on the compute thread beside `histo`/`max_hold`/`wf`
/// (same locking discipline — one lock per drain), read by
/// [`PipelineSpectrumTap`]. Unlike [`phosphene_core::tap::FrameView`] (a
/// caller-filled, then-drained snapshot buffer), this is the tap's own
/// backing store: it always holds the most recent `cap` frames, oldest
/// silently overwritten. No frame it ever holds is skipped when read out —
/// gaps are visible only across polls, when frames were overwritten before
/// a consumer got to them (AI-1's own NFR-A5 overload signal).
#[cfg(feature = "analyze")]
struct FrameRing {
    bins: usize,
    cap: usize,
    /// `cap × bins`, row-major.
    data: Vec<f32>,
    seqs: Vec<u64>,
    write_pos: usize,
    len: usize,
    next_seq: u64,
}

#[cfg(feature = "analyze")]
impl FrameRing {
    fn new(bins: usize, cap: usize) -> Self {
        FrameRing {
            bins,
            cap,
            data: vec![0.0; bins * cap],
            seqs: vec![0; cap],
            write_pos: 0,
            len: 0,
            next_seq: 0,
        }
    }

    fn push(&mut self, frame: &[f32]) {
        let slot = self.write_pos;
        self.data[slot * self.bins..(slot + 1) * self.bins].copy_from_slice(frame);
        self.seqs[slot] = self.next_seq;
        self.next_seq += 1;
        self.write_pos = (self.write_pos + 1) % self.cap;
        self.len = (self.len + 1).min(self.cap);
    }

    /// Fill `out` with up to `out.max_frames()` of the most recent frames,
    /// oldest first — the [`phosphene_core::tap::SpectrumTap::latest_frames`]
    /// contract exactly (mirrored by `OfflineTap`'s own implementation):
    /// fewer, never more than the caller's own view can hold.
    fn fill_latest(&self, out: &mut phosphene_analyze::FrameView) {
        out.clear();
        let take = out.max_frames().min(self.len);
        let start = (self.write_pos + self.cap - take) % self.cap;
        for i in 0..take {
            let slot = (start + i) % self.cap;
            out.push(
                &self.data[slot * self.bins..(slot + 1) * self.bins],
                self.seqs[slot],
            );
        }
    }
}

/// AI-1's `SpectrumTap` implementation over the power spectra the compute
/// thread already produces (nextgen spec §2.2: "the display already
/// computes these power frames; the tap just exposes the latest"). Reads
/// only — never recomputes, never blocks the compute thread beyond the same
/// short-held lock `histo`/`max_hold`/`wf` already use.
#[cfg(feature = "analyze")]
pub struct PipelineSpectrumTap {
    shared: Arc<Shared>,
    fft_size: usize,
}

#[cfg(feature = "analyze")]
impl phosphene_analyze::SpectrumTap for PipelineSpectrumTap {
    fn latest_frames(&self, out: &mut phosphene_analyze::FrameView) {
        self.shared
            .frame_ring
            .lock()
            .expect("frame ring poisoned")
            .fill_latest(out);
    }

    fn meta(&self) -> phosphene_analyze::BandMeta {
        let meta = self
            .shared
            .meta
            .lock()
            .expect("source meta poisoned")
            .clone();
        let bin_hz = match meta.sample_rate_hz {
            Some(rate) => rate / self.fft_size as f64,
            // D-028 §2 / D-098 §7.3: no declared rate, so the axis (and
            // every annotation on it) is normalized cycles/sample over a
            // span of exactly 1.0 — never an invented Hz figure.
            None => 1.0 / self.fft_size as f64,
        };
        phosphene_analyze::BandMeta {
            center_freq_hz: meta.center_freq_hz,
            span_hz: bin_hz * self.fft_size as f64,
            bins: self.fft_size,
            bin_hz,
            // Matches `WfShared::dt`'s existing convention (D-054): the
            // real cadence for a rated source, one abstract FFT-tick for a
            // rate-less one whose only clock is the FFT itself.
            frame_dt_s: match meta.sample_rate_hz {
                Some(rate) => self.fft_size as f64 / rate,
                None => 1.0,
            },
            cal_offset_db: 0.0,
        }
    }
}

/// Memory budget for the pending-row ring: enough headroom for a display
/// hiccup at the fastest realistic cadence without §7.7-hostile bulk at
/// large FFT sizes. Rows beyond it overwrite the oldest pending row, which
/// the RING_ROWS-deep ring texture would have overwritten anyway.
const WF_PENDING_BUDGET_BYTES: usize = 16 << 20;

/// Pending-row capacity for a given spectrum width.
fn wf_pending_rows(bins: usize) -> usize {
    (WF_PENDING_BUDGET_BYTES / (bins * 4)).clamp(64, phosphene_render::RING_ROWS as usize)
}

impl Shared {
    fn now_s(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// D-125 Amendment 4 item 3 fix round (review 1): a file-backed pipeline
    /// is `paced: true, backpressure: true` at once — paced so the *sample*
    /// ring between the source and compute threads is real-time-shaped,
    /// backpressure so the *frame* ring between compute and analysis never
    /// silently drops one. Those two rings are fed by the same compute
    /// thread; when [`fold_run`]'s frame-ring wait stalls that thread, it
    /// stops draining the sample ring too, so a paced [`RingSink`] governed
    /// by `paced` alone would shed whole raw sample batches before they ever
    /// become frames — "a file analyses every frame" silently broken one
    /// layer upstream of where item 3 was proven. So `paced` sources also
    /// wait for room in the sample ring, never shed, whenever backpressure
    /// is configured — the same "wait, don't drop" contract item 3 already
    /// gives the frame ring, one layer earlier.
    #[cfg(feature = "analyze")]
    fn wants_backpressure(&self) -> bool {
        self.backpressure
    }
    #[cfg(not(feature = "analyze"))]
    fn wants_backpressure(&self) -> bool {
        false
    }
}

struct TraceBuf {
    /// Latest live trace, dBFS/bin, DC-centered; [`DBFS_FLOOR`] before the
    /// first published batch.
    data: Vec<f32>,
    /// Bumped per publish; 0 means nothing published yet.
    seq: u64,
}

impl Pipeline {
    /// Spawn the source and compute threads over `source`.
    pub fn start(
        source: Box<dyn SampleSource + Send>,
        config: PipelineConfig,
    ) -> Result<Pipeline, String> {
        Self::start_inner(source, config, None)
    }

    /// Test-only: a pipeline whose compute thread consumes one `gate` permit
    /// per batch — the injected slow consumer the D-028 backlog seal (and
    /// the §7.6 feed seals in `crate::window`, which need the aggregator
    /// configured before spectra flow) drives deterministically through the
    /// real two-thread pipeline.
    #[cfg(test)]
    pub(crate) fn start_gated(
        source: Box<dyn SampleSource + Send>,
        config: PipelineConfig,
        gate: Arc<AtomicU64>,
    ) -> Result<Pipeline, String> {
        Self::start_inner(source, config, Some(gate))
    }

    fn start_inner(
        mut source: Box<dyn SampleSource + Send>,
        config: PipelineConfig,
        gate: Option<Arc<AtomicU64>>,
    ) -> Result<Pipeline, String> {
        let n = config.fft_size;
        let meta = source.meta();
        // FR-S7/D-056: what this source can control, and the channel that
        // reaches it while it streams. Both are taken before the source moves
        // onto its own thread — after that, nothing in this process holds it.
        let caps = source.caps();
        let control = source.control_handle();
        let label = meta.label.clone();
        let backend = ComputeBackend::select();
        let analyzer = SpectrumAnalyzer::new(n, config.window).map_err(|e| e.to_string())?;
        let health = PipelineHealth::new(n, HEALTH_WINDOW_S).map_err(|e| e.to_string())?;
        let batcher = Batcher::new(n).map_err(|e| e.to_string())?;
        let capacity = ring_capacity_samples(meta.sample_rate_hz, n)?;
        let (producer, consumer) = rtrb::RingBuffer::<Complex<f32>>::new(capacity);
        // Receipt stamps, one per in-flight batch, alongside the sample ring
        // (D-028: a completed batch is attributed at its receipt time). The
        // +2 keeps the stamp ring strictly roomier than the sample ring can
        // ever be in whole batches.
        let (ts_producer, ts_consumer) = rtrb::RingBuffer::<BatchTag>::new(capacity / n + 2);
        let trace_state = TraceState::new(n, meta.sample_rate_hz)?;
        // §7.3: the intensity grid spans the DISPLAYED dB range — the caller
        // hands over the display's own window, never a private one.
        let histo = PersistenceHistogram::new(n, DEFAULT_LEVELS, config.db_bottom, config.db_top)
            .map_err(|e| format!("cannot build the persistence histogram: {e}"))?;
        // §7.5 (FR-D3): default mode and τ are the accumulator's own —
        // decay-toward-live at DEFAULT_TAU_MAX_HOLD (D-007), runtime-switched
        // by the display through `max_hold_frame`.
        let max_hold =
            MaxHold::new(n).map_err(|e| format!("cannot build the max-hold trace: {e}"))?;

        // The §7.6 waterfall feed (D-051), preallocated (§7.7). The starting
        // cadence mirrors the view defaults with a placeholder panel height;
        // the first display frame reconfigures it through
        // [`Pipeline::waterfall_frame`] before any row could mislead.
        let spectrum_interval_s = spectrum_interval_s(meta.sample_rate_hz, n);
        let cadence = derive_cadence(
            WaterfallMode::default(),
            spectrum_interval_s,
            30.0,
            phosphene_render::FAST_INTERVAL_DEFAULT_S,
            phosphene_render::RING_ROWS as f32,
            1,
        );
        let wf = WfShared {
            agg: RowAggregator::new(n, RowAggregation::default(), cadence),
            ring: RowRing::new(n, wf_pending_rows(n)),
            dt: wf_dt(cadence, spectrum_interval_s),
        };

        // D-125 Amendment 4 item 3: built here, before `Shared`, so the
        // exact same `Arc` reaches both `Shared::analysis` (the compute
        // thread's backpressure read) and `Pipeline::analysis` (the
        // worker's own handle, `start_analysis_worker` below) — one
        // handle, not two that could drift.
        #[cfg(feature = "analyze")]
        let analysis_handle = crate::analyze::AnalysisHandle::new();

        let shared = Arc::new(Shared {
            start: Instant::now(),
            stop: AtomicBool::new(false),
            health: Mutex::new(health),
            trace: Mutex::new(TraceBuf {
                data: vec![DBFS_FLOOR; n],
                seq: 0,
            }),
            histo: Mutex::new(histo),
            max_hold: Mutex::new(max_hold),
            live_tau: Mutex::new(DEFAULT_TAU_LIVE),
            meta: Mutex::new(meta),
            source_done: Mutex::new(None),
            device_overflows: AtomicU64::new(0),
            wf: Mutex::new(wf),
            wf_dropped: AtomicU64::new(0),
            pending_meta: Mutex::new(Vec::new()),
            retune_gen: AtomicU64::new(0),
            forced_epoch: AtomicU64::new(0),
            #[cfg(feature = "analyze")]
            frame_ring: Mutex::new(FrameRing::new(n, crate::analyze::TAP_HISTORY_FRAMES)),
            #[cfg(feature = "analyze")]
            analysis: analysis_handle.clone(),
            #[cfg(feature = "analyze")]
            backpressure: config.backpressure,
            #[cfg(feature = "analyze")]
            frames_pushed_to_ring: AtomicU64::new(0),
            batches_folded: AtomicU64::new(0),
            #[cfg(test)]
            test_fold_delay: AtomicBool::new(false),
        });

        let source_shared = shared.clone();
        let paced = config.paced;
        let source_handle = std::thread::Builder::new()
            .name("phosphene-source".into())
            .spawn(move || {
                let mut sink = RingSink {
                    shared: source_shared.clone(),
                    producer,
                    ts_producer,
                    batcher,
                    paced,
                    lost_remainder: 0,
                    epoch: 0,
                    accepted_any: false,
                };
                let result = source.stream(&mut sink).map_err(|e| e.to_string());
                *source_shared.source_done.lock().unwrap() = Some(result);
            })
            .map_err(|e| format!("cannot spawn the source thread: {e}"))?;

        let compute_shared = shared.clone();
        let compute_handle = std::thread::Builder::new()
            .name("phosphene-compute".into())
            .spawn(move || {
                compute_loop(
                    compute_shared,
                    consumer,
                    ts_consumer,
                    analyzer,
                    trace_state,
                    gate,
                )
            })
            .map_err(|e| format!("cannot spawn the compute thread: {e}"))?;

        let annotation_feed = Arc::new(phosphene_render::AnnotationFeed::new());

        // Only mutated by `start_analysis_worker` below, which compiles out
        // entirely without the `analyze` feature.
        #[cfg_attr(not(feature = "analyze"), allow(unused_mut))]
        let mut pipeline = Pipeline {
            shared,
            compute: Some(compute_handle),
            source: Some(source_handle),
            fft_size: n,
            backend,
            caps,
            control,
            label,
            annotation_feed,
            #[cfg(feature = "analyze")]
            analysis: analysis_handle,
            #[cfg(feature = "analyze")]
            analysis_worker: None,
        };
        #[cfg(feature = "analyze")]
        pipeline.start_analysis_worker();
        Ok(pipeline)
    }

    /// Spawn the live analysis worker over [`Self::spectrum_tap`], inactive
    /// until [`Self::set_analyze_active`] (FR-AU1). Called once, from
    /// [`Self::start_inner`].
    #[cfg(feature = "analyze")]
    fn start_analysis_worker(&mut self) {
        let tap = self.spectrum_tap();
        let feed = self.annotation_feed.clone();
        let gen_shared = self.shared.clone();
        let meta_shared = self.shared.clone();
        self.analysis_worker = Some(crate::analyze::spawn(
            tap,
            feed,
            move || gen_shared.retune_gen.load(Ordering::Acquire),
            move || {
                meta_shared
                    .meta
                    .lock()
                    .expect("source meta poisoned")
                    .sample_rate_hz
                    .is_some()
            },
            self.analysis.clone(),
        ));
    }

    /// FFT size N (also the batch size).
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The HUD label of the compute path this pipeline actually selected at
    /// start-up (FR-D11, §7.8 step 4) — see [`ComputeBackend::select`].
    pub fn backend(&self) -> &'static str {
        self.backend.label()
    }

    /// Copy the latest published live trace into `out` (resized to N).
    /// Returns `false` while nothing has been published yet (out is then the
    /// [`DBFS_FLOOR`] baseline).
    pub fn copy_trace(&self, out: &mut Vec<f32>) -> bool {
        let buf = self.shared.trace.lock().unwrap();
        out.clone_from(&buf.data);
        buf.seq > 0
    }

    /// One display frame's read of the §7.3 persistence grid (FR-D1): fold
    /// the counts accumulated since the last call into the intensity grid
    /// (`PersistenceHistogram::tick`) and copy it into `out`. Returns
    /// `(bins, levels)` — the dimensions `RtsaInput` needs alongside the
    /// data. The grid is all zeros until spectra arrive; it is never empty.
    ///
    /// `db_bottom`/`db_top` are the DISPLAYED dB window — `DisplayParams`'
    /// own values, which M1-C's keyboard map moves at runtime. §7.3 pins the
    /// grid to that window, so when it differs from the range the histogram
    /// was accumulating against, the histogram is **rebuilt at the new range
    /// and accumulation restarts** — history quantised against the old
    /// window cannot be honestly re-drawn against the new one. The rebuild
    /// allocates, but only on a reference-level / dB-per-div change — a
    /// keypress-rate event, not a per-batch or per-frame cost (§7.7; same
    /// class as the `TraceState` cadence rebuild below).
    ///
    /// A `dt` that is not finite and positive skips the tick (the grid holds
    /// its state) rather than fabricating a time step.
    pub fn intensity_frame(
        &self,
        dt: f32,
        db_bottom: f32,
        db_top: f32,
        out: &mut Vec<f32>,
    ) -> (usize, usize) {
        let mut histo = self.shared.histo.lock().unwrap();
        if histo.p_bottom() != db_bottom || histo.p_top() != db_top {
            // A degenerate window (top ≤ bottom) cannot be accumulated
            // against: keep the last valid range rather than dying —
            // DisplayParams' keyboard bounds never produce one.
            if let Ok(rebuilt) =
                PersistenceHistogram::new(histo.bins(), histo.levels(), db_bottom, db_top)
            {
                *histo = rebuilt;
            }
        }
        if dt.is_finite() && dt > 0.0 {
            histo.tick(dt);
        }
        let grid = histo.intensity();
        out.resize(grid.len(), 0.0);
        out.copy_from_slice(grid);
        (histo.bins(), histo.levels())
    }

    /// One display frame's step of the §7.6 waterfall feed (D-051): adopt
    /// the view's mode, aggregation and mode-owned quantity — deriving the
    /// cadence in the unit the source's rate makes honest (D-046/D-054) —
    /// then move every row the compute thread finished since the last frame
    /// into `out` (flat `rows × bins`, oldest first) for **one batched
    /// upload** (D-047).
    ///
    /// The returned [`WfDrain`] states the seconds of signal time each row
    /// represents, `None` for a rate-less source — the value the display
    /// hands to the composer so the FR-D5 time labels state the cadence the
    /// rows actually carried (D-031), and never a unit the source does not
    /// have.
    ///
    /// `out` is cleared, then filled without reallocating once warm (§7.7):
    /// the drain is bounded by the pending ring's fixed capacity.
    pub fn waterfall_frame(&self, cfg: &WfFrameConfig, out: &mut Vec<f32>) -> WfDrain {
        let spectrum_interval = spectrum_interval_s(
            self.shared.meta.lock().unwrap().sample_rate_hz,
            self.fft_size,
        );
        let cadence = derive_cadence(
            cfg.mode,
            spectrum_interval,
            cfg.span_s,
            cfg.fast_interval_s,
            cfg.span_rows,
            cfg.panel_rows.max(1),
        );
        let mut wf = self.shared.wf.lock().unwrap();
        wf.agg.set_mode(cfg.aggregation);
        // A no-op unless the cadence really changed (the aggregator resets
        // its open interval on a real change).
        wf.agg.set_cadence(cadence);
        wf.dt = wf_dt(cadence, spectrum_interval);
        out.clear();
        let rows = wf.ring.drain_into(out);
        WfDrain {
            bins: wf.ring.bins(),
            rows,
            cadence: wf.agg.cadence(),
            row_interval_s: wf.agg.row_interval_s(),
        }
    }

    /// One display frame's read of the §7.5 max hold (FR-D3): apply the
    /// display's D-007 `mode` (the keyboard map switches it at runtime —
    /// the trace carries over, only future ticks change behaviour), decay
    /// the retained trace toward `live` — the frame's own copy of the
    /// published live trace, which is why the tick runs here and not on the
    /// compute thread — and copy the trace into `out` (resized to N once,
    /// a start-up cost, not per-frame — §7.7). Accumulation runs on the
    /// compute thread; before the first spectrum the trace is the
    /// [`DBFS_FLOOR`] baseline. A `dt` that is not finite and positive, or
    /// a `live` of the wrong length, skips the tick (the trace holds its
    /// state) rather than fabricating a step.
    pub fn max_hold_frame(&self, dt: f32, mode: MaxHoldMode, live: &[f32], out: &mut Vec<f32>) {
        let mut max_hold = self.shared.max_hold.lock().unwrap();
        if max_hold.mode() != mode {
            max_hold.set_mode(mode);
        }
        if dt.is_finite() && dt > 0.0 && live.len() == max_hold.bins() {
            max_hold.tick(dt, live);
        }
        let trace = max_hold.trace();
        out.resize(trace.len(), DBFS_FLOOR);
        out.copy_from_slice(trace);
    }

    /// FR-D3's reset key: clear the max-hold trace to the [`DBFS_FLOOR`]
    /// baseline; it re-seeds from the next completed spectrum.
    pub fn reset_max_hold(&self) {
        self.shared.max_hold.lock().unwrap().reset();
    }

    /// Multiply the §7.3 persistence decay τ by `factor` — FR-D1's
    /// "persistence time" control, reachable at last (D-035) — clamped to
    /// [`TAU_MIN_S`]..[`TAU_MAX_S`] so held-down keys can never run it away.
    /// Returns the τ now in effect. A keypress-rate event, not a per-frame
    /// cost; the accumulated intensity carries over — only future ticks
    /// decay differently.
    pub fn adjust_persistence_tau(&self, factor: f32) -> f32 {
        let mut histo = self.shared.histo.lock().unwrap();
        let tau = clamp_tau(histo.tau_decay(), factor);
        // The clamp precludes the non-finite/non-positive values set_tau
        // rejects.
        let _ = histo.set_tau_decay(tau);
        histo.tau_decay()
    }

    /// Multiply the D-007 max-hold decay τ by `factor` (D-035: exposed
    /// alongside the persistence τ, through the same key table). Same clamp
    /// and same keypress-rate cost as
    /// [`adjust_persistence_tau`](Self::adjust_persistence_tau).
    pub fn adjust_max_hold_tau(&self, factor: f32) -> f32 {
        let mut max_hold = self.shared.max_hold.lock().unwrap();
        let tau = clamp_tau(max_hold.tau(), factor);
        let _ = max_hold.set_tau(tau);
        max_hold.tau()
    }

    /// Multiply the §7.4 live-trace averaging τ by `factor` — FR-D2's
    /// "averaging time configurable", reachable at last (M1-J closes the
    /// fourth dead control of D-036) — with the same clamp, stepping and
    /// keypress-rate cost as the other two τ controls (D-035: one idiom,
    /// not a third). Returns the τ now in effect; the compute thread, which
    /// owns the EMA, adopts it at the top of its next drain.
    pub fn adjust_live_tau(&self, factor: f32) -> f32 {
        let mut tau = self.shared.live_tau.lock().unwrap();
        *tau = clamp_tau(*tau, factor);
        *tau
    }

    /// The §7.3 persistence decay τ now in effect (D-043 HUD readout): the
    /// same value [`adjust_persistence_tau`](Self::adjust_persistence_tau)
    /// reads and writes, never a mirrored copy that could drift from it.
    pub fn persistence_tau(&self) -> f32 {
        self.shared.histo.lock().unwrap().tau_decay()
    }

    /// The D-007 max-hold decay τ now in effect (D-043 HUD readout), read
    /// from the same accumulator
    /// [`adjust_max_hold_tau`](Self::adjust_max_hold_tau) mutates.
    pub fn max_hold_tau(&self) -> f32 {
        self.shared.max_hold.lock().unwrap().tau()
    }

    /// The §7.4 live-trace averaging τ now in effect (D-043 HUD readout),
    /// read from the same shared value
    /// [`adjust_live_tau`](Self::adjust_live_tau) mutates.
    pub fn live_tau(&self) -> f32 {
        *self.shared.live_tau.lock().unwrap()
    }

    /// What the active source says it can control (FR-S7), captured from the
    /// source before it moved to its own thread — `caps()` is static per
    /// source, so this cannot drift from it.
    pub fn caps(&self) -> ControlCaps {
        self.caps
    }

    /// The centre-frequency range the **device** reports it can reach
    /// (D-058), or `None` when the source cannot tune or reports no range.
    /// Never a guess: a UI given `None` offers no bounded control rather than
    /// one bounded by an invented limit.
    pub fn tune_range(&self) -> Option<TuneRange> {
        self.caps.tune_range
    }

    /// **Live retune** (D-056/D-058): ask the streaming source to move to
    /// `target_hz` and return the centre frequency the **device reports it
    /// actually reached**.
    ///
    /// This is the one retune path in the process. Both D-058 interactions —
    /// the frequency box and the window-stepping slider — call it, so there
    /// is one mechanism and one source of truth about where the radio is.
    ///
    /// What it guarantees, in the order it guarantees them:
    ///
    /// 1. **A source that cannot tune is refused here**, before any channel
    ///    is involved, with an error naming the source (§8.4). Never a silent
    ///    no-op that would leave the display asserting a frequency the radio
    ///    is not on.
    /// 2. **The device is the arbiter.** The returned value is the device's
    ///    own readback (C5) — a clamped or snapped tune returns where the
    ///    radio went, not where it was asked to go — and a device that
    ///    refuses returns its own error.
    /// 3. **The display follows the samples, not this call.** A successful
    ///    retune announces the readback in band; [`Pipeline::meta`] starts
    ///    reporting it at the exact batch the new centre becomes true of, and
    ///    D-056's accumulator reset happens in the same move. So the axis
    ///    never labels old-centre history with a new centre, not even for one
    ///    frame.
    ///
    /// Blocks until the source answers, at most `CONTROL_TIMEOUT`. A
    /// keypress-rate call — never on a per-frame path.
    pub fn retune(&self, target_hz: f64) -> Result<f64, String> {
        if !target_hz.is_finite() || target_hz <= 0.0 {
            return Err(format!(
                "center frequency must be finite and positive, got {target_hz} Hz"
            ));
        }
        if !self.caps.tune {
            return Err(format!(
                "the {} source cannot be tuned: it has no device to retune \
                 (a file, a pipe or the signal generator plays back what it \
                 was given)",
                self.label
            ));
        }
        let Some(control) = &self.control else {
            return Err(format!(
                "the {} source advertises tuning but offers no live control \
                 channel, so it cannot be retuned while streaming",
                self.label
            ));
        };
        let meta = control
            .set(Control::CenterFreqHz(target_hz))
            .map_err(|e| e.to_string())?;
        meta.center_freq_hz.ok_or_else(|| {
            format!(
                "the {} source applied the retune but reports no center \
                 frequency, so there is nothing to label the axis with",
                self.label
            )
        })
    }

    /// How many times D-056's retune reset has been applied, counted by the
    /// compute thread as it crosses each retune boundary.
    ///
    /// The frame loop watches this to clear the waterfall history that lives
    /// in the render crate's ring texture, which the pipeline cannot reach —
    /// see `window::consume_waterfall_reset`. It changes when the reset
    /// actually happens (at the first batch under the new centre), never when
    /// the retune is merely requested.
    ///
    /// Acquire, paired with the Release store in `apply_retune`: a caller
    /// that observes a new generation here is thereby guaranteed to see
    /// every accumulator reset that produced it too, even though each is
    /// guarded by its own separate lock rather than one lock shared with
    /// this counter.
    pub fn retune_generation(&self) -> u64 {
        self.shared.retune_gen.load(Ordering::Acquire)
    }

    /// AI-1's `SpectrumTap` over this pipeline's own power spectra (nextgen
    /// spec §2.2). Cheap to clone (an `Arc` handle); a fresh one may be
    /// taken at any time.
    #[cfg(feature = "analyze")]
    pub fn spectrum_tap(&self) -> PipelineSpectrumTap {
        PipelineSpectrumTap {
            shared: self.shared.clone(),
            fft_size: self.fft_size,
        }
    }

    /// AI-1's render seam (D-017): the feed the live analysis worker
    /// publishes into, and `chrome::draw`/`draw_tunable` read from every
    /// frame. Always present and always safe to pass, whether or not this
    /// binary was built with `analyze` or the worker is currently active
    /// (AC-11: an inactive/absent worker leaves it empty forever).
    pub fn annotation_feed(&self) -> Arc<phosphene_render::AnnotationFeed> {
        self.annotation_feed.clone()
    }

    /// Turn the live analysis worker on/off (the `--analyze` flag's initial
    /// state, and the `I` key thereafter, FR-AU1). A no-op when this binary
    /// was not built with the `analyze` feature.
    pub fn set_analyze_active(&self, active: bool) {
        #[cfg(feature = "analyze")]
        self.analysis.set_active(active);
        #[cfg(not(feature = "analyze"))]
        let _ = active;
    }

    /// Flip the live analysis worker on/off (the `I` key, §2.7). A no-op
    /// when this binary was not built with the `analyze` feature.
    pub fn toggle_analyze_active(&self) {
        #[cfg(feature = "analyze")]
        self.analysis.toggle();
    }

    /// Whether the FR-AU1 analysis pipeline is active this frame — the
    /// "INSPECTOR" status chip and the annotation overlay draw pass read
    /// this (design AC-11: always `false` when this binary was not built
    /// with `analyze`).
    #[cfg(feature = "analyze")]
    pub fn inspector_active(&self) -> bool {
        self.analysis.is_active()
    }
    #[cfg(not(feature = "analyze"))]
    pub fn inspector_active(&self) -> bool {
        false
    }

    /// Live track count this cycle (design situations 1/3).
    #[cfg(feature = "analyze")]
    pub fn annotation_track_count(&self) -> usize {
        self.analysis.track_count()
    }
    #[cfg(not(feature = "analyze"))]
    pub fn annotation_track_count(&self) -> usize {
        0
    }

    /// Whether the analysis pipeline is shedding tracks under overload this
    /// cycle (NFR-A5).
    #[cfg(feature = "analyze")]
    pub fn analysis_shedding(&self) -> bool {
        self.analysis.is_shedding()
    }
    #[cfg(not(feature = "analyze"))]
    pub fn analysis_shedding(&self) -> bool {
        false
    }

    /// D-125 Amendment 4 item 3: the honest coverage count, `(frames
    /// analysed, frames received)`, cumulative since the worker started or
    /// last retuned — see [`crate::analyze::AnalysisHandle::
    /// frames_analysed`]/[`frames_received`](crate::analyze::AnalysisHandle::
    /// frames_received).
    #[cfg(feature = "analyze")]
    pub fn analysis_coverage(&self) -> (u64, u64) {
        (
            self.analysis.frames_analysed(),
            self.analysis.frames_received(),
        )
    }
    #[cfg(not(feature = "analyze"))]
    pub fn analysis_coverage(&self) -> (u64, u64) {
        (0, 0)
    }

    /// The FR-D11 figures, as of now, from the pipeline's own accounting.
    pub fn health(&self) -> HealthSnapshot {
        let t = self.shared.now_s();
        self.shared.health.lock().unwrap().snapshot(t)
    }

    /// Latest source metadata (rate/center honestly absent when unknown —
    /// clarification C5).
    pub fn meta(&self) -> SourceMeta {
        self.shared.meta.lock().unwrap().clone()
    }

    /// `true` once the source thread finished **cleanly** — end of stream: a
    /// file fully replayed without `--loop`, or stdin's pipe closed. The
    /// headless loop uses this to stop at end-of-file instead of spinning
    /// (M2-A build step 3); a failed source reports through
    /// [`source_error`](Self::source_error) instead.
    pub fn source_finished(&self) -> bool {
        matches!(&*self.shared.source_done.lock().unwrap(), Some(Ok(())))
    }

    /// If the source thread has failed, its error message (§8.4). A clean end
    /// of stream (file fully replayed, no `--loop`) is not an error: the
    /// display keeps showing the last state and the rates decay to zero.
    pub fn source_error(&self) -> Option<String> {
        match &*self.shared.source_done.lock().unwrap() {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        }
    }

    /// Device-reported overflow events so far (D-050): each is "the device
    /// lost an unknown amount of data". Deliberately NOT part of
    /// [`HealthSnapshot`] — `PROC %` and `DROPS` describe only what the
    /// pipeline received and shed, and an event count must never be dressed
    /// up as one. Read by the HUD's `ovfl` field (D-043/M2-B).
    pub fn device_overflow_events(&self) -> u64 {
        self.shared.device_overflows.load(Ordering::Relaxed)
    }

    /// Batches actually folded into the display accumulators so far (D-123 —
    /// FL-5). `health().batches_completed` counts a batch once its FFT has
    /// run; this counts it once `fold_run` has folded it into the
    /// histogram, max hold, waterfall and `frame_ring`. A caller that means
    /// to wait for a folded read to be safe — as opposed to merely counted —
    /// waits on this, not on `batches_completed` (D-094: wait on the
    /// condition asserted, not a proxy for it). See the field doc on
    /// `Shared::batches_folded` for the ordering that makes this exact.
    ///
    /// No production caller yet — this lane's territory is `pipeline.rs`
    /// only, so wiring a display consumer (the HUD, say) is left to
    /// whichever lane touches `window.rs` next. The counter itself is
    /// production code, always compiled; only its callers today are tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn batches_folded(&self) -> u64 {
        self.shared.batches_folded.load(Ordering::Acquire)
    }

    /// D-125 Amendment 4 item 3's negative controls: total frames ever
    /// folded into `frame_ring` so far — see `Shared::frames_pushed_to_ring`.
    /// `#[cfg(test)]`-only: no production caller, purely for the
    /// backpressure tests to compare against `analysis_coverage()`.
    #[cfg(all(test, feature = "analyze"))]
    pub(crate) fn frames_pushed_to_ring(&self) -> u64 {
        self.shared.frames_pushed_to_ring.load(Ordering::Relaxed)
    }

    /// Stop both threads. The compute thread is joined; the source thread is
    /// given a grace period (it may be parked in a blocking read — stdin —
    /// that only process exit can interrupt) and detached if still blocked.
    pub fn shutdown(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        #[cfg(feature = "analyze")]
        {
            self.analysis.stop();
            if let Some(handle) = self.analysis_worker.take() {
                let _ = handle.join();
            }
        }
        if let Some(handle) = self.compute.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.source.take() {
            let deadline = Instant::now() + Duration::from_millis(250);
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
            // Otherwise: detached; a blocking stdin read ends with the
            // process.
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The view-owned inputs to one [`Pipeline::waterfall_frame`] step: the
/// §7.6 mode and aggregation plus every mode-owned quantity — the feed picks
/// the one the mode and the source's rate make meaningful (D-046/D-054) —
/// and the panel height the cadence maps onto.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WfFrameConfig {
    /// The active speed mode (`ViewState::wf_mode`).
    pub mode: WaterfallMode,
    /// Max or mean row aggregation (`ViewState::wf_aggregation`).
    pub aggregation: RowAggregation,
    /// Traditional/rated: the panel time span, seconds.
    pub span_s: f32,
    /// Fast/rated: the row interval, seconds (clamped no finer than one
    /// spectrum interval at derivation).
    pub fast_interval_s: f32,
    /// Traditional/rate-less: the span as a row count (D-054).
    pub span_rows: f32,
    /// Waterfall panel height in texture rows.
    pub panel_rows: u32,
}

/// What one [`Pipeline::waterfall_frame`] drain handed over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WfDrain {
    /// Spectrum width of each row.
    pub bins: usize,
    /// Rows moved into `out` this frame.
    pub rows: usize,
    /// The cadence the rows were aggregated at — interval and time base
    /// (seconds, or spectra for a rate-less source, D-054) — handed to the
    /// composer with the rows so the FR-D5 labels state the axis in the one
    /// honest unit.
    pub cadence: RowCadence,
    /// Seconds of signal time per row — `None` for a rate-less source,
    /// whose cadence is measured in spectra and never restated in a unit it
    /// does not have (D-054). Convenience view of `cadence`.
    pub row_interval_s: Option<f32>,
}

/// `fft_size / sample_rate`: the seconds of signal time one spectrum spans,
/// `None` when the source declares no rate (C5 — never invented).
fn spectrum_interval_s(sample_rate_hz: Option<f64>, fft_size: usize) -> Option<f32> {
    sample_rate_hz
        .filter(|r| r.is_finite() && *r > 0.0)
        .map(|r| (fft_size as f64 / r) as f32)
}

/// The per-spectrum time step for the waterfall aggregator, in the cadence's
/// own unit (D-054): seconds of signal time when the cadence is
/// seconds-based, exactly one spectrum otherwise.
fn wf_dt(cadence: RowCadence, spectrum_interval_s: Option<f32>) -> f32 {
    match cadence.base() {
        // A seconds cadence only derives from a declared rate, so the
        // interval is present; the fallback is unreachable but must not
        // fabricate a wild step if that invariant ever breaks.
        RowTimeBase::Seconds => spectrum_interval_s.unwrap_or(0.0),
        RowTimeBase::Spectra => 1.0,
    }
}

/// Bounds for the runtime τ controls (D-035): generous around §7.3's
/// suggested 0.25–2 s user range, but a τ can never be driven to zero or
/// infinity from the keyboard.
const TAU_MIN_S: f32 = 0.05;
const TAU_MAX_S: f32 = 30.0;

/// One τ keypress: multiply and clamp. A non-finite `factor` (impossible
/// from the keymap's fixed step, but this is the safety boundary) leaves the
/// τ unchanged.
fn clamp_tau(current: f32, factor: f32) -> f32 {
    if factor.is_finite() && factor > 0.0 {
        (current * factor).clamp(TAU_MIN_S, TAU_MAX_S)
    } else {
        current
    }
}

/// The producer side: batcher + ring + drop accounting (§8.2: drop
/// accounting lives at the producer).
struct RingSink {
    shared: Arc<Shared>,
    producer: rtrb::Producer<Complex<f32>>,
    /// Receipt stamp and retune epoch, one per accepted batch, paired FIFO
    /// with the sample ring (D-028/D-056).
    ts_producer: rtrb::Producer<BatchTag>,
    batcher: Batcher,
    paced: bool,
    /// Device-reported lost samples not yet amounting to a whole batch
    /// (D-048): the accounting is batch-grained (D-013), so losses accrue
    /// here and convert to dropped batches as they fill one.
    lost_remainder: u64,
    /// Retune generation stamped onto every batch accepted from now on
    /// (D-056) — bumped by [`SampleSink::meta_changed`] when the source's
    /// centre frequency moves.
    epoch: u64,
    /// Whether any batch has been offered yet. Before the first one there is
    /// no history to invalidate and nothing downstream to sequence against,
    /// so the opening metadata publishes immediately.
    accepted_any: bool,
}

impl SampleSink for RingSink {
    fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
        if self.shared.stop.load(Ordering::Relaxed) {
            return SinkFlow::Stop;
        }
        // One receipt instant for the whole block (blocks are ≤ 10 ms of
        // stream time): every batch resolved out of it — accepted or shed —
        // is attributed to this same instant on the same clock (D-028).
        let t = self.shared.now_s();
        let tag = BatchTag {
            t_receipt: t,
            epoch: self.epoch,
        };
        self.accepted_any = true;
        let (producer, ts) = (&mut self.producer, &mut self.ts_producer);
        // Time-paced sources shed a full ring — except when backpressure is
        // configured (a file source, D-125 Amendment 4 item 3): the compute
        // thread that drains this ring is the same thread that stalls in
        // `fold_run`'s frame-ring wait, so a shed here would silently drop
        // raw samples upstream of the very fix that made the frame ring
        // lossless. Backpressure means "the file falls behind, never loses
        // a frame" at *every* layer between it and analysis, not only the
        // last one (`Shared::wants_backpressure`'s own doc comment).
        let waits_for_room = !self.paced || self.shared.wants_backpressure();
        let outcome = if waits_for_room {
            // Wait for room — consumption is the pace. At shutdown the
            // in-flight batch is skipped unwritten and uncounted; the
            // pipeline is closing.
            let stop = &self.shared.stop;
            self.batcher.push(samples, |batch| loop {
                if try_write(producer, ts, batch, tag) {
                    return true;
                }
                if stop.load(Ordering::Relaxed) {
                    return true;
                }
                std::thread::sleep(IDLE_SLEEP);
            })
        } else {
            // Time-paced, no backpressure: a full ring sheds this whole
            // batch, counted below.
            self.batcher
                .push(samples, |batch| try_write(producer, ts, batch, tag))
        };
        if outcome.dropped > 0 {
            self.shared
                .health
                .lock()
                .unwrap()
                .record_dropped(t, outcome.dropped);
            // §7.6: the waterfall folds shed batches in as gap time, so the
            // interval they emptied renders as floor (D-051).
            self.shared
                .wf_dropped
                .fetch_add(outcome.dropped, Ordering::Relaxed);
        }
        if self.shared.stop.load(Ordering::Relaxed) {
            SinkFlow::Stop
        } else {
            SinkFlow::Continue
        }
    }

    /// The source announced new metadata in band (spec §5.3's segment
    /// boundary). Two cases, and the difference is D-056's whole point.
    ///
    /// * **The centre moved** — a retune. Everything the pipeline has
    ///   accumulated describes the *old* centre, and whole batches at the old
    ///   centre are still in the ring behind this announcement. Publishing
    ///   the new centre now would label that history with a frequency it was
    ///   never measured at. So the announcement is *queued against the next
    ///   epoch*: the compute thread publishes it, and wipes the
    ///   frequency-dependent accumulators, at the exact batch where the new
    ///   centre starts being true.
    ///
    ///   The partial batch the batcher is holding is discarded with the same
    ///   move: it straddles the retune, and an FFT over samples from two
    ///   different centres is not a spectrum of either. Fewer than one batch
    ///   of samples go, and **no counter is touched** — D-013 counts whole
    ///   batches, completed or shed, and a partial batch has always been
    ///   "neither processed, dropped, nor counted" (`Batcher`'s own
    ///   contract). The health window is left exactly as it was: a retune is
    ///   not a drop.
    /// * **Anything else** (the opening announcement, a provenance or label
    ///   change) publishes immediately, as it always did — there is no stale
    ///   history to sequence against.
    fn meta_changed(&mut self, meta: &SourceMeta) {
        let retuned = {
            let published = self.shared.meta.lock().unwrap();
            self.accepted_any && published.center_freq_hz != meta.center_freq_hz
        };
        if !retuned {
            *self.shared.meta.lock().unwrap() = meta.clone();
            return;
        }
        self.epoch += 1;
        if let Ok(fresh) = Batcher::new(self.batcher.batch_len()) {
            self.batcher = fresh;
        }
        self.shared
            .pending_meta
            .lock()
            .unwrap()
            .push((self.epoch, meta.clone()));
    }

    /// **D-060** — the tuner moved and the readback failed. There is no new
    /// centre to announce, which is precisely why `meta_changed` cannot
    /// carry this: its retune test is "did the centre change?", and the
    /// thing that failed was the act of finding out.
    ///
    /// Two consequences, both of them the honest reading of "the hardware is
    /// somewhere we cannot name":
    ///
    /// * **The published centre is withdrawn.** The metadata keeps its rate,
    ///   its label and its provenance — those are still true — and loses
    ///   `center_freq_hz`, so the axis falls back to the relative labelling
    ///   a centre-less source has always used (FR-C3). It never states a
    ///   frequency the radio has left.
    /// * **Every frequency-dependent accumulator is stale**, and unlike an
    ///   ordinary retune this reset cannot wait for the batch it becomes
    ///   true of: the stream is about to end with
    ///   [`SourceError::TuneUnverified`], so no such batch will arrive. The
    ///   epoch is published in `forced_epoch` and the compute thread applies
    ///   it on its next pass, with or without samples.
    ///
    /// The partial batch goes with it, for the same reason a retune's does:
    /// it straddles the move. No counter is touched — D-013 counts whole
    /// batches, and a retune, verified or not, is not a drop.
    ///
    /// [`SourceError::TuneUnverified`]: phosphene_sources::source::SourceError::TuneUnverified
    fn tune_unverified(&mut self) {
        self.epoch += 1;
        if let Ok(fresh) = Batcher::new(self.batcher.batch_len()) {
            self.batcher = fresh;
        }
        let unknown = {
            let published = self.shared.meta.lock().unwrap();
            SourceMeta {
                center_freq_hz: None,
                ..published.clone()
            }
        };
        self.shared
            .pending_meta
            .lock()
            .unwrap()
            .push((self.epoch, unknown));
        self.shared
            .forced_epoch
            .store(self.epoch, Ordering::Release);
    }

    /// D-048: samples the device reported as lost upstream are **received
    /// and dropped** — they enter the D-013 figures through the same
    /// [`PipelineHealth::record_dropped`] a locally shed batch uses, so
    /// `PROC %` falls and `DROPS` rises identically for both. The counter is
    /// batch-grained (D-013), so losses accrue in `lost_remainder` and
    /// convert as whole batches fill; attribution time is the report's
    /// receipt, the D-028 time base.
    fn device_lost(&mut self, samples: u64) {
        self.lost_remainder += samples;
        let batch_len = self.batcher.batch_len() as u64;
        let batches = self.lost_remainder / batch_len;
        if batches > 0 {
            self.lost_remainder -= batches * batch_len;
            let t = self.shared.now_s();
            self.shared
                .health
                .lock()
                .unwrap()
                .record_dropped(t, batches);
            // Device-reported losses are signal time with no data too: the
            // waterfall renders them as floor, like any counted drop.
            self.shared.wf_dropped.fetch_add(batches, Ordering::Relaxed);
        }
    }

    /// D-050: an overflow EVENT is counted and surfaced, never converted
    /// into batches — the D-013 figures stay unpolluted by invented
    /// quantities.
    fn device_overflow(&mut self) {
        self.shared.device_overflows.fetch_add(1, Ordering::Relaxed);
    }
}

/// Write one whole batch and its receipt stamp into the rings, or refuse
/// without writing anything — all-or-nothing, so the ring only ever holds
/// whole batches, each with exactly one stamp.
fn try_write(
    producer: &mut rtrb::Producer<Complex<f32>>,
    ts_producer: &mut rtrb::Producer<BatchTag>,
    batch: &[Complex<f32>],
    tag: BatchTag,
) -> bool {
    let Ok(mut chunk) = producer.write_chunk(batch.len()) else {
        return false;
    };
    // The stamp goes in before the samples commit, so a consumer that sees
    // the batch always finds its stamp. The stamp ring is sized to be
    // strictly roomier than the sample ring in whole batches, so this push
    // cannot fail; refusing on the impossible keeps the pairing intact
    // rather than desynchronising it.
    if ts_producer.push(tag).is_err() {
        return false;
    }
    let (a, b) = chunk.as_mut_slices();
    let split = a.len();
    a.copy_from_slice(&batch[..split]);
    b.copy_from_slice(&batch[split..]);
    chunk.commit_all();
    true
}

fn compute_loop(
    shared: Arc<Shared>,
    mut consumer: rtrb::Consumer<Complex<f32>>,
    mut ts_consumer: rtrb::Consumer<BatchTag>,
    mut analyzer: SpectrumAnalyzer,
    mut trace_state: TraceState,
    gate: Option<Arc<AtomicU64>>,
) {
    let n = analyzer.size();
    let mut batch = vec![Complex::new(0.0f32, 0.0); n];
    let mut spectra = vec![0.0f32; n * MAX_BATCHES_PER_DRAIN];
    let mut tags = [BatchTag {
        t_receipt: 0.0,
        epoch: 0,
    }; MAX_BATCHES_PER_DRAIN];
    // Scratch the retune reset drains the pending waterfall rows into —
    // emptying the ring in place rather than reallocating it (§7.7).
    let mut wf_discard: Vec<f32> = Vec::new();
    let mut epoch = 0u64;

    while !shared.stop.load(Ordering::Relaxed) {
        // Drain up to one §7.0 K's worth of whole batches.
        let mut k = 0;
        while k < MAX_BATCHES_PER_DRAIN {
            match consumer.read_chunk(n) {
                Ok(chunk) => {
                    if let Some(gate) = &gate {
                        // Test-only injected slow consumer (the D-028
                        // backlog seal): without a permit the batch stays in
                        // the ring — the chunk is dropped uncommitted.
                        let starved = gate
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                                v.checked_sub(1)
                            })
                            .is_err();
                        if starved {
                            break;
                        }
                    }
                    let (a, b) = chunk.as_slices();
                    batch[..a.len()].copy_from_slice(a);
                    batch[a.len()..].copy_from_slice(b);
                    chunk.commit_all();
                    // The paired receipt stamp: pushed before the samples
                    // commit, into a ring strictly roomier (in whole
                    // batches) than the sample ring, so a batch without its
                    // stamp is unreachable. If it ever happens anyway, be
                    // LOUD: falling back to the current clock would silently
                    // re-introduce the exact two-time-base skew D-028
                    // forbids, so that last resort panics in debug builds
                    // and announces itself in release ones.
                    tags[k] = match ts_consumer.pop() {
                        Ok(tag) => tag,
                        Err(_) => {
                            debug_assert!(
                                false,
                                "batch popped without its receipt stamp (D-028 pairing broken)"
                            );
                            eprintln!(
                                "phosphene: BUG: batch without receipt stamp — HUD window \
                                 attribution degraded to completion time (D-028)"
                            );
                            BatchTag {
                                t_receipt: shared.now_s(),
                                epoch,
                            }
                        }
                    };
                    analyzer.process(&batch, &mut spectra[k * n..(k + 1) * n]);
                    k += 1;
                }
                Err(_) => break,
            }
        }
        if k == 0 {
            // Nothing arrived — but a tune that could not be read back
            // (D-060) has no batch coming to carry its reset, so it is
            // applied here on an idle pass, and the wiped trace is published
            // so the display picks it up.
            if take_forced_retune(&shared, &mut epoch, &mut trace_state, &mut wf_discard) {
                let mut buf = shared.trace.lock().unwrap();
                buf.data.copy_from_slice(trace_state.trace());
                buf.seq += 1;
            }
            std::thread::sleep(IDLE_SLEEP);
            continue;
        }

        // These FFTs are now *completed* — record them (D-013: the samples
        // count as processed only once their FFT actually ran), each
        // attributed to its RECEIPT instant so numerator and denominator
        // share one time base (D-028).
        {
            let mut health = shared.health.lock().unwrap();
            for tag in &tags[..k] {
                health.record_completed(tag.t_receipt, 1);
            }
        }

        // D-123 (FL-5): test-only, deterministic reproduction of the
        // tau-decay flake's window — completion is now recorded, and this
        // drain's fold has not started. Compiles to nothing in a non-test
        // binary (`fl5_test_fold_delay` doc comment; symbol-count proof in
        // the FL-5 branch evidence).
        #[cfg(test)]
        fl5_test_fold_delay(&shared);

        // The display's K/L keys land in `shared.live_tau`; this thread owns
        // the EMA, so it adopts the value here, before folding the drain's
        // spectra — a no-op equality check when nothing changed (M1-J).
        trace_state.set_tau(*shared.live_tau.lock().unwrap());

        // §7.6: counted drops are gap time, folded in before this drain's
        // spectra — a shed batch is signal time with no data, and §7.6
        // renders it as floor instead of silently splicing the time axis.
        // (The interleaving of a drain's sheds and completes is
        // batch-approximate — display honesty, not sample accounting;
        // D-013's figures are untouched.)
        {
            let mut wf = shared.wf.lock().unwrap();
            let WfShared { agg, ring, dt } = &mut *wf;
            let shed = shared.wf_dropped.swap(0, Ordering::Relaxed);
            if shed > 0 {
                agg.advance(shed as f32 * *dt, |row| ring.push_row(row));
            }
        }

        // Fold the drain in runs of one retune epoch (D-056) — see
        // `fold_drain_runs` for why a run behind `epoch` is discarded rather
        // than treated as a boundary.
        fold_drain_runs(
            &shared,
            &mut epoch,
            &tags[..k],
            &spectra[..k * n],
            n,
            &mut trace_state,
            &mut wf_discard,
        );

        // D-060, applied after this drain has folded and before the trace is
        // published: whatever was in flight belongs to a centre the radio has
        // already left, so it is folded and then wiped in the same pass. The
        // display's next read sees the reset state, never old-centre history
        // under an axis that can no longer name a centre.
        take_forced_retune(&shared, &mut epoch, &mut trace_state, &mut wf_discard);

        let mut buf = shared.trace.lock().unwrap();
        buf.data.copy_from_slice(trace_state.trace());
        buf.seq += 1;
    }
}

/// Fold one drain's tagged spectra in runs of one retune epoch (D-056),
/// applying [`apply_retune`] at each **forward** boundary crossing.
///
/// In the common case — no retune in flight — this is exactly one run over
/// the whole drain, one extra integer comparison. When an ordinary retune
/// boundary falls inside the drain, the reset happens *between* the two
/// runs, on the thread that folds them: nothing at the old centre survives
/// it and nothing at the new centre is folded before it.
///
/// `*epoch` can also have been advanced **out of band**, by
/// `take_forced_retune` reacting to `forced_epoch` (D-060: a tune the
/// readback could not confirm resets on the compute thread's own next pass,
/// with no batch to carry the boundary). When that happens while the ring
/// still holds more pre-tune batches than one drain reads
/// (`MAX_BATCHES_PER_DRAIN`), a later call sees a straggler run tagged with
/// an epoch **behind** `*epoch` — not a new boundary, but one the forced
/// reset already crossed without it. Treating any mismatch as a boundary
/// (the pre-lane behaviour) regressed `*epoch` backward and folded that
/// stale, pre-tune run into the accumulators the reset had just cleared — a
/// pre-retune frame landing after the generation counter had already
/// announced the reset as done (the D-060 retune-flake, D-093). Only a
/// forward move is a boundary; a run behind `*epoch` is discarded instead,
/// never folded and never applied as a reset. Its FFTs were already counted
/// as completed by the caller, so D-013's figures are untouched either way.
///
/// See `tests::a_straggler_run_behind_the_tracker_is_discarded_not_folded`
/// for this exact situation, constructed directly and asserted
/// deterministically, with no ring or thread involved.
fn fold_drain_runs(
    shared: &Arc<Shared>,
    epoch: &mut u64,
    tags: &[BatchTag],
    spectra: &[f32],
    n: usize,
    trace_state: &mut TraceState,
    wf_discard: &mut Vec<f32>,
) {
    let k = tags.len();
    let mut i = 0;
    while i < k {
        let run_epoch = tags[i].epoch;
        let mut j = i + 1;
        while j < k && tags[j].epoch == run_epoch {
            j += 1;
        }
        if run_epoch < *epoch {
            i = j;
            continue;
        }
        if run_epoch > *epoch {
            apply_retune(shared, run_epoch, trace_state, wf_discard);
            *epoch = run_epoch;
        }
        fold_run(
            shared,
            &spectra[i * n..j * n],
            n,
            trace_state,
            shared.now_s(),
        );
        i = j;
    }
}

/// Fold one run of completed spectra — all under one retune epoch — into the
/// display accumulators.
///
/// Locks are strictly sequential (histo, then max hold, then waterfall, then
/// the trace — each released before the next), and the frame-side readers
/// take them one at a time too, so no lock order can deadlock. Batch
/// accounting is untouched: completion was recorded when the FFT ran (D-013).
///
/// `shared.batches_folded` is bumped once, at the very end, by however many
/// spectra this run folded — after every accumulator above has been written
/// and its lock released (D-123). The `Release` ordering on that bump pairs
/// with the `Acquire` load in [`Pipeline::batches_folded`], so a thread that
/// observes the counter reach a value is guaranteed to see every write this
/// call made, not just eventually — this is what makes waiting on it exact
/// rather than a race narrowed to "unlikely".
fn fold_run(
    shared: &Arc<Shared>,
    spectra: &[f32],
    n: usize,
    trace_state: &mut TraceState,
    t_s: f64,
) {
    // §7.4 (M1-J): the live trace's EMA, owned by this thread.
    trace_state.accumulate(spectra, t_s);

    // §7.3 per-spectrum accumulation (D-033: the producer M1-H wired).
    {
        let mut histo = shared.histo.lock().unwrap();
        for spectrum in spectra.chunks_exact(n) {
            histo.accumulate(spectrum);
        }
    }

    // §7.5 per-spectrum accumulation (FR-D3, D-034). The D-007 decay ticks
    // display-side (`max_hold_frame`) because it needs the live trace and the
    // display's own Δt; interleaving with this accumulation is safe — decay
    // only ever applies to the retained trace, so a fresh maximum is at most
    // one frame from being displayed undecayed (§7.5's ordering contract,
    // kept across the thread boundary).
    {
        let mut max_hold = shared.max_hold.lock().unwrap();
        for spectrum in spectra.chunks_exact(n) {
            max_hold.accumulate(spectrum);
        }
    }

    // §7.6 per-spectrum waterfall aggregation (D-051: the feed consumes the
    // spectra stream, not one sample per display frame — this is what makes
    // `Max` mean what the spec says).
    {
        let mut wf = shared.wf.lock().unwrap();
        let WfShared { agg, ring, dt } = &mut *wf;
        for spectrum in spectra.chunks_exact(n) {
            agg.push_spectrum(spectrum, *dt, |row| ring.push_row(row));
        }
    }

    // AI-1's SpectrumTap backing store (nextgen spec §2.2): fed beside the
    // three accumulators above, same locking discipline.
    //
    // D-125 Amendment 4 item 3, the owner's ruling: "if reading from a live
    // source, shed the count; if reading from a file — lag behind." For a
    // file-backed pipeline (`shared.backpressure`) this waits, per frame,
    // until the analysis worker has consumed enough of `frame_ring`'s fixed
    // capacity that this push cannot silently overwrite one it has not yet
    // accounted for — the reader (this compute thread) waits for the
    // analyser, exactly as the brief asks, entirely within this crate: no
    // change to `phosphene-sources`. `frame_ring`'s own capacity
    // (`TAP_HISTORY_FRAMES`) never grows — this only ever delays a push, it
    // never queues one anywhere else, so memory stays bounded regardless of
    // how far behind the file falls. Only waits while analysis is actually
    // active: an inactive worker never advances `frames_analysed`, and
    // waiting on a count nothing is updating would hang the pipeline.
    #[cfg(feature = "analyze")]
    {
        let mut ring = shared.frame_ring.lock().unwrap();
        for spectrum in spectra.chunks_exact(n) {
            if shared.backpressure && shared.analysis.is_active() {
                drop(ring);
                loop {
                    let pushed = shared.frames_pushed_to_ring.load(Ordering::Relaxed);
                    let analysed = shared.analysis.frames_analysed();
                    if pushed < analysed + crate::analyze::TAP_HISTORY_FRAMES as u64 {
                        break;
                    }
                    if shared.stop.load(Ordering::Relaxed) || !shared.analysis.is_active() {
                        break;
                    }
                    std::thread::sleep(IDLE_SLEEP);
                }
                ring = shared.frame_ring.lock().unwrap();
            }
            ring.push(spectrum);
            shared.frames_pushed_to_ring.fetch_add(1, Ordering::Relaxed);
        }
    }

    // D-123 (FL-5): the fold is done — every accumulator above has this
    // run's spectra and every lock that guarded them has been released.
    // Bump last, with `Release`, so `Pipeline::batches_folded`'s `Acquire`
    // load can only observe this once it is actually true.
    shared
        .batches_folded
        .fetch_add((spectra.len() / n) as u64, Ordering::Release);
}

/// D-123 (FL-5): a deterministic, test-only reproduction of the tau-decay
/// flake's exact window. Called from `compute_loop`, between
/// `record_completed` and the fold — when `shared.test_fold_delay` is set,
/// the compute thread pauses here long enough that a caller which has just
/// observed `health().batches_completed` reach a count (the proxy D-094
/// forbids waiting on) is guaranteed to read the display accumulators
/// before this drain's `fold_run` has touched them, on every run, on any
/// machine. With `test_fold_delay` unset (the default for every pipeline
/// that does not opt in) this is one relaxed atomic load and a branch not
/// taken.
///
/// No natural reproduction was seen on a Linux lane-class machine at this
/// lane's branch point: 0 failures in 200 runs of
/// `persistence_tau_control_changes_the_phosphor_decay` back to back, 0 in
/// 200 with 8 instances running at once, and 32 of 32 passes with 32 at once
/// under added CPU load. The window D-123 names is microseconds wide, which
/// is why this delay exists instead of a load recipe.
///
/// `#[inline(never)]`: a function this small, whose only call site sits
/// behind a branch that is false in every test but the one that opts in
/// (`persistence_tau_control_changes_the_phosphor_decay`), is exactly what
/// LLVM inlines away given the chance — folding its symbol into
/// `compute_loop`'s and turning the FL-5 seal's symbol-count proof (a
/// marker present in the test binary, absent from the release one) into an
/// unfalsifiable 0-against-0 instead of 0-against-1 (the federator's
/// advisory, 2026-09-15). [`FL5_TEST_FOLD_DELAY_MARKER`] anchors the same
/// symbol a second way, independent of the call site surviving at all.
///
/// **Proof, run in this branch:**
/// ```text
/// $ cargo build -p phosphene-app --features analyze
/// $ nm -C target/debug/phosphene | grep -c fl5_test_fold_delay
/// 0
/// $ cargo test -p phosphene-app --features analyze --no-run
/// $ nm -C target/debug/deps/phosphene-<hash> | grep -c fl5_test_fold_delay
/// 1
/// $ nm -C target/debug/deps/phosphene-<hash> | grep fl5_test_fold_delay
/// 0000000000a553b0 t phosphene::pipeline::fl5_test_fold_delay
/// 0000000003b356d0 d phosphene::pipeline::FL5_TEST_FOLD_DELAY_MARKER
/// ```
/// 0 against 1, never 0 against 0: the symbol exists, and only in the test
/// binary.
#[cfg(test)]
#[inline(never)]
fn fl5_test_fold_delay(shared: &Shared) {
    if shared.test_fold_delay.load(Ordering::Relaxed) {
        std::thread::sleep(FL5_TEST_FOLD_DELAY_DURATION);
    }
}

/// How long [`fl5_test_fold_delay`] pauses the compute thread when armed —
/// large against the microsecond race window D-123 describes, and against
/// this crate's lock/atomic overhead, so the reproduction is not itself a
/// coin flip.
#[cfg(test)]
const FL5_TEST_FOLD_DELAY_DURATION: Duration = Duration::from_millis(200);

/// Keeps [`fl5_test_fold_delay`] reachable, and therefore its symbol
/// present in the test binary, independent of whether its one call site in
/// `compute_loop` survives optimisation — see that function's doc comment.
#[cfg(test)]
#[used]
static FL5_TEST_FOLD_DELAY_MARKER: fn(&Shared) = fl5_test_fold_delay;

/// Apply a pending **unverified-tune** reset (**D-060**) if one is due, and
/// say whether it fired.
///
/// Unlike the epoch boundary the drain loop crosses, this reset is not
/// carried by any batch: after a tune the device would not confirm, no batch
/// under the new epoch will ever arrive. The accumulators are stale all the
/// same — the hardware moved — so the compute thread, which owns them, wipes
/// them on its own next pass. The same [`apply_retune`] does the work, so
/// there is one reset in the program and not two.
fn take_forced_retune(
    shared: &Arc<Shared>,
    epoch: &mut u64,
    trace_state: &mut TraceState,
    wf_discard: &mut Vec<f32>,
) -> bool {
    let forced = shared.forced_epoch.load(Ordering::Acquire);
    if forced <= *epoch {
        return false;
    }
    apply_retune(shared, forced, trace_state, wf_discard);
    *epoch = forced;
    true
}

/// **D-056's reset**, applied at the batch where a retune starts being true.
///
/// After a retune the persistence histogram, the max-hold trace, the live
/// trace and the waterfall all hold history at the **old** centre frequency.
/// Drawing that against the new axis is a straightforward lie — the same
/// class as labelling a rate-less axis in Hz — and it would look entirely
/// plausible. So the stale history is discarded rather than redrawn under a
/// new label, and the new metadata is published in the same move: the label
/// and the data change together.
///
/// What is reset, and how, without a single accumulator learning what a
/// retune is:
///
/// * **Persistence** (§7.3) is rebuilt at its own bins/levels/dB window,
///   carrying the τ values the `,`/`.` keys have set — a reset must not
///   silently undo a user's controls. This is the same rebuild
///   [`Pipeline::intensity_frame`] already performs on a dB-window change,
///   for the same reason: history quantised against one setting cannot be
///   honestly redrawn against another.
/// * **Max hold** (§7.5) resets to its `DBFS_FLOOR` baseline and re-seeds
///   from the next spectrum, keeping its mode and τ.
/// * **The live trace** (§7.4) is re-seeded too. D-056 names three
///   accumulators; the EMA is a fourth by the same argument — it is a
///   frequency-dependent average, and one still blending old-centre spectra
///   into a new-centre trace is the same false display in miniature.
/// * **The waterfall** (§7.6): the aggregator drops its half-filled row and
///   the pending-row ring is drained into scratch (no reallocation). Rows the
///   display has *already* taken live in the render crate's ring texture,
///   which the pipeline cannot reach — [`Pipeline::retune_generation`] is the
///   signal the frame loop consumes to clear those.
///
/// The health accounting is deliberately not mentioned above, because it is
/// deliberately not touched: **a retune is not a drop** (D-013), and nothing
/// here records one.
fn apply_retune(
    shared: &Arc<Shared>,
    epoch: u64,
    trace_state: &mut TraceState,
    wf_discard: &mut Vec<f32>,
) {
    // Publish the metadata this epoch was announced with — the device
    // readback (C5), reaching the axis at the first sample it describes.
    {
        let mut pending = shared.pending_meta.lock().unwrap();
        if let Some(idx) = pending.iter().rposition(|(e, _)| *e <= epoch) {
            let meta = pending[idx].1.clone();
            pending.drain(..=idx);
            *shared.meta.lock().unwrap() = meta;
        }
    }

    {
        let mut histo = shared.histo.lock().unwrap();
        if let Ok(mut fresh) = PersistenceHistogram::new(
            histo.bins(),
            histo.levels(),
            histo.p_bottom(),
            histo.p_top(),
        ) {
            let _ = fresh.set_tau_rise(histo.tau_rise());
            let _ = fresh.set_tau_decay(histo.tau_decay());
            *histo = fresh;
        }
    }
    shared.max_hold.lock().unwrap().reset();
    trace_state.reset();
    {
        let mut wf = shared.wf.lock().unwrap();
        let WfShared { agg, ring, .. } = &mut *wf;
        // Re-applying the cadence the aggregator is already on is a no-op, so
        // the open row is dropped by rebuilding at that same cadence: the
        // half-filled row spans the retune and belongs to neither centre.
        *agg = RowAggregator::new(ring.bins(), agg.mode(), agg.cadence());
        ring.drain_into(wf_discard);
        wf_discard.clear();
    }
    // D-125 Amendment 4 item 3: reset alongside the accumulators above, on
    // this same (compute) thread, so it stays on the same baseline as the
    // analysis worker's own `frames_analysed`/`frames_received` reset (both
    // triggered by the same retune-gen bump, just observed on different
    // threads) — `fold_run`'s backpressure compares the two directly, and a
    // stale, un-reset `frames_pushed_to_ring` well ahead of a freshly-reset
    // `frames_analysed` would misread a normal retune as an analysis
    // backlog and wait for one that does not exist.
    #[cfg(feature = "analyze")]
    shared.frames_pushed_to_ring.store(0, Ordering::Relaxed);
    // Release: every reset above — histo, max hold, the live trace, the
    // waterfall — and the meta publish are all sequenced before this store in
    // program order, but each is guarded by its own, separate lock (there is
    // no single lock spanning all of D-056's accumulators plus the counter).
    // Relaxed would publish this counter as a channel of its own, with no
    // ordering relative to those other locks' releases — a reader could
    // observe the bump on one core before the resets it is meant to
    // announce become visible on that core, which is exactly "the new
    // generation together with pre-retune state" D-060/D-056 forbid. Paired
    // with the Acquire load in `Pipeline::retune_generation`, this store
    // makes every prior write here happen-before any read that observes it.
    shared.retune_gen.fetch_add(1, Ordering::Release);
}

/// The §7.4 live trace, with its per-spectrum interval either fixed from the
/// declared rate (N / rate) or — for rate-less sources, where inventing a
/// rate is forbidden (C5) — measured from the actual FFT cadence and applied
/// on the rare occasions it shifts by more than 2×.
struct TraceState {
    live: LiveTrace,
    fft_size: usize,
    /// Per-spectrum interval currently baked into `live`, seconds — the one
    /// this state knows, so a rebuild (cadence drift, or D-056's retune
    /// reset) can restore it without the accumulator having to hand it back.
    interval_s: f32,
    adapt: Option<Adapt>,
}

struct Adapt {
    /// Interval currently baked into `live`, seconds.
    interval_s: f32,
    window_start_t: f64,
    ffts: u64,
}

/// Starting per-spectrum interval before any cadence has been measured.
const ADAPT_INITIAL_INTERVAL_S: f32 = 1e-3;
/// Re-measure period, seconds.
const ADAPT_PERIOD_S: f64 = 0.5;

impl TraceState {
    fn new(fft_size: usize, sample_rate_hz: Option<f64>) -> Result<Self, String> {
        let (interval, adapt) = match sample_rate_hz {
            Some(rate) => ((fft_size as f64 / rate) as f32, None),
            None => (
                ADAPT_INITIAL_INTERVAL_S,
                Some(Adapt {
                    interval_s: ADAPT_INITIAL_INTERVAL_S,
                    window_start_t: 0.0,
                    ffts: 0,
                }),
            ),
        };
        let live = LiveTrace::new(fft_size, interval, DEFAULT_TAU_LIVE)
            .map_err(|e| format!("cannot build the live trace: {e}"))?;
        Ok(TraceState {
            live,
            fft_size,
            interval_s: interval,
            adapt,
        })
    }

    /// Adopt the display's τ_live (FR-D2, M1-J). A no-op unless the value
    /// changed; the trace carries over — only the smoothing of future
    /// spectra changes. The caller (the compute loop) is the only writer of
    /// the EMA, so the τ can never race the accumulation it governs.
    fn set_tau(&mut self, tau: f32) {
        if tau != self.live.tau() {
            // `clamp_tau` bounds every value that can reach here, which
            // precludes the non-finite/non-positive inputs set_tau rejects.
            let _ = self.live.set_tau(tau);
        }
    }

    /// Re-seed the EMA from the next spectrum, keeping the cadence and the
    /// τ the display has set (D-056's reset — the averaging-time control
    /// must survive a retune, only the *history* is discarded). Same rebuild
    /// the cadence drift below performs, and the same reason it is affordable:
    /// a reconfiguration, not a per-batch cost.
    fn reset(&mut self) {
        if let Ok(live) = LiveTrace::new(self.fft_size, self.interval_s, self.live.tau()) {
            self.live = live;
        }
    }

    fn accumulate(&mut self, spectra: &[f32], t_s: f64) {
        self.live.accumulate_batch(spectra);
        let Some(adapt) = &mut self.adapt else {
            return;
        };
        adapt.ffts += (spectra.len() / self.fft_size) as u64;
        let elapsed = t_s - adapt.window_start_t;
        if elapsed < ADAPT_PERIOD_S {
            return;
        }
        if adapt.ffts > 0 {
            let measured = (elapsed / adapt.ffts as f64) as f32;
            let drifted = measured.is_finite()
                && measured > 0.0
                && (measured > 2.0 * adapt.interval_s || measured < 0.5 * adapt.interval_s);
            if drifted {
                // Reconfiguration, not the hot path: rebuild the EMA at the
                // measured cadence (it re-seeds from the next spectrum),
                // carrying the τ the display has set (M1-J) — a cadence
                // rebuild must not silently reset the averaging-time control.
                if let Ok(live) = LiveTrace::new(self.fft_size, measured, self.live.tau()) {
                    self.live = live;
                    self.interval_s = measured;
                    adapt.interval_s = measured;
                }
            }
        }
        adapt.window_start_t = t_s;
        adapt.ffts = 0;
    }

    fn trace(&self) -> &[f32] {
        self.live.trace()
    }
}

/// Test fixture shared with the windowed seals (`crate::window`): a
/// deterministic tone-then-silence cf32 capture — `tone_batches` batches of
/// an on-bin tone at −20 dBFS (raw bin N/4), then `silent_batches` of
/// silence. The decaying-tone shape FR-D3 exists to display: the live trace
/// falls away while the max hold retains the peak. Phase repeats every 4
/// samples (rel 0.25), so it is exact in f32.
/// Test fixture shared with the windowed seals (`crate::window`): a cf32
/// capture of `groups × per_group` batches in which exactly one batch per
/// group — index `burst_idx` within the group — carries the on-bin −20 dBFS
/// tone (raw bin N/4) and every other batch is silence. The shape the D-051
/// transient seal needs: at `per_group` spectra per row, a feed that samples
/// one spectrum per frame (or per interval boundary) misses the burst; a
/// feed that folds every spectrum cannot.
#[cfg(test)]
pub(crate) fn burst_capture_bytes(
    n: usize,
    groups: usize,
    per_group: usize,
    burst_idx: usize,
) -> Vec<u8> {
    assert!(burst_idx < per_group);
    let mut bytes = Vec::with_capacity(groups * per_group * n * 8);
    for _ in 0..groups {
        for b in 0..per_group {
            if b == burst_idx {
                for i in 0..n {
                    let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
                    let (im, re) = phase.sin_cos();
                    bytes.extend_from_slice(&(re * 0.1).to_le_bytes());
                    bytes.extend_from_slice(&(im * 0.1).to_le_bytes());
                }
            } else {
                bytes.resize(bytes.len() + n * 8, 0);
            }
        }
    }
    bytes
}

#[cfg(test)]
pub(crate) fn tone_then_silence_bytes(
    n: usize,
    tone_batches: usize,
    silent_batches: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((tone_batches + silent_batches) * n * 8);
    for i in 0..tone_batches * n {
        let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
        let (im, re) = phase.sin_cos();
        bytes.extend_from_slice(&(re * 0.1).to_le_bytes());
        bytes.extend_from_slice(&(im * 0.1).to_le_bytes());
    }
    bytes.resize(bytes.len() + silent_batches * n * 8, 0);
    bytes
}

/// D-123 / D-122: the FL-5 seal, run in full against this file's committed
/// tree (378c2ad — "FL-5 (D-123): wait on the fold itself, not on
/// batches_completed"), never only reported in the PR description.
///
/// `SOAPY_SDR_ROOT=/nonexistent/phosphene-no-soapy
/// SOAPY_SDR_PLUGIN_PATH=/nonexistent/phosphene-no-soapy`, then:
/// * `cargo fmt --all -- --check` — clean.
/// * `cargo clippy --workspace --all-targets -- -D warnings` — clean.
/// * `cargo clippy -p phosphene-analyze --all-targets --features analyze -- -D warnings` — clean.
/// * `cargo clippy -p phosphene-app --all-targets --features analyze -- -D warnings` — clean.
/// * `cargo test --workspace` — 0 failed across every counted binary.
/// * `cargo test -p phosphene-analyze --features analyze` — 135 + 4 passed, 0 failed.
/// * `cargo test -p phosphene-app --features analyze` — 90 passed (2 ignored) + 10 passed (1 ignored) + 5 + 1 + 1 + 2 + 6 passed, 0 failed anywhere.
/// * `cargo deny check licenses` — `licenses ok`.
///
/// Five consecutive runs of `persistence_tau_control_changes_the_phosphor_
/// decay` alone, delay armed, all green — see that test's own doc comment
/// for the quoted red run this replaced.
#[cfg(test)]
mod tests {
    use super::*;
    use phosphene_core::RING_HEADROOM_S;
    use phosphene_sources::{
        Control, ControlCaps, FileSource, FileSourceConfig, Input, IqFormat, SigGen, SigGenConfig,
        SourceError,
    };

    /// NFR-P3 arithmetic at the app layer: the ring the pipeline would build
    /// holds ≥ 250 ms at the configured rate — computed from that rate, not
    /// from a constant that matches today's default.
    #[test]
    fn ring_capacity_holds_250ms_from_the_configured_rate() {
        for rate in [250_000.0, 2_048_000.0, 20e6, 61.44e6] {
            let cap = ring_capacity_samples(Some(rate), 1024).unwrap();
            assert!(
                cap as f64 / rate >= RING_HEADROOM_S,
                "ring of {cap} samples holds under 250 ms at {rate} Hz"
            );
        }
        // Tiny rates still get room for whole batches.
        assert!(ring_capacity_samples(Some(8.0), 4096).unwrap() >= 8 * 4096);
        // Rate-less sources get the fixed preallocation.
        assert_eq!(
            ring_capacity_samples(None, 1024).unwrap(),
            UNRATED_RING_SAMPLES
        );
        assert!(ring_capacity_samples(Some(f64::NAN), 1024).is_err());
    }

    /// A hang detector, never a race against CPU contention (fix pass 2/3):
    /// this machine has been directly measured to stall a whole process —
    /// every thread in it, worker and test alike — for over a second under
    /// **no other load at all** (see `analyze_goldens.rs`'s and
    /// `render_look_review`'s module docs). Under the real parallel test
    /// load `cargo test` itself imposes, `live_analysis_worker_detects_a_
    /// strong_bursty_signal` timed out at the old 20 s bound in this same
    /// pass, purely from that contention, then passed immediately in
    /// isolation with the exact same code. 90 s is chosen to stay a hang
    /// detector (a real deadlock or an actually-broken worker still fails
    /// this in well under a minute) while giving real, observed multi-
    /// second host stalls generous headroom to clear.
    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(90);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// D-125 Amendment 4 item 3's file-source negative controls: run
    /// through the real two-thread pipeline plus the real analysis worker
    /// (not a direct `fold_run` unit test) — a pull-paced `SigGen` (produces
    /// exactly as fast as the compute thread accepts, the same shape an
    /// unthrottled file replay has) at a small FFT size, under `backpressure:
    /// true`, with the analysis worker made active so `frames_analysed`
    /// genuinely advances every real `CYCLE` (250ms). These tests take real
    /// wall-clock time — a handful of analysis cycles' worth.
    #[cfg(feature = "analyze")]
    fn start_backpressure_test_pipeline(n: usize) -> Pipeline {
        let source = Box::new(SigGen::new(SigGenConfig::new(8_000.0)).unwrap());
        let pipeline = Pipeline::start(
            source,
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: true,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .expect("start the backpressure test pipeline");
        pipeline.set_analyze_active(true);
        pipeline
    }

    /// The first negative control: "a file source over the limit must
    /// analyse every frame" — no frame is ever permanently skipped. Over
    /// several real analysis cycles, `frames_analysed` must reach a total
    /// well past one `frame_ring` depth (`TAP_HISTORY_FRAMES`), which is
    /// impossible if the ring were silently overwriting frames faster than
    /// analysis could ever catch up — exactly what a *live* source's own
    /// honest-shed path does on purpose (the sibling test in `analyze.rs`,
    /// `a_sequence_gap_is_counted_honestly_not_silently_absorbed`, is that
    /// path's own negative control).
    #[cfg(feature = "analyze")]
    #[test]
    fn file_source_backpressure_analyses_every_frame_no_skips() {
        let mut pipeline = start_backpressure_test_pipeline(512);
        let target = crate::analyze::TAP_HISTORY_FRAMES as u64 * 3;
        wait_until("backpressure to analyse past 3 ring depths", || {
            pipeline.analysis_coverage().0 >= target
        });
        // `pushed` keeps moving (the source never stops), so the honest
        // "every frame gets analysed" check is: analysis must go on to
        // reach whatever `pushed` was at *some* instant, not that the two
        // ever line up in the same snapshot — which they provably cannot
        // for a still-running producer.
        let pushed_snapshot = pipeline.frames_pushed_to_ring();
        wait_until("analysis to catch up to a pushed-count snapshot", || {
            pipeline.analysis_coverage().0 >= pushed_snapshot
        });
        let (analysed, received) = pipeline.analysis_coverage();
        assert_eq!(
            analysed, received,
            "a file-backed pipeline must never shed: analysed must equal received"
        );
        assert!(
            analysed >= pushed_snapshot,
            "analysis never caught up to a frame count it had already pushed \
             — a skipped frame would keep this permanently short"
        );
        pipeline.shutdown();
    }

    /// The second negative control: "bounded memory" / "never an unbounded
    /// queue" — `frame_ring`'s own fixed capacity means it cannot literally
    /// grow, but a broken backpressure implementation could still let the
    /// compute thread run arbitrarily far ahead of analysis by never
    /// waiting; this samples the live gap repeatedly through a real run and
    /// asserts it never exceeds one ring's worth.
    #[cfg(feature = "analyze")]
    #[test]
    fn file_source_backpressure_never_runs_more_than_one_ring_ahead_of_analysis() {
        let mut pipeline = start_backpressure_test_pipeline(512);
        let cap = crate::analyze::TAP_HISTORY_FRAMES as u64;
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut samples = 0u32;
        while Instant::now() < deadline {
            let pushed = pipeline.frames_pushed_to_ring();
            let analysed = pipeline.analysis_coverage().0;
            assert!(
                pushed <= analysed + cap,
                "frame_ring ran {} frames ahead of analysis, more than its \
                 own {cap}-frame capacity — an unbounded queue, exactly what \
                 item 3 forbids",
                pushed - analysed
            );
            samples += 1;
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            samples > 10,
            "the sampling loop itself did not run enough to mean anything"
        );
        pipeline.shutdown();
    }

    /// The two tests above only ever exercised a PULL-PACED source
    /// (`start_backpressure_test_pipeline`'s `paced: false`) — which already
    /// waits for sample-ring room on its own, with or without
    /// `backpressure`, so they could not have caught a `backpressure`-blind
    /// [`RingSink`]. A real rated file replay is `paced: true` (D-125
    /// Amendment 4 item 3's own module doc comment): its sample ring sheds
    /// a full ring **by default**, and only `backpressure` overrides that
    /// to "wait" (`Shared::wants_backpressure`'s doc comment; review 1's
    /// fix). Same shape as `start_backpressure_test_pipeline`, `paced:
    /// true` instead, at the same declared rate — `ring_capacity_samples`
    /// sizes the sample ring to exactly 8 whole batches at 8 kHz/n=512
    /// (the D-028 backlog seal test's own arithmetic), small enough that
    /// an unfixed `RingSink` would shed almost immediately once the
    /// frame-ring backpressure below stalls the compute thread that drains
    /// it.
    #[cfg(feature = "analyze")]
    fn start_paced_backpressure_test_pipeline(n: usize) -> Pipeline {
        let source = Box::new(SigGen::new(SigGenConfig::new(8_000.0)).unwrap());
        let pipeline = Pipeline::start(
            source,
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: true,
                backpressure: true,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .expect("start the paced backpressure test pipeline");
        pipeline.set_analyze_active(true);
        pipeline
    }

    /// D-125 Amendment 4 item 3, fix round (review 1): "a rated file replay
    /// is `paced: true`, so while backpressure stalls the compute thread
    /// the upstream sample ring sheds whole batches before they become
    /// frames." In the real shape (`paced: true` **and** `backpressure:
    /// true` together) zero batches may ever be shed, and every frame
    /// reaching `frame_ring` must be analysed — the same two guarantees
    /// `file_source_backpressure_analyses_every_frame_no_skips` already
    /// proves for a pull-paced source, now proven for a time-paced one,
    /// where a `paced`-only-governed sink would have shed instead (RED on
    /// the pre-fix `RingSink::push`, which read only `self.paced`).
    #[cfg(feature = "analyze")]
    #[test]
    fn file_source_backpressure_sheds_no_batches_when_paced() {
        let mut pipeline = start_paced_backpressure_test_pipeline(512);
        let target = crate::analyze::TAP_HISTORY_FRAMES as u64 * 3;
        wait_until("backpressure to analyse past 3 ring depths", || {
            pipeline.analysis_coverage().0 >= target
        });
        let pushed_snapshot = pipeline.frames_pushed_to_ring();
        wait_until("analysis to catch up to a pushed-count snapshot", || {
            pipeline.analysis_coverage().0 >= pushed_snapshot
        });
        let (analysed, received) = pipeline.analysis_coverage();
        assert_eq!(
            analysed, received,
            "a file-backed pipeline must never shed: analysed must equal received"
        );
        assert!(
            analysed >= pushed_snapshot,
            "analysis never caught up to a frame count it had already pushed \
             — a skipped frame would keep this permanently short"
        );
        assert_eq!(
            pipeline.health().dropped_batches,
            0,
            "a paced sink under backpressure sheds a raw sample batch before \
             it ever became a frame — 'every frame analysed' is meaningless \
             if frames never got made in the first place"
        );
        pipeline.shutdown();
    }

    /// A minimal `Shared`, built the same way `Pipeline::start_inner` does,
    /// but with no ring, no thread and no source — so `fold_drain_runs` can
    /// be exercised directly and deterministically (D-095), against the
    /// exact accumulators and `apply_retune`/`fold_run` the real pipeline
    /// uses, with none of the timing a live ring or thread would add.
    fn test_shared(n: usize) -> Arc<Shared> {
        let spectrum_interval_s = spectrum_interval_s(Some(1_000.0), n);
        let cadence = derive_cadence(
            WaterfallMode::default(),
            spectrum_interval_s,
            30.0,
            phosphene_render::FAST_INTERVAL_DEFAULT_S,
            phosphene_render::RING_ROWS as f32,
            1,
        );
        let wf = WfShared {
            agg: RowAggregator::new(n, RowAggregation::default(), cadence),
            ring: RowRing::new(n, wf_pending_rows(n)),
            dt: wf_dt(cadence, spectrum_interval_s),
        };
        Arc::new(Shared {
            start: Instant::now(),
            stop: AtomicBool::new(false),
            health: Mutex::new(PipelineHealth::new(n, HEALTH_WINDOW_S).unwrap()),
            trace: Mutex::new(TraceBuf {
                data: vec![DBFS_FLOOR; n],
                seq: 0,
            }),
            histo: Mutex::new(PersistenceHistogram::new(n, DEFAULT_LEVELS, -110.0, 0.0).unwrap()),
            max_hold: Mutex::new(MaxHold::new(n).unwrap()),
            live_tau: Mutex::new(DEFAULT_TAU_LIVE),
            meta: Mutex::new(SourceMeta {
                sample_rate_hz: Some(1_000.0),
                center_freq_hz: Some(100e6),
                label: "test".into(),
                provenance: "test".into(),
            }),
            source_done: Mutex::new(None),
            device_overflows: AtomicU64::new(0),
            wf: Mutex::new(wf),
            wf_dropped: AtomicU64::new(0),
            pending_meta: Mutex::new(Vec::new()),
            retune_gen: AtomicU64::new(0),
            forced_epoch: AtomicU64::new(0),
            #[cfg(feature = "analyze")]
            frame_ring: Mutex::new(FrameRing::new(n, crate::analyze::TAP_HISTORY_FRAMES)),
            #[cfg(feature = "analyze")]
            analysis: crate::analyze::AnalysisHandle::new(),
            #[cfg(feature = "analyze")]
            backpressure: false,
            #[cfg(feature = "analyze")]
            frames_pushed_to_ring: AtomicU64::new(0),
            batches_folded: AtomicU64::new(0),
            #[cfg(test)]
            test_fold_delay: AtomicBool::new(false),
        })
    }

    /// **D-095's deterministic negative control for the D-060 retune-flake
    /// (D-093).** Constructs the exact situation directly — a backlog
    /// deeper than one drain pass (`MAX_BATCHES_PER_DRAIN`), with the epoch
    /// tracker already ahead, as `take_forced_retune` leaves it after a
    /// forced reset with no batch to carry the boundary — and asserts that
    /// a run tagged behind the tracker is discarded: never folded into the
    /// accumulators, and never applied as a (regressing) reset.
    ///
    /// No ring, no thread, no timing: this calls `fold_drain_runs` directly,
    /// so it fails on every run, on any machine, unloaded, whenever the
    /// straggler-discard fix is absent — unlike the loaded-harness evidence
    /// in the PR, which needs pinned CPU contention to reproduce at all.
    #[test]
    fn a_straggler_run_behind_the_tracker_is_discarded_not_folded() {
        const TONE_DBFS: f32 = -20.0;
        let n = 4;
        let shared = test_shared(n);
        let mut trace_state = TraceState::new(n, Some(1_000.0)).unwrap();
        let mut wf_discard = Vec::new();

        // Seed the accumulators with a recognisable pre-tune spectrum,
        // exactly as real folded batches would before a retune — max hold's
        // peak is what the reset must clear and what a wrongly-folded
        // straggler would bring back.
        let hot = vec![TONE_DBFS; n];
        fold_run(&shared, &hot, n, &mut trace_state, 0.0);
        assert!(
            shared
                .max_hold
                .lock()
                .unwrap()
                .trace()
                .iter()
                .all(|&v| v == TONE_DBFS),
            "test setup: max hold must hold the pre-tune tone before the reset"
        );

        // The forced reset (D-060) has already run and left the tracker at
        // epoch 1 — exactly `take_forced_retune`'s post-condition, reached
        // here directly rather than through `forced_epoch`/a live ring.
        let mut epoch = 1u64;
        apply_retune(&shared, epoch, &mut trace_state, &mut wf_discard);
        assert!(
            shared
                .max_hold
                .lock()
                .unwrap()
                .trace()
                .iter()
                .all(|&v| v <= DBFS_FLOOR + 1.0),
            "test setup: the reset must clear max hold"
        );

        // More stragglers than one drain pass reads, all tagged epoch 0 —
        // behind the tracker — each carrying the pre-tune tone.
        let backlog = MAX_BATCHES_PER_DRAIN * 2;
        let tags = vec![
            BatchTag {
                t_receipt: 0.0,
                epoch: 0,
            };
            backlog
        ];
        let spectra = vec![TONE_DBFS; n * backlog];

        fold_drain_runs(
            &shared,
            &mut epoch,
            &tags,
            &spectra,
            n,
            &mut trace_state,
            &mut wf_discard,
        );

        assert_eq!(
            epoch, 1,
            "a run behind the tracker must never be treated as a boundary — \
             the tracker regressed to {epoch}"
        );
        let peak = shared
            .max_hold
            .lock()
            .unwrap()
            .trace()
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            peak <= DBFS_FLOOR + 1.0,
            "a straggler run behind the tracker was folded in: max hold \
             holds {peak} dBFS after the reset"
        );
    }

    /// AI-1: `Pipeline::spectrum_tap` serves the same dBFS/bin power frames
    /// the display trace is built from — real ones, from the running
    /// pipeline, not a stand-in.
    ///
    /// D-123 (FL-5) class check: this waits on `batches_completed` (32),
    /// then reads `frame_ring` (fed by `fold_run`, same as the τ-decay
    /// test's histogram) once, with a loose assertion (at least one frame,
    /// peak in a wide range). `MAX_BATCHES_PER_DRAIN` is 16, so reaching a
    /// count of 32 crosses at least one full drain boundary: the *previous*
    /// drain's fold is always complete by then, only the newest (≤ 16
    /// batches) can still be pending — which still leaves the ring
    /// non-empty. Cannot race in a way this assertion can see.
    #[cfg(feature = "analyze")]
    #[test]
    fn spectrum_tap_serves_the_pipelines_own_power_frames() {
        use phosphene_analyze::SpectrumTap;

        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: 1024,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        wait_until("the first published batches", || {
            pipeline.health().batches_completed >= 32
        });

        let tap = pipeline.spectrum_tap();
        let meta = tap.meta();
        assert_eq!(meta.bins, 1024);
        assert!(meta.bin_hz > 0.0 && meta.frame_dt_s > 0.0);

        let mut view = phosphene_analyze::FrameView::new(1024, 32);
        tap.latest_frames(&mut view);
        assert!(!view.is_empty(), "the tap must serve at least one frame");
        // The demo scene's tone must be visible in at least one served frame.
        let peak = (0..view.len())
            .flat_map(|i| view.frame(i).iter().copied())
            .fold(f32::MIN, f32::max);
        assert!(
            (-30.0..=-10.0).contains(&peak),
            "expected the −20 dBFS demo tone in a tapped frame, peak was {peak}"
        );
    }

    /// AI-1: the live, wall-clock-cadenced analysis worker (`crate::analyze
    /// ::spawn`, via `Pipeline::start_inner`'s own wiring) actually detects,
    /// tracks, measures and annotates a real signal over a real `Pipeline`
    /// — not the deterministic `AnalysisState::cycle` engine driven
    /// directly (already covered in `crate::analyze::tests`), the thread
    /// this pipeline itself spawns.
    #[cfg(feature = "analyze")]
    #[test]
    fn live_analysis_worker_detects_a_strong_bursty_signal() {
        use phosphene_sources::{HopOrder, HopperConfig, NoiseConfig};

        let mut cfg = SigGenConfig::new(2_048_000.0);
        cfg.seed = 7;
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        cfg.hoppers.push(HopperConfig {
            offsets_hz: vec![300_000.0],
            dwell_s: 0.010,
            period_s: 0.030,
            level_dbfs: -20.0,
            order: HopOrder::Cycle,
        });
        let source = SigGen::new(cfg).unwrap();
        let pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: 512,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        pipeline.set_analyze_active(true);

        let feed = pipeline.annotation_feed();
        wait_until("the live worker to publish an annotation", || {
            !feed.snapshot().is_empty()
        });
        let snapshot = feed.snapshot();
        assert!(
            snapshot
                .iter()
                .any(|a| (a.anchor_hz - 300_000.0).abs() < 10_000.0),
            "expected an annotation near 300 kHz, got {snapshot:?}"
        );
        assert!(pipeline.inspector_active());
        assert!(pipeline.annotation_track_count() > 0);
    }

    /// Isolates whether going through `FileSource`'s real byte-decoding path
    /// (vs feeding `SigGen` directly in-process, as the test above does)
    /// changes anything about detection — the same scene, written to a
    /// temp `.cf32` file and decoded back.
    ///
    /// Fix pass 2, blocker 1: this used to drive the real async worker
    /// thread (`pipeline.set_analyze_active` + `wait_until` polling the
    /// feed) over a `RealTime`-paced `FileSource`, and it timed out once in
    /// a full `--features analyze` run. Diagnosis (temporary instrumented
    /// runs, since removed): with **no other load at all**, a single
    /// invocation showed the worker's own `sleep(CYCLE)` — and the *main
    /// thread's* `wait_until` poll loop, sleeping every 5 ms — both stall
    /// for over a second between an already-successful cycle (annotation
    /// published at t≈250 ms) and the next scheduled wake-up (expected at
    /// t≈500 ms, observed at t≈1.98 s). That is a host-level scheduling
    /// gap affecting every thread in the process, not a product bug in the
    /// tracker, the detector, or the file pacer — and a `wait_until`
    /// deadline cannot be sized against it with any confidence, since nothing
    /// bounds how many such gaps land inside one run, or how long each is.
    /// So rather than widen the deadline and hope, this test no longer
    /// waits on wall-clock time or a background thread at all: it drives
    /// `FileSource::stream` directly on the test thread with `Unthrottled`
    /// pacing (no `sleep` in the reader at all — decode is bounded purely by
    /// CPU) into a capturing `SampleSink`, then folds the decoded samples
    /// through the same deterministic `AnalysisState::cycle` the batched
    /// `SigGen` test (`analyze::tests::
    /// cycle_detects_a_strong_bursty_signal_when_batched`) already proves
    /// correct — so what's actually new here, decode parity, is exactly
    /// what's tested, with nothing left to race. The async worker's own
    /// correctness over a real thread is what
    /// `live_analysis_worker_detects_a_strong_bursty_signal` (above, via
    /// `SigGen` directly) already covers, and the full `--analyze` CLI path
    /// over a real `FileSource` — worker thread included — is covered
    /// end-to-end by `analyze_goldens.rs`, whose subprocess captures already
    /// retry past exactly this kind of host stall (`run_analyze_until_hot`)
    /// rather than trusting a single sample.
    ///
    /// If this environment's scheduling gaps turn out to matter to the real
    /// live product (NFR-A1: the display must never wait on analysis), that
    /// is a live-session finding for whoever runs it on real hardware, not
    /// something a `--headless`/CI test can observe — the windowed path
    /// never blocks on the worker either way (verified in code, `window.rs`
    /// reads the feed without waiting on it).
    #[cfg(feature = "analyze")]
    #[test]
    fn file_source_decoded_samples_detect_the_same_signal_as_siggen() {
        use phosphene_sources::{HopOrder, HopperConfig, NoiseConfig};
        use std::io::Write as _;

        let mut cfg = SigGenConfig::new(2_048_000.0);
        cfg.seed = 7;
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        cfg.hoppers.push(HopperConfig {
            offsets_hz: vec![300_000.0],
            dwell_s: 0.010,
            period_s: 0.030,
            level_dbfs: -20.0,
            order: HopOrder::Cycle,
        });
        let mut gen = SigGen::new(cfg).unwrap();

        let path = std::env::temp_dir().join(format!(
            "phosphene-file-source-detect-{}.cf32",
            std::process::id()
        ));
        // 1000 512-sample frames is 0.25 s at this rate — over 8 of the
        // hopper's 30 ms periods, comfortable margin over the single hop
        // birth needs. Kept modest deliberately: `Detector`/`Tracker`'s
        // per-frame cost is real in an unoptimized debug build (the
        // existing `analyze::tests::
        // cycle_detects_a_strong_bursty_signal_when_batched`, 2000 frames,
        // takes ~13 s CPU-bound on this machine; this test's 1000 takes
        // ~6-7 s) — this now runs on every CI leg (D-101), so frame count
        // trades margin against total CI time deliberately, not by accident.
        const FRAMES: usize = 1000;
        {
            let mut file = std::fs::File::create(&path).unwrap();
            let mut chunk = vec![Complex::new(0.0f32, 0.0); 512];
            let mut bytes = Vec::new();
            for _ in 0..FRAMES {
                gen.fill(&mut chunk);
                bytes.clear();
                for c in &chunk {
                    bytes.extend_from_slice(&c.re.to_le_bytes());
                    bytes.extend_from_slice(&c.im.to_le_bytes());
                }
                file.write_all(&bytes).unwrap();
            }
        }

        let mut source = FileSource::new(FileSourceConfig {
            input: Input::Path(path.clone()),
            format: IqFormat::Cf32,
            sample_rate_hz: Some(2_048_000.0),
            center_freq_hz: None,
            // No sleeping in the reader at all (§ doc above) — decode runs
            // as fast as this thread's CPU allows, so the only thing this
            // test can time out on is a genuine decode bug, never scheduling.
            pace: phosphene_sources::ReplayPace::Unthrottled,
            loop_replay: false,
        })
        .unwrap();

        struct Capture {
            samples: Vec<Complex<f32>>,
            target: usize,
        }
        impl SampleSink for Capture {
            fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
                self.samples.extend_from_slice(samples);
                if self.samples.len() >= self.target {
                    SinkFlow::Stop
                } else {
                    SinkFlow::Continue
                }
            }
        }
        let mut sink = Capture {
            samples: Vec::new(),
            target: FRAMES * 512,
        };
        source.stream(&mut sink).unwrap();
        assert_eq!(
            sink.samples.len(),
            FRAMES * 512,
            "FileSource must decode every sample this test wrote, unthrottled"
        );

        let fft = 512;
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();
        let meta = phosphene_analyze::BandMeta {
            center_freq_hz: None,
            span_hz: 2_048_000.0,
            bins: fft,
            bin_hz: 2_048_000.0 / fft as f64,
            frame_dt_s: fft as f64 / 2_048_000.0,
            cal_offset_db: 0.0,
        };
        let mut state = crate::analyze::AnalysisState::new(&meta);
        let batch = 100usize;
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        let mut annotations = Vec::new();
        // Batched the same way the headless loop and
        // `analyze::tests::cycle_detects_a_strong_bursty_signal_when_batched`
        // do: many spectra per `cycle()` call, not one.
        for batch_samples in sink.samples.chunks(fft * batch) {
            let n_frames = batch_samples.len() / fft;
            if n_frames == 0 {
                break;
            }
            let mut view = phosphene_analyze::FrameView::new(fft, n_frames);
            for frame in batch_samples.chunks_exact(fft) {
                analyzer.process(frame, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            let (anns, _shedding) = state.cycle(&view, &meta, true);
            if !anns.is_empty() {
                annotations = anns;
            }
        }

        assert!(
            !annotations.is_empty(),
            "no annotation produced from FileSource-decoded samples"
        );
        assert!(
            (annotations[0].anchor_hz - 300_000.0).abs() < 10_000.0,
            "expected an annotation near 300 kHz via FileSource, got {annotations:?}"
        );
    }

    /// End-to-end over a pull-paced source: the NFR-P3 negative half at the
    /// integration level. The signal generator is paced by consumption
    /// (clarification C5), so input *cannot* outpace compute: the drop count
    /// must be exactly zero and the percentage exactly 100.
    ///
    /// D-123 (FL-5) class check: waits on `batches_completed >= 32`, then
    /// reads `copy_trace` once with a loose bound (peak somewhere in a wide
    /// range). `MAX_BATCHES_PER_DRAIN` is 16, so 32 crosses at least one
    /// drain boundary — the earlier drain's trace publish is always
    /// complete by then. Cannot race in a way this assertion can see.
    #[test]
    fn siggen_pipeline_runs_with_exactly_zero_drops_and_100_percent() {
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: 1024,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        wait_until("the first published batches", || {
            pipeline.health().batches_completed >= 32
        });
        let mut trace = Vec::new();
        assert!(pipeline.copy_trace(&mut trace));
        assert_eq!(trace.len(), 1024);
        // The demo scene's −20 dBFS tone must be visible in the trace.
        let peak = trace.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (-30.0..=-10.0).contains(&peak),
            "expected the −20 dBFS demo tone, trace peak was {peak}"
        );

        let snap = pipeline.health();
        assert_eq!(snap.dropped_batches, 0);
        assert_eq!(snap.processed_pct, 100.0);
        let meta = pipeline.meta();
        assert_eq!(meta.sample_rate_hz, Some(2_048_000.0));
        pipeline.shutdown();
    }

    /// Deterministic end-of-stream arithmetic: a finite in-memory capture of
    /// exactly 10.5 batches yields exactly 10 completed FFTs — the partial
    /// tail is staged, never processed and never counted (D-013 whole-batch
    /// granularity at the integration level).
    #[test]
    fn finite_capture_completes_exactly_its_whole_batches() {
        let n = 1024usize;
        let total = 10 * n + n / 2;
        let mut bytes = Vec::with_capacity(total * 8);
        for i in 0..total {
            let v = (i as f32 / total as f32) * 0.5;
            bytes.extend_from_slice(&v.to_le_bytes());
            bytes.extend_from_slice(&(-v).to_le_bytes());
        }
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        // No rate: unthrottled replay, pull-paced (C5) — nothing may drop.
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        wait_until("the capture to finish", || {
            pipeline
                .shared
                .source_done
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|r| r.is_ok())
        });
        wait_until("the ring to drain", || {
            pipeline.health().batches_completed == 10
        });
        let snap = pipeline.health();
        assert_eq!(snap.batches_completed, 10);
        assert_eq!(snap.dropped_batches, 0);
        assert_eq!(snap.processed_pct, 100.0);
        assert!(pipeline.source_error().is_none());
        // C5: the axis metadata is honestly rate-less.
        assert_eq!(pipeline.meta().sample_rate_hz, None);
        pipeline.shutdown();
    }

    /// D-033 seal, producer side, through the REAL two-thread pipeline: the
    /// production compute loop folds every completed spectrum into the §7.3
    /// histogram, and the per-frame read hands back a non-empty grid with
    /// the correct `bins × levels` — with the demo scene's −20 dBFS tone lit
    /// at its quantised (bin, level) cell. A test that builds its own
    /// `PersistenceHistogram` and checks its output is exactly the shape
    /// that passed while FR-D1 was dead; this one starts from a source.
    ///
    /// D-123 (FL-5) class check: the initial `wait_until` is only a floor
    /// (`>= 32`), not the read's own condition — the read that matters is
    /// the loop below, which keeps calling `intensity_frame` against a live,
    /// continuously streaming source for 60 ticks (≈ 500 ms), with a loose
    /// band-sum bound (`> 0.5`), not an exact ratio. A fold lagging behind
    /// the wait by one drain cannot make this loop see less than "the tone
    /// showed up eventually". Cannot race in a way this assertion can see.
    #[test]
    fn pipeline_feeds_the_persistence_grid_at_the_display_window() {
        let n = 1024usize;
        let levels = phosphene_core::DEFAULT_LEVELS;
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        wait_until("the first batches", || {
            pipeline.health().batches_completed >= 32
        });

        // A second of display frames, spectra flowing between ticks, so the
        // §7.3 asymmetric EMA converges on the steady tone.
        let mut grid = Vec::new();
        let mut dims = (0, 0);
        for _ in 0..60 {
            std::thread::sleep(Duration::from_millis(8));
            dims = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut grid);
        }
        assert_eq!(dims, (n, levels), "grid dimensions must be bins × levels");
        assert_eq!(grid.len(), n * levels);
        assert!(
            grid.iter().all(|v| (0.0..=1.0).contains(v)),
            "intensity must stay in [0,1]"
        );

        // The demo's +300 kHz tone at 2.048 MS/s lands exactly on raw bin
        // 150 (fftshifted to the upper half); −20 dBFS quantises to level
        // ≈ 104 of 128 over [−110, 0]. Sum a small band to absorb the ±1
        // level of noise wobble.
        let bin = phosphene_core::fftshift_index(150, n);
        let level = ((-20.0f32 - -110.0) / 110.0 * (levels - 1) as f32).round() as usize;
        let band: f32 = (level - 2..=level + 2)
            .map(|l| grid[bin * levels + l])
            .sum();
        assert!(
            band > 0.5,
            "the −20 dBFS tone did not light its histogram cell \
             (bin {bin}, levels {} ±2: sum {band})",
            level
        );
        pipeline.shutdown();
    }

    /// Build step 3: the grid must track the DISPLAYED dB window. When the
    /// reference level / dB-per-div keys move it, the histogram is rebuilt
    /// at the new range — history quantised against the old window is
    /// discarded, never re-drawn against the new one.
    ///
    /// D-123 (FL-5) class check: same shape as
    /// `pipeline_feeds_the_persistence_grid_at_the_display_window` — the
    /// initial wait is a floor, and every read that matters follows a loop
    /// of ticks against a live source with a loose bound, never a single
    /// read racing one specific fold. Cannot race in a way these assertions
    /// can see.
    #[test]
    fn intensity_grid_rebuilds_when_the_display_db_window_moves() {
        let n = 1024usize;
        let levels = phosphene_core::DEFAULT_LEVELS;
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        wait_until("the first batches", || {
            pipeline.health().batches_completed >= 32
        });
        let mut grid = Vec::new();
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(8));
            pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut grid);
        }
        let bin = phosphene_core::fftshift_index(150, n);
        let old_level = ((-20.0f32 - -110.0) / 110.0 * (levels - 1) as f32).round() as usize;
        assert!(
            (old_level - 2..=old_level + 2)
                .map(|l| grid[bin * levels + l])
                .sum::<f32>()
                > 0.5,
            "test premise: the tone is lit before the window moves"
        );

        // The display window narrows to [−55, 0] (a dB/div change). The
        // very first read is a FRESH grid: everything accumulated against
        // the old window is gone, and no counts have landed since.
        let dims = pipeline.intensity_frame(1.0 / 60.0, -55.0, 0.0, &mut grid);
        assert_eq!(dims, (n, levels), "a rebuild keeps the dimensions");
        assert!(
            grid.iter().all(|&v| v == 0.0),
            "history accumulated against the old dB window must not survive \
             into the new one"
        );

        // Fresh accumulation quantises against the new window: −20 dBFS now
        // lands at level ≈ 81 of 128 over [−55, 0], and the old cell index
        // (≈ 104, which now means ≈ −10 dBFS) stays dark.
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(8));
            pipeline.intensity_frame(1.0 / 60.0, -55.0, 0.0, &mut grid);
        }
        let new_level = ((-20.0f32 - -55.0) / 55.0 * (levels - 1) as f32).round() as usize;
        assert!(
            (new_level - 2..=new_level + 2)
                .map(|l| grid[bin * levels + l])
                .sum::<f32>()
                > 0.5,
            "the tone must re-appear at the new window's quantisation"
        );
        assert!(
            (old_level - 1..=old_level + 1)
                .map(|l| grid[bin * levels + l])
                .sum::<f32>()
                < 0.05,
            "the old quantisation (now ≈ −10 dBFS) must be dark"
        );
        pipeline.shutdown();
    }

    /// FR-D3 through the REAL two-thread pipeline (D-031/D-034): the
    /// production compute loop seeds the max hold from a tone-then-silence
    /// capture, and the display-side read then proves both D-007 modes and
    /// the reset — pure hold does not decay, decay-toward-live converges on
    /// the live trace, reset clears — without ever constructing a `MaxHold`
    /// in the test. Waiting for the whole capture to drain first makes the
    /// decay arithmetic deterministic: `dt` is passed, not measured, and no
    /// further spectra arrive.
    ///
    /// D-123 (FL-5), same class: this used to wait on
    /// `health().batches_completed` and then read `copy_trace` and
    /// `max_hold_frame` immediately — both fed by `fold_run`, same as the
    /// τ-decay test's histogram. Waits on [`Pipeline::batches_folded`]
    /// instead.
    #[test]
    fn max_hold_modes_and_reset_through_the_production_pipeline() {
        let n = 512usize;
        let (tone_batches, silent_batches) = (100usize, 400usize);
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let bytes = tone_then_silence_bytes(n, tone_batches, silent_batches);
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        wait_until("the capture to drain", || {
            pipeline.batches_folded() == (tone_batches + silent_batches) as u64
        });

        let mut live = Vec::new();
        pipeline.copy_trace(&mut live);
        let bin = phosphene_core::fftshift_index(n / 4, n);
        assert!(
            live[bin] < -150.0,
            "test premise: after the silence the live trace has fallen away, got {}",
            live[bin]
        );

        // Pure hold: the retained −20 dBFS peak does not decay, tick after
        // tick — and its presence at all proves the production compute loop
        // fed the accumulator.
        let mut trace = Vec::new();
        pipeline.max_hold_frame(0.5, MaxHoldMode::PureHold, &live, &mut trace);
        let held = trace[bin];
        assert!(
            (held - -20.0).abs() < 1.0,
            "the compute loop did not seed the max hold from the tone \
             (bin {bin}: {held} dBFS)"
        );
        for _ in 0..4 {
            pipeline.max_hold_frame(0.5, MaxHoldMode::PureHold, &live, &mut trace);
        }
        assert_eq!(trace[bin], held, "pure hold decayed");

        // Decay-toward-live: strictly decreasing every tick, and after 5 s
        // at the default τ = 2 s (e^{-2.5} ≈ 8% of the gap) essentially
        // converged on the live trace.
        let mut prev = held;
        for _ in 0..10 {
            pipeline.max_hold_frame(0.5, MaxHoldMode::DecayTowardLive, &live, &mut trace);
            assert!(
                trace[bin] < prev,
                "decay-toward-live did not decay ({prev} -> {})",
                trace[bin]
            );
            prev = trace[bin];
        }
        assert!(
            (trace[bin] - live[bin]).abs() < 0.1 * (held - live[bin]).abs(),
            "decay-toward-live did not converge on the live trace \
             (trace {}, live {})",
            trace[bin],
            live[bin]
        );

        // Reset clears to the floor baseline; the stream has ended, so
        // nothing re-seeds it.
        pipeline.reset_max_hold();
        pipeline.max_hold_frame(0.5, MaxHoldMode::DecayTowardLive, &live, &mut trace);
        assert!(
            trace.iter().all(|&v| v == DBFS_FLOOR),
            "reset did not clear the trace"
        );
        pipeline.shutdown();
    }

    /// A source that delivers `tone_batches` of an on-bin −20 dBFS tone up
    /// front, then one `burst_batches`-batch block of **silence** each time
    /// `requested` rises above the count already sent — fresh spectra on
    /// demand between display-side ticks, which is what makes the D-035
    /// τ-decay arithmetic below exact (the histogram folds only when
    /// spectra arrived).
    struct ToneThenSilenceOnDemand {
        meta: SourceMeta,
        requested: Arc<AtomicU64>,
        finished: Arc<AtomicBool>,
        n: usize,
        tone_batches: usize,
        burst_batches: usize,
    }

    impl SampleSource for ToneThenSilenceOnDemand {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "test-tone-then-silence",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta);
            let mut tone = Vec::with_capacity(self.tone_batches * self.n);
            for i in 0..self.tone_batches * self.n {
                let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
                let (im, re) = phase.sin_cos();
                tone.push(Complex::new(re * 0.1, im * 0.1));
            }
            if sink.push(&tone) == SinkFlow::Stop {
                return Ok(());
            }
            let silence = vec![Complex::new(0.0f32, 0.0); self.burst_batches * self.n];
            let mut sent = 0u64;
            let deadline = Instant::now() + Duration::from_secs(60);
            while !self.finished.load(Ordering::Relaxed) && Instant::now() < deadline {
                if self.requested.load(Ordering::Relaxed) > sent {
                    if sink.push(&silence) == SinkFlow::Stop {
                        return Ok(());
                    }
                    sent += 1;
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
            Ok(())
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps::NONE
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            Err(SourceError::UnsupportedControl {
                backend: "test-tone-then-silence",
                control: ctl.name(),
            })
        }

        fn meta(&self) -> SourceMeta {
            self.meta.clone()
        }
    }

    /// D-035, FR-D1's "persistence time" control, through the REAL
    /// two-thread pipeline: shortening the §7.3 persistence decay τ via the
    /// production handler the comma key reaches changes how fast the
    /// phosphor actually fades. With one silent burst delivered between
    /// ticks, the tone cell sees `T = 0` each frame and its intensity
    /// scales by exactly `exp(−dt/τ_decay)` per tick — measured before and
    /// after the τ change, against the exact expectation.
    ///
    /// ## D-123 (FL-5): waits on the fold itself, and proves it under a
    /// forced fold delay
    ///
    /// This test used to wait on `health().batches_completed` — a batch is
    /// counted there when its FFT completes (D-013), which `fold_run` folds
    /// into the §7.3 histogram only afterwards — before reading
    /// `intensity_frame`. When the fold lagged the count, the last tone
    /// batches counted as complete were still un-folded, landed in the decay
    /// frame instead of the pre-decay one, and the measured ratio came out
    /// **above** `exp(−dt/τ)`: 0.672 against 0.607 on `main` (D-123). The
    /// window was microseconds wide — no natural reproduction was seen on a
    /// Linux lane-class machine at this lane's branch point (0 failures in
    /// 200 runs back to back, 0 in 200 with 8 instances at once, 32 of 32
    /// passes with 32 at once under added CPU load) — so this test now
    /// forces the window open with [`fl5_test_fold_delay`] and waits on
    /// [`Pipeline::batches_folded`] instead, proving the fix on every run
    /// rather than on the rare one that would have hit the natural race.
    ///
    /// **Red, before the fix.** With the delay armed and the two
    /// `wait_until` calls below reverted to the old `pipeline.health()
    /// .batches_completed == …` proxy (this lane's diagnostic step, not the
    /// code in this branch — `batches_folded` and `fl5_test_fold_delay` did
    /// not exist yet at that point either):
    /// ```text
    /// thread 'pipeline::tests::persistence_tau_control_changes_the_phosphor_decay' (118409) panicked at crates/phosphene-app/src/pipeline.rs:3008:9:
    /// default τ decay ratio 1.0000454 != exp(−dt/τ) = 0.60653067
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... FAILED
    /// ```
    /// (The forced delay widens the window past what the natural flake ever
    /// hit — 1.0000454 is essentially "no decay measured at all", because
    /// the entire burst was still sitting un-folded when `intensity_frame`
    /// read the histogram.)
    ///
    /// **Green, with the fix** (this branch's code — the delay still armed,
    /// the waits below on `batches_folded`), five consecutive runs:
    /// ```text
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... ok
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... ok
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... ok
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... ok
    /// test pipeline::tests::persistence_tau_control_changes_the_phosphor_decay ... ok
    /// ```
    #[test]
    fn persistence_tau_control_changes_the_phosphor_decay() {
        use phosphene_render::TAU_STEP;

        let n = 512usize;
        let (tone_batches, burst_batches) = (50usize, 10usize);
        let requested = Arc::new(AtomicU64::new(0));
        let finished = Arc::new(AtomicBool::new(false));
        let source = ToneThenSilenceOnDemand {
            meta: SourceMeta {
                sample_rate_hz: None,
                center_freq_hz: None,
                label: "tone-drip".to_owned(),
                provenance: "test".to_owned(),
            },
            requested: requested.clone(),
            finished: finished.clone(),
            n,
            tone_batches,
            burst_batches,
        };
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        // D-123: force the fold-lag window open for the whole test, so
        // waiting on `batches_folded` below is proven under the exact
        // adversity the natural flake needed and almost never got.
        pipeline
            .shared
            .test_fold_delay
            .store(true, Ordering::Relaxed);
        wait_until("the tone burst", || {
            pipeline.batches_folded() == tone_batches as u64
        });

        let levels = DEFAULT_LEVELS;
        let bin = phosphene_core::fftshift_index(n / 4, n);
        let level = ((-20.0f32 - -110.0) / 110.0 * (levels - 1) as f32).round() as usize;
        let band = |grid: &[f32]| -> f32 {
            (level - 2..=level + 2)
                .map(|l| grid[bin * levels + l])
                .sum()
        };

        // Fold the tone counts: every tone spectrum hits the same cell, so
        // T = 1 there and the near-instant §7.3 rise saturates it.
        let mut grid = Vec::new();
        pipeline.intensity_frame(0.5, -110.0, 0.0, &mut grid);
        let i1 = band(&grid);
        assert!(i1 > 0.9, "test premise: the tone cell is lit, got {i1}");

        // One silent burst, then one tick: the cell sees T = 0 and decays
        // by exactly exp(−dt/τ_decay). Waiting on `batches_folded` (D-123)
        // rather than `batches_completed` is what makes this exact: the
        // burst's un-folded tail can no longer land in the read.
        let drip_tick = |req: u64, grid: &mut Vec<f32>| -> f32 {
            requested.store(req, Ordering::Relaxed);
            wait_until("a silent burst", || {
                pipeline.batches_folded() == (tone_batches + req as usize * burst_batches) as u64
            });
            pipeline.intensity_frame(0.5, -110.0, 0.0, grid);
            band(grid)
        };
        let i2 = drip_tick(1, &mut grid);
        let r_default = i2 / i1;
        let expect = (-0.5f32 / phosphene_core::DEFAULT_TAU_DECAY).exp();
        assert!(
            (r_default - expect).abs() < 0.01,
            "default τ decay ratio {r_default} != exp(−dt/τ) = {expect}"
        );

        // The comma key's production handler, two steps shorter: the same
        // measurement must now follow the NEW τ.
        let tau = pipeline.adjust_persistence_tau(TAU_STEP.powi(-2));
        assert!(tau < phosphene_core::DEFAULT_TAU_DECAY);
        let i3 = drip_tick(2, &mut grid);
        let r_short = i3 / i2;
        let expect_short = (-0.5f32 / tau).exp();
        assert!(
            (r_short - expect_short).abs() < 0.01,
            "shortened τ = {tau}: decay ratio {r_short} != {expect_short}"
        );
        assert!(
            r_short < r_default - 0.05,
            "a shorter persistence τ must fade faster ({r_short} vs {r_default})"
        );
        finished.store(true, Ordering::Relaxed);
        pipeline.shutdown();
    }

    /// FR-D2 "averaging time configurable", the measured half (M1-J closes
    /// AUDIT-1's fourth dead row): shortening the §7.4 live-trace τ via the
    /// production handler the K key reaches changes the trace's MEASURED
    /// settling through the real two-thread pipeline. With a declared rate
    /// the per-spectrum interval is fixed (N/rate), so a silent burst of B
    /// batches relaxes the tone bin's gap to the floor by exactly
    /// `exp(−B·interval/τ)` — measured before and after the τ change,
    /// against the exact expectation. The difference is the seal, not the
    /// assignment.
    ///
    /// D-123 (FL-5), same class: this used to wait on
    /// `health().batches_completed` and then settle on a fixed 20 ms sleep
    /// before reading `copy_trace` — a sleep standing in for the condition
    /// this test then asserts exactly (D-094 forbids exactly that). Waits on
    /// [`Pipeline::batches_folded`] instead; the sleep is gone.
    #[test]
    fn live_tau_control_changes_the_measured_settling() {
        use phosphene_render::TAU_STEP;

        let n = 512usize;
        let rate = 2_048_000.0f64;
        let interval = n as f32 / rate as f32; // 0.25 ms per spectrum
        let (tone_batches, burst_batches) = (50usize, 40usize);
        let requested = Arc::new(AtomicU64::new(0));
        let finished = Arc::new(AtomicBool::new(false));
        let source = ToneThenSilenceOnDemand {
            meta: SourceMeta {
                sample_rate_hz: Some(rate),
                center_freq_hz: None,
                label: "tone-drip-rated".to_owned(),
                provenance: "test".to_owned(),
            },
            requested: requested.clone(),
            finished: finished.clone(),
            n,
            tone_batches,
            burst_batches,
        };
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        wait_until("the tone burst", || {
            pipeline.batches_folded() == tone_batches as u64
        });

        let bin = phosphene_core::fftshift_index(n / 4, n);
        let mut trace = Vec::new();
        pipeline.copy_trace(&mut trace);
        let v0 = trace[bin];
        assert!(
            v0 > -25.0,
            "test premise: the tone seeded the EMA, got {v0}"
        );

        // One silent burst at the default τ: the gap to the −200 floor
        // relaxes by exp(−B·interval/τ) exactly.
        let gap = |v: f32| v - DBFS_FLOOR;
        let drip = |req: u64, trace: &mut Vec<f32>| -> f32 {
            requested.store(req, Ordering::Relaxed);
            wait_until("a silent burst", || {
                pipeline.batches_folded() == (tone_batches + req as usize * burst_batches) as u64
            });
            pipeline.copy_trace(trace);
            trace[bin]
        };
        let v1 = drip(1, &mut trace);
        let r_default = gap(v1) / gap(v0);
        let expect = (-(burst_batches as f32) * interval / DEFAULT_TAU_LIVE).exp();
        assert!(
            (r_default - expect).abs() < 0.01,
            "default τ_live settling ratio {r_default} != {expect}"
        );

        // The K key's production handler, three steps shorter: the same
        // measurement must now follow the NEW τ — faster settling.
        let tau = pipeline.adjust_live_tau(TAU_STEP.powi(-3));
        assert!(tau < DEFAULT_TAU_LIVE);
        let v2 = drip(2, &mut trace);
        let r_short = gap(v2) / gap(v1);
        let expect_short = (-(burst_batches as f32) * interval / tau).exp();
        assert!(
            (r_short - expect_short).abs() < 0.01,
            "shortened τ_live = {tau}: settling ratio {r_short} != {expect_short}"
        );
        assert!(
            r_short < r_default - 0.02,
            "a shorter averaging τ must settle faster ({r_short} vs {r_default})"
        );
        finished.store(true, Ordering::Relaxed);
        pipeline.shutdown();
    }

    /// D-035 τ safety boundary: keypress factors multiply from the current
    /// value, clamp at both ends, and a degenerate factor changes nothing.
    #[test]
    fn tau_adjustments_multiply_clamp_and_survive_degenerate_factors() {
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: 1024,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        let t = pipeline.adjust_persistence_tau(2.0);
        assert!((t - 2.0 * phosphene_core::DEFAULT_TAU_DECAY).abs() < 1e-6);
        assert_eq!(pipeline.adjust_persistence_tau(1e12), TAU_MAX_S);
        assert_eq!(pipeline.adjust_persistence_tau(1e-12), TAU_MIN_S);
        assert_eq!(
            pipeline.adjust_persistence_tau(f32::NAN),
            TAU_MIN_S,
            "a non-finite factor must change nothing"
        );
        let m = pipeline.adjust_max_hold_tau(0.5);
        assert!((m - 0.5 * phosphene_core::DEFAULT_TAU_MAX_HOLD).abs() < 1e-6);
        assert_eq!(pipeline.adjust_max_hold_tau(0.0), m, "zero factor ignored");
        assert_eq!(pipeline.adjust_max_hold_tau(1e12), TAU_MAX_S);
        // The §7.4 live τ (M1-J): same clamp, same idiom.
        let l = pipeline.adjust_live_tau(2.0);
        assert!((l - 2.0 * DEFAULT_TAU_LIVE).abs() < 1e-6);
        assert_eq!(pipeline.adjust_live_tau(1e12), TAU_MAX_S);
        assert_eq!(pipeline.adjust_live_tau(1e-12), TAU_MIN_S);
        assert_eq!(
            pipeline.adjust_live_tau(f32::NAN),
            TAU_MIN_S,
            "a non-finite factor must change nothing"
        );
        pipeline.shutdown();
    }

    /// A test source delivering two bursts, the second held back until the
    /// test releases it — the deterministic overload shape the D-028 backlog
    /// seal needs.
    struct TwoBurstSource {
        meta: SourceMeta,
        release: Arc<AtomicBool>,
        samples_per_burst: usize,
    }

    impl SampleSource for TwoBurstSource {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "test-two-burst",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta);
            let block = vec![Complex::new(0.05f32, 0.0); self.samples_per_burst];
            if sink.push(&block) == SinkFlow::Stop {
                return Ok(());
            }
            let deadline = Instant::now() + Duration::from_secs(30);
            while !self.release.load(Ordering::Relaxed) {
                if Instant::now() > deadline {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            sink.push(&block);
            Ok(())
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps::NONE
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            Err(SourceError::UnsupportedControl {
                backend: "test-two-burst",
                control: ctl.name(),
            })
        }

        fn meta(&self) -> SourceMeta {
            self.meta.clone()
        }
    }

    /// The D-028 seal: the rolling window attributes a completed batch at its
    /// RECEIPT time, on the same clock as the drops it arrived with — proven
    /// through the real two-thread pipeline, driven deterministically by an
    /// injected slow consumer (a permit gate on the compute thread) and fixed
    /// counts, not a wall-clock race.
    ///
    /// Shape: burst 1 (20 batches) hits a gated compute thread — the 8-batch
    /// ring fills, 12 whole batches are shed at receipt. More than one full
    /// window later the gate releases and the 8 queued batches complete;
    /// then burst 2 repeats the overload. At the final snapshot, only burst
    /// 2's 12 drops were *received* inside the window, so the honest
    /// percentage is exactly 0. Stamping completions at FFT-completion time
    /// instead (the pre-D-028 defect) would put the 8 stale completions
    /// inside the window and report 40% — this test is RED on that code.
    #[test]
    fn backlog_completions_are_attributed_at_receipt_time() {
        let n = 512;
        let burst_batches = 20usize;
        // rate 8 kHz → ring = max(ceil(0.25·8000), 8·512) = 4096 samples =
        // exactly 8 whole batches; paced ⇒ the 12 that do not fit are shed.
        let meta = SourceMeta {
            sample_rate_hz: Some(8_000.0),
            center_freq_hz: None,
            label: "two-burst".to_owned(),
            provenance: "test".to_owned(),
        };
        let release = Arc::new(AtomicBool::new(false));
        let gate = Arc::new(AtomicU64::new(0));
        let source = TwoBurstSource {
            meta,
            release: release.clone(),
            samples_per_burst: burst_batches * n,
        };
        let mut pipeline = Pipeline::start_gated(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: true,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate.clone(),
        )
        .unwrap();

        // Burst 1: 8 batches queue (compute is starved), 12 shed at receipt.
        wait_until("burst-1 drops", || pipeline.health().dropped_batches == 12);
        assert_eq!(pipeline.health().batches_completed, 0);

        // Let the backlog age past the whole rolling window, then complete
        // it: 8 permits, 8 queued batches, FFTs run NOW — but they were
        // received alongside burst 1's drops.
        std::thread::sleep(Duration::from_secs_f64(HEALTH_WINDOW_S + 0.3));
        gate.store(8, Ordering::Relaxed);
        wait_until("the backlog to complete", || {
            pipeline.health().batches_completed == 8
        });

        // Burst 2: a fresh overload with compute starved again.
        release.store(true, Ordering::Relaxed);
        wait_until("burst-2 drops", || pipeline.health().dropped_batches == 24);

        let snap = pipeline.health();
        assert_eq!(snap.batches_completed, 8);
        assert_eq!(snap.dropped_batches, 24);
        // Received in the window: burst 2's 12 shed batches only. The 8
        // completions belong — by receipt — to a window that has passed;
        // counting them here against only the fresh drops would overstate
        // health (40%) exactly when the HUD matters.
        assert_eq!(snap.processed_pct, 0.0);
        pipeline.shutdown();
    }

    /// A source that delivers one burst up front and then one further burst
    /// each time `requested` rises above the count already sent — the
    /// deterministic drip the D-030 cyclic seal orchestrates.
    struct BurstsOnDemandSource {
        meta: SourceMeta,
        requested: Arc<AtomicU64>,
        finished: Arc<AtomicBool>,
        samples_per_burst: usize,
    }

    impl SampleSource for BurstsOnDemandSource {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "test-bursts-on-demand",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta);
            let block = vec![Complex::new(0.05f32, 0.0); self.samples_per_burst];
            if sink.push(&block) == SinkFlow::Stop {
                return Ok(());
            }
            let mut sent = 0u64;
            let deadline = Instant::now() + Duration::from_secs(60);
            while !self.finished.load(Ordering::Relaxed) && Instant::now() < deadline {
                if self.requested.load(Ordering::Relaxed) > sent {
                    if sink.push(&block) == SinkFlow::Stop {
                        return Ok(());
                    }
                    sent += 1;
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
            Ok(())
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps::NONE
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            Err(SourceError::UnsupportedControl {
                backend: "test-bursts-on-demand",
                control: ctl.name(),
            })
        }

        fn meta(&self) -> SourceMeta {
            self.meta.clone()
        }
    }

    /// The D-030 seal, through the real two-thread pipeline: a completion
    /// whose RECEIPT has already aged out of the rolling window when its FFT
    /// finally runs must not move `processed_pct` or `ffts_per_s` — and must
    /// still appear in the lifetime totals D-013 requires.
    ///
    /// Shape: the opening burst fills the 8-batch ring at receipt t₀ (12
    /// shed); the queued receipts then age past the whole window. Eight
    /// cycles follow, each one: a fresh burst of shed batches pins the
    /// window's head to ≈now, then exactly ONE backlogged FFT (receipt t₀,
    /// aged out) completes while the test samples the figures tightly. At
    /// every observed instant the window holds fresh drops and no in-window
    /// completions, so the honest percentage is exactly 0 and the FFT rate
    /// exactly 0 — any other value is an aged batch being counted inside a
    /// window it left. The pre-D-030 clamp fails exactly that way, within a
    /// bucket-width of each cycle's completion; eight cycles of independent
    /// bucket phase make the tight sampler observe it with near-certainty,
    /// while on correct code every sample is 0 regardless of timing.
    #[test]
    fn aged_out_receipts_leave_the_window_but_not_the_totals() {
        let n = 512;
        let burst_batches = 20usize;
        let cycles = 8u64;
        let meta = SourceMeta {
            sample_rate_hz: Some(8_000.0),
            center_freq_hz: None,
            label: "bursts-on-demand".to_owned(),
            provenance: "test".to_owned(),
        };
        let requested = Arc::new(AtomicU64::new(0));
        let finished = Arc::new(AtomicBool::new(false));
        let gate = Arc::new(AtomicU64::new(0));
        let source = BurstsOnDemandSource {
            meta,
            requested: requested.clone(),
            finished: finished.clone(),
            samples_per_burst: burst_batches * n,
        };
        let mut pipeline = Pipeline::start_gated(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: true,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate.clone(),
        )
        .unwrap();

        // Opening burst: the 8-batch ring fills at receipt t₀, 12 shed.
        wait_until("the opening drops", || {
            pipeline.health().dropped_batches == 12
        });
        assert_eq!(pipeline.health().batches_completed, 0);
        // Age the queued receipts past the whole rolling window.
        std::thread::sleep(Duration::from_secs_f64(HEALTH_WINDOW_S + 0.3));

        let mut expected_drops = 12u64;
        for cycle in 0..cycles {
            // Fresh shed batches pin the window head to ≈now. The first
            // on-demand burst finds the ring still full (20 shed); later
            // ones find the one slot freed by the previous cycle's
            // completion (1 accepted with a FRESH receipt — queued behind
            // the stale ones — and 19 shed).
            requested.store(cycle + 1, Ordering::Relaxed);
            expected_drops += if cycle == 0 { 20 } else { 19 };
            wait_until("this cycle's drops", || {
                pipeline.health().dropped_batches == expected_drops
            });

            // One permit: the OLDEST queued batch — receipt t₀, aged out —
            // completes now. Sample tightly until it lands: at no observable
            // instant may it move the window figures.
            gate.store(1, Ordering::Relaxed);
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                let snap = pipeline.health();
                assert_eq!(
                    snap.processed_pct, 0.0,
                    "an aged-out receipt moved the window (cycle {cycle}): {snap:?}"
                );
                assert_eq!(
                    snap.ffts_per_s, 0.0,
                    "an aged-out receipt moved the FFT rate (cycle {cycle}): {snap:?}"
                );
                if snap.batches_completed == cycle + 1 {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for cycle {cycle}'s completion"
                );
                std::thread::sleep(Duration::from_micros(200));
            }
        }

        let snap = pipeline.health();
        // The lifetime totals D-013 requires keep every aged batch…
        assert_eq!(snap.batches_completed, cycles);
        assert_eq!(snap.dropped_batches, 12 + 20 + (cycles - 1) * 19);
        // …while the window never saw one.
        assert_eq!(snap.processed_pct, 0.0);
        finished.store(true, Ordering::Relaxed);
        pipeline.shutdown();
    }

    /// A source that delivers a fixed number of clean batches, then reports
    /// device-side losses (the gated D-048 path) and overflow events
    /// (D-050) — the driver for both honesty seals.
    struct LossReportingSource {
        meta: SourceMeta,
        batches: usize,
        batch_len: usize,
        /// Two loss reports, in samples, delivered after the clean batches.
        losses: [u64; 2],
        /// Overflow EVENTS reported after the losses (D-050).
        overflow_events: usize,
    }

    impl SampleSource for LossReportingSource {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "test-loss-reporting",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta);
            let block = vec![Complex::new(0.05f32, 0.0); self.batch_len];
            for _ in 0..self.batches {
                if sink.push(&block) == SinkFlow::Stop {
                    return Ok(());
                }
            }
            for loss in self.losses {
                sink.device_lost(loss);
            }
            for _ in 0..self.overflow_events {
                sink.device_overflow();
            }
            Ok(())
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps::NONE
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            Err(SourceError::UnsupportedControl {
                backend: "test-loss-reporting",
                control: ctl.name(),
            })
        }

        fn meta(&self) -> SourceMeta {
            self.meta.clone()
        }
    }

    /// The D-048 seal: a device-reported loss moves BOTH FR-D11 figures
    /// through the real accounting — `PROC %` falls (the lost samples
    /// entered the denominator) and `DROPS` rises (they count as dropped
    /// batches) — exactly as for a locally shed batch, driven through the
    /// real two-thread pipeline.
    ///
    /// The loss arrives as 3.5 batches then 0.5 batches: the final count of
    /// **4** dropped batches also proves sub-batch remainders accrue across
    /// reports (per-report flooring would say 3; ceiling would say 5).
    #[test]
    fn device_reported_loss_moves_proc_and_drops_through_the_accounting() {
        let n = 512usize;
        let source = LossReportingSource {
            meta: SourceMeta {
                sample_rate_hz: Some(8_000.0),
                center_freq_hz: None,
                label: "loss-reporting".to_owned(),
                provenance: "test".to_owned(),
            },
            batches: 8,
            batch_len: n,
            losses: [3 * n as u64 + n as u64 / 2, n as u64 / 2],
            overflow_events: 0,
        };
        let pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                // Pull-paced: nothing is shed locally, so every drop the
                // snapshot shows came from the device report.
                paced: false,
                backpressure: false,
                db_bottom: -120.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        wait_until("8 completions and 4 device-loss drops", || {
            let snap = pipeline.health();
            snap.batches_completed == 8 && snap.dropped_batches == 4
        });
        let snap = pipeline.health();
        assert_eq!(
            snap.dropped_batches, 4,
            "remainder accrual broken: {snap:?}"
        );
        assert_eq!(snap.batches_completed, 8);
        // 8 completed of 12 resolved in the window: the HUD stops claiming
        // 100% the moment the device reports a loss.
        assert!(
            (snap.processed_pct - 100.0 * 8.0 / 12.0).abs() < 0.1,
            "PROC% did not absorb the device loss: {snap:?}"
        );
    }

    /// The D-050 seal: overflow EVENTS are plumbed through and countable —
    /// and they do NOT touch the D-013 figures. `PROC %` stays exactly 100
    /// and `DROPS` stays 0, because an event carries no quantity and
    /// inventing one is the defect D-050 removed.
    #[test]
    fn overflow_events_are_counted_and_never_pollute_the_accounting() {
        let n = 512usize;
        let source = LossReportingSource {
            meta: SourceMeta {
                sample_rate_hz: Some(8_000.0),
                center_freq_hz: None,
                label: "overflow-events".to_owned(),
                provenance: "test".to_owned(),
            },
            batches: 4,
            batch_len: n,
            losses: [0, 0],
            overflow_events: 3,
        };
        let pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -120.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        wait_until("4 completions and 3 overflow events", || {
            pipeline.health().batches_completed == 4 && pipeline.device_overflow_events() == 3
        });
        let snap = pipeline.health();
        assert_eq!(pipeline.device_overflow_events(), 3);
        assert_eq!(
            snap.dropped_batches, 0,
            "an event was dressed up as a drop: {snap:?}"
        );
        assert_eq!(
            snap.processed_pct, 100.0,
            "an event with no quantity moved PROC%: {snap:?}"
        );
    }

    /// A rated source that pushes `before` tone batches, waits for the
    /// test's `proceed` signal, reports a device loss of `lost` whole
    /// batches, then pushes `after` more — the shape the §7.6 gap seal
    /// needs: counted drops **between** data, at a moment the test controls
    /// so the drained row order is deterministic.
    struct GapReportingSource {
        meta: SourceMeta,
        n: usize,
        before: usize,
        lost: u64,
        after: usize,
        proceed: Arc<AtomicBool>,
    }

    impl SampleSource for GapReportingSource {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "test-gap-reporting",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta);
            let tone: Vec<Complex<f32>> = (0..self.n)
                .map(|i| {
                    let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
                    let (im, re) = phase.sin_cos();
                    Complex::new(re * 0.1, im * 0.1)
                })
                .collect();
            for _ in 0..self.before {
                if sink.push(&tone) == SinkFlow::Stop {
                    return Ok(());
                }
            }
            let deadline = Instant::now() + Duration::from_secs(30);
            while !self.proceed.load(Ordering::Relaxed) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            sink.device_lost(self.lost * self.n as u64);
            for _ in 0..self.after {
                if sink.push(&tone) == SinkFlow::Stop {
                    return Ok(());
                }
            }
            Ok(())
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps::NONE
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            Err(SourceError::UnsupportedControl {
                backend: "test-gap-reporting",
                control: ctl.name(),
            })
        }

        fn meta(&self) -> SourceMeta {
            self.meta.clone()
        }
    }

    /// §7.6/D-051, the floor-on-empty half at the production seam: a
    /// counted drop — here a device-reported loss travelling through the
    /// REAL `RingSink::device_lost` accounting — is signal time with no
    /// data, and the waterfall renders it as floor rows between the data
    /// rows instead of silently splicing the time axis. The proceed signal
    /// makes the interleaving exact: the loss lands between two
    /// fully-drained halves, so the drained sequence is
    /// [data, data, floor, floor, data, data].
    ///
    /// This test and its windowed siblings go RED on the pre-D-051 feed
    /// (one spectrum per display frame): rows there never saw the drop
    /// accounting at all.
    ///
    /// D-123 (FL-5) class check: `collect` below already polls
    /// `waterfall_frame` in a loop until the expected row count shows up,
    /// with its own comment on why ("a single drain can land between the
    /// compute thread's completion record and its waterfall fold; the
    /// display drains every frame and never cares") — this is the same
    /// reasoning D-123 formalizes, applied here before this lane existed.
    /// No single read races one specific fold. Cannot race in a way these
    /// assertions can see.
    #[test]
    fn counted_drops_render_as_floor_rows_between_data() {
        use phosphene_render::{RowAggregation, WaterfallMode, ROW_FLOOR_DBFS};

        let n = 512usize;
        // 8 kHz rate → one spectrum spans 0.064 s; every quantity below is
        // that times a power of two, so the row boundaries are exact in f32.
        let si = (n as f64 / 8_000.0) as f32;
        let gate = Arc::new(AtomicU64::new(0));
        let proceed = Arc::new(AtomicBool::new(false));
        let source = GapReportingSource {
            meta: SourceMeta {
                sample_rate_hz: Some(8_000.0),
                center_freq_hz: None,
                label: "gap-reporting".to_owned(),
                provenance: "test".to_owned(),
            },
            n,
            before: 4,
            lost: 4,
            after: 4,
            proceed: proceed.clone(),
        };
        let mut pipeline = Pipeline::start_gated(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate.clone(),
        )
        .unwrap();

        // Two spectra per row: a 4-row panel spanning 8 spectra.
        let cfg = WfFrameConfig {
            mode: WaterfallMode::Traditional,
            aggregation: RowAggregation::Max,
            span_s: 8.0 * si,
            fast_interval_s: 1e-3,
            span_rows: 1024.0,
            panel_rows: 4,
        };
        let mut rows = Vec::new();
        // Configure before any spectrum flows.
        pipeline.waterfall_frame(&cfg, &mut rows);

        // Accumulate drains until `expect` rows arrive (a single drain can
        // land between the compute thread's completion record and its
        // waterfall fold; the display drains every frame and never cares).
        let collect = |expect: usize| -> Vec<f32> {
            let mut all = Vec::new();
            let mut scratch = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(10);
            while all.len() / n < expect && Instant::now() < deadline {
                pipeline.waterfall_frame(&cfg, &mut scratch);
                all.extend_from_slice(&scratch);
                std::thread::sleep(Duration::from_millis(2));
            }
            assert_eq!(all.len() / n, expect, "expected {expect} rows");
            all
        };

        // First half: 4 completed batches, fully folded — 2 data rows —
        // while the loss report waits on the proceed signal.
        gate.fetch_add(4, Ordering::Relaxed);
        let first = collect(2);
        assert!(first.chunks_exact(n).all(|r| r.iter().any(|&v| v > -150.0)));

        // Release the loss, then the second half: the drain folds the
        // counted gap first, then the post-loss spectra — two floor rows,
        // then two data rows, in that order.
        proceed.store(true, Ordering::Relaxed);
        wait_until("the loss to be recorded", || {
            pipeline.health().dropped_batches == 4
        });
        gate.fetch_add(4, Ordering::Relaxed);
        let second = collect(4);
        let drained: Vec<&[f32]> = second.chunks_exact(n).collect();
        for (i, row) in drained[..2].iter().enumerate() {
            assert!(
                row.iter().all(|&v| v == ROW_FLOOR_DBFS),
                "gap row {i} is not floor — the drop was spliced out of the \
                 time axis (§7.6/D-051)"
            );
        }
        for (i, row) in drained[2..].iter().enumerate() {
            assert!(
                row.iter().any(|&v| v > -150.0),
                "post-gap data row {i} came out floor"
            );
        }
        pipeline.shutdown();
    }

    /// The same property for **locally shed** batches (NFR-P3's paced-mode
    /// degradation, through the REAL `RingSink::push` shed path): starve
    /// the compute gate under a paced source until the ring sheds, then let
    /// it drain — the drained rows must contain both floor rows (the shed
    /// stretches) and data rows. Shed timing is scheduler-dependent, so the
    /// assertions are presence, not counts.
    ///
    /// D-123 (FL-5) class check: the `batches_completed > 64` wait is a
    /// floor before a polling loop (up to 10 s) that keeps reading
    /// `waterfall_frame` until both a floor row and a data row have shown
    /// up. No single read races one specific fold. Cannot race in a way
    /// these assertions can see.
    #[test]
    fn locally_shed_batches_render_as_floor_rows() {
        use phosphene_render::{RowAggregation, WaterfallMode};

        let n = 1024usize;
        let si = (n as f64 / 2_048_000.0) as f32;
        let gate = Arc::new(AtomicU64::new(0));
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start_gated(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: WindowKind::Hann,
                // Time-paced: a full ring sheds whole batches, counted.
                paced: true,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate.clone(),
        )
        .unwrap();

        // One spectrum per row, so every shed batch is one floor row.
        let cfg = WfFrameConfig {
            mode: WaterfallMode::Traditional,
            aggregation: RowAggregation::Max,
            span_s: 64.0 * si,
            fast_interval_s: 1e-3,
            span_rows: 1024.0,
            panel_rows: 64,
        };
        let mut rows = Vec::new();
        pipeline.waterfall_frame(&cfg, &mut rows);

        // Starved gate + paced source: the ring fills and sheds.
        wait_until("the ring to shed under starvation", || {
            pipeline.health().dropped_batches > 8
        });
        // Open the gate and let real spectra flow after the shed stretch.
        gate.fetch_add(u64::MAX / 2, Ordering::Relaxed);
        wait_until("post-shed completions", || {
            pipeline.health().batches_completed > 64
        });
        let mut floor_rows = 0usize;
        let mut data_rows = 0usize;
        let deadline = Instant::now() + Duration::from_secs(10);
        while (floor_rows == 0 || data_rows == 0) && Instant::now() < deadline {
            pipeline.waterfall_frame(&cfg, &mut rows);
            for row in rows.chunks_exact(n) {
                if row.iter().any(|&v| v > -150.0) {
                    data_rows += 1;
                } else {
                    floor_rows += 1;
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            floor_rows > 0,
            "shed batches produced no floor rows — the gap was spliced out \
             (§7.6/D-051; {data_rows} data rows seen)"
        );
        assert!(data_rows > 0, "no data rows after the shed stretch");
        pipeline.shutdown();
    }
}
