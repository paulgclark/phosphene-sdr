// SPDX-License-Identifier: MIT

//! The live analysis engine (AI-1, nextgen spec §8.2): [`AnalysisState`]
//! runs detect → track → measure → annotate deterministically over whatever
//! frames it is handed, and [`spawn`] drives it on its own thread at
//! NFR-A1's wall-clock cadence, reading the app's `SpectrumTap`
//! implementation and publishing through
//! `phosphene_render::AnnotationFeed::push_annotations` (D-017's frozen
//! render seam).
//!
//! [`AnalysisState`] is deliberately cadence-agnostic — it does no sleeping
//! and touches no clock — so the deterministic `--headless` path
//! ([`crate::headless`]) can drive it frame-count-synchronously (NFR-A4) and
//! the live windowed path can drive it wall-clock-cadenced, from the exact
//! same engine.
//!
//! Compiled only behind the `analyze` cargo feature; the live worker is
//! further gated at runtime by [`AnalysisHandle::set_active`] (the
//! `--analyze` flag / the `I` key, FR-AU1) — with either off, nothing here
//! runs and the display is exactly v1's.
//!
//! **The display must never wait for the live worker (NFR-A1).** It owns no
//! lock any display-path code also takes: it only reads the tap (which
//! itself takes the same short-held lock the compute thread already uses to
//! fill it, per the module's existing "one lock per drain" discipline) and
//! writes the feed (its own independent mutex, guarding only a pointer
//! swap).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use phosphene_analyze::detect::{Detector, DetectorConfig};
use phosphene_analyze::measure::{MeasureConfig, Measurer};
use phosphene_analyze::track::{Tracker, TrackerConfig};
use phosphene_analyze::{build_annotation, BandMeta, FrameView, SpectrumTap};
use phosphene_render::{AnalysisLoad, AnnotationFeed};

/// Cadence the live worker re-evaluates tracks at — NFR-A1's "a few hundred
/// ms". `pub(crate)` (D-110): the lockstep headless capture
/// (`crate::headless::run_file_lockstep_frames`) reads this same constant,
/// never a copy, to reproduce this worker's real frame coverage
/// deterministically instead of full coverage.
pub(crate) const CYCLE: Duration = Duration::from_millis(250);

/// Frames the tap retains between analysis polls (≈ [`CYCLE`] worth at a
/// moderate FFT rate). Sized deliberately bounded, not "big enough for any
/// rate": NFR-A1 caps analysis CPU at roughly one core, so a source
/// producing more spectra than this between two polls is exactly NFR-A5's
/// overload case — the tap cannot have retained all of them, the worker
/// reports `AnalysisLoad { shedding: true }` (never silently falls further
/// and further behind trying to catch up), and the display is never made to
/// wait for it either way.
pub const TAP_HISTORY_FRAMES: usize = 256;

/// The deterministic detect → track → measure → annotate engine (NFR-A4: the
/// same frames and configuration yield the same tracks and annotations,
/// regardless of what drives [`Self::cycle`] or how often). Owns no clock —
/// callers decide cadence.
pub struct AnalysisState {
    detector: Detector,
    tracker: Tracker,
    measurer: Measurer,
    last_seq: Option<u64>,
    /// D-125 Amendment 4 item 3, the owner's ruling: "show an honest
    /// coverage count: frames analysed out of frames received." Cumulative
    /// since construction/[`reset`](Self::reset); every raw frame the
    /// source has produced since then, by sequence-number delta, whether or
    /// not it survived to reach [`Self::cycle`]'s `view`.
    frames_received: u64,
    /// D-125 Amendment 4 item 3: cumulative frames actually folded into the
    /// detector/tracker since construction/reset — always `<=
    /// frames_received`.
    frames_analysed: u64,
}

impl AnalysisState {
    /// A fresh engine for a band of `meta.bins` bins.
    pub fn new(meta: &BandMeta) -> Self {
        AnalysisState {
            detector: Detector::new(meta.bins, DetectorConfig::new())
                .expect("default detector config is valid"),
            tracker: Tracker::new(TrackerConfig::default())
                .expect("default tracker config is valid"),
            measurer: Measurer::new(MeasureConfig::default())
                .expect("default measure config is valid"),
            last_seq: None,
            frames_received: 0,
            frames_analysed: 0,
        }
    }

    /// Rebuild fresh, as if newly constructed — D-056/AC-9's retune reset:
    /// every frequency-dependent accumulator here is discarded, exactly like
    /// persistence/max-hold/waterfall.
    pub fn reset(&mut self, meta: &BandMeta) {
        *self = Self::new(meta);
    }

    /// Fold every frame in `view` newer than the last one this engine has
    /// seen, then return the resulting annotations and whether the caller's
    /// frame source has fallen behind — `view` held fewer frames than were
    /// actually produced since the last call, so some were lost before this
    /// engine could see them (NFR-A5: real overload, not a bug).
    pub fn cycle(
        &mut self,
        view: &FrameView,
        meta: &BandMeta,
        rate_known: bool,
    ) -> (Vec<phosphene_render::Annotation>, bool) {
        if view.is_empty() {
            return (Vec::new(), false);
        }
        let shedding = self.last_seq.is_some_and(|last| view.seq(0) > last + 1);

        // D-125 Amendment 4 item 3: the honest coverage count. Every
        // sequence number that has elapsed since the last cycle counts as
        // "received", whether or not it survived to be in `view` — the
        // producer's own sequence counter (`FrameView`/`FrameRing`'s
        // contract) is monotonic and gapless at the source, so the delta
        // from the last frame this engine ever saw to the newest frame in
        // `view` now is exactly how many frames the source produced in
        // between, shed ones included. On the very first cycle there is no
        // prior frame to delta against, so this cycle's own frames are the
        // only ones countable as received.
        let newest_seq = view.seq(view.len() - 1);
        self.frames_received += match self.last_seq {
            Some(last) => newest_seq.saturating_sub(last),
            None => view.len() as u64,
        };

        // The tracker's M-of-N birth/death and the detector's own
        // time-domain blob formation are both frame-sequential state
        // machines: each frame's own (usually empty) detections must reach
        // `Tracker::observe` in its own call, tagged with that frame's own
        // sequence number — never accumulated across many frames and
        // reported under one seq, which would corrupt the birth window.
        let last_new_index =
            (0..view.len()).rfind(|&i| self.last_seq.is_none_or(|last| view.seq(i) > last));
        for i in 0..view.len() {
            let seq = view.seq(i);
            if self.last_seq.is_some_and(|last| seq <= last) {
                continue;
            }
            let mut detections = self.detector.push_frame(view.frame(i), seq);
            if Some(i) == last_new_index {
                // A signal present continuously never closes its own blob
                // (§7.3: a component "ends" only when its bins stop being
                // occupied), so a persistent track would never be reported
                // at all without a periodic flush. Flushing once per cycle
                // — here, at the newest frame of the batch — reports every
                // still-active region at the analysis cadence (NFR-A1),
                // exactly like a bursty signal's blob closing on its own.
                detections.extend(self.detector.finish());
            }
            self.tracker.observe(seq, &detections, meta);
            self.last_seq = Some(seq);
            self.frames_analysed += 1;
        }

        let noise_floor = self.detector.noise_floor().to_vec();
        // D-125 item 4, re-wired on full coverage (item 6): publish a track
        // that confirmed and then died (coast timeout) within this same
        // cycle, not only what `tracker.tracks()` still reports live at the
        // last frame processed above. First attempted, then reverted, under
        // *partial* coverage: the old TAP_HISTORY_FRAMES-capped lockstep
        // capture handed each cycle a trailing ~64ms/256-frame window that
        // could contain several independent bursts, so draining every death
        // in that window turned one expected label into three or four
        // duplicate ones (same anchor, drifting power) — a real regression
        // against the goldens of that time, not a case of them needing
        // retuning. Under full coverage (D-125 item 6: every raw frame
        // reaches this method, in order, not a trailing slice of them) that
        // shape does not arise: a cycle's births and deaths are exactly the
        // real ones, at most one confirm-then-die per real burst that
        // happens to end inside this cycle's boundary, which is what item 4
        // asks to publish in the first place.
        //
        // `retired` only ever holds tracks that reached `Confirmed` before
        // dying (`Tracker::observe`'s own doc comment: M-of-N suppressed a
        // tentative before it could retire), so every drained track here is
        // a real, previously-visible signal, not a suppressed blip.
        let newly_dead = self.tracker.drain_retired();
        let mut annotations = Vec::new();
        for track in self.tracker.tracks().cloned().chain(newly_dead) {
            let mut track = track;
            if let Ok(m) = self
                .measurer
                .measure_track(&track, view, meta, &noise_floor)
            {
                track.measurements = Some(m);
            }
            if let Some(ann) = build_annotation(&track, meta, rate_known) {
                annotations.push(convert(ann));
            }
        }
        (annotations, shedding)
    }

    /// D-125 Amendment 4 item 3: cumulative frames the source has produced
    /// since construction/[`reset`](Self::reset), by sequence-number delta
    /// — see [`Self::cycle`]'s own accounting.
    pub fn frames_received(&self) -> u64 {
        self.frames_received
    }

    /// D-125 Amendment 4 item 3: cumulative frames actually folded into the
    /// detector/tracker since construction/[`reset`](Self::reset). Always
    /// `<= frames_received()`.
    pub fn frames_analysed(&self) -> u64 {
        self.frames_analysed
    }
}

/// Cross-thread handle to the live worker's runtime state: whether it is
/// active (FR-AU1), the current live track count (for the status-bar chip),
/// and whether it is shedding (NFR-A5). Also the stop flag, checked between
/// sleeps so the thread exits promptly on shutdown.
#[derive(Debug, Default)]
pub struct AnalysisHandle {
    active: AtomicBool,
    track_count: AtomicUsize,
    shedding: AtomicBool,
    stop: AtomicBool,
    /// D-125 Amendment 4 item 3: the honest coverage count, mirrored from
    /// [`AnalysisState::frames_received`]/[`AnalysisState::frames_analysed`]
    /// once per cycle — see [`Pipeline::analysis_coverage`] (the
    /// cross-thread reader) and [`fold_run`]'s file-source backpressure
    /// (the cross-thread *writer's* other reader: the compute thread polls
    /// `frames_analysed` to know how far behind it may safely run).
    frames_received: AtomicU64,
    frames_analysed: AtomicU64,
}

impl AnalysisHandle {
    /// A handle with analysis inactive.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Whether the analysis worker is currently doing anything (FR-AU1).
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Turn the analysis worker on/off (the `--analyze` flag's initial
    /// state, and the `I` key thereafter).
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Flip the analysis worker on/off (the `I` key, §2.7).
    pub fn toggle(&self) {
        self.active.fetch_xor(true, Ordering::Relaxed);
    }

    /// The most recently published live track count.
    pub fn track_count(&self) -> usize {
        self.track_count.load(Ordering::Relaxed)
    }

    /// Whether the worker is currently shedding tracks under overload
    /// (NFR-A5).
    pub fn is_shedding(&self) -> bool {
        self.shedding.load(Ordering::Relaxed)
    }

    /// D-125 Amendment 4 item 3: cumulative frames the source has produced
    /// since the worker started or last retuned — see
    /// [`AnalysisState::frames_received`].
    pub fn frames_received(&self) -> u64 {
        self.frames_received.load(Ordering::Relaxed)
    }

    /// D-125 Amendment 4 item 3: cumulative frames actually analysed since
    /// the worker started or last retuned — see
    /// [`AnalysisState::frames_analysed`]. Always `<= frames_received()`.
    pub fn frames_analysed(&self) -> u64 {
        self.frames_analysed.load(Ordering::Relaxed)
    }

    /// Ask the worker thread to stop; [`spawn`]'s returned handle can then be
    /// joined.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Spawn the live analysis worker over `tap`. `retune_gen` mirrors
/// `Pipeline::retune_generation` — the worker resets [`AnalysisState`] and
/// clears `feed` the instant it observes a new generation, in step with the
/// display's own D-056 accumulator reset (AC-9). `rate_known` reports
/// whether the source declared a sample rate (D-098 §7.3), read fresh each
/// cycle since a source's rate is fixed at start but the worker has no other
/// way to learn it.
pub fn spawn<T>(
    tap: T,
    feed: Arc<AnnotationFeed>,
    retune_gen: impl Fn() -> u64 + Send + 'static,
    rate_known: impl Fn() -> bool + Send + 'static,
    handle: Arc<AnalysisHandle>,
) -> JoinHandle<()>
where
    T: SpectrumTap + Send + 'static,
{
    std::thread::Builder::new()
        .name("phosphene-analyze".to_string())
        .spawn(move || run(tap, &feed, retune_gen, rate_known, &handle))
        .expect("failed to spawn the analysis worker thread")
}

fn run(
    tap: impl SpectrumTap,
    feed: &AnnotationFeed,
    retune_gen: impl Fn() -> u64,
    rate_known: impl Fn() -> bool,
    handle: &AnalysisHandle,
) {
    let mut state = AnalysisState::new(&tap.meta());
    let mut view = FrameView::new(tap.meta().bins, TAP_HISTORY_FRAMES);
    let mut seen_gen = retune_gen();

    while !handle.stop.load(Ordering::Relaxed) {
        std::thread::sleep(CYCLE);
        if handle.stop.load(Ordering::Relaxed) {
            break;
        }

        let meta = tap.meta();
        let gen = retune_gen();
        if gen != seen_gen {
            seen_gen = gen;
            state.reset(&meta);
            feed.push_annotations(&[]);
            feed.set_load(AnalysisLoad::default());
            handle.track_count.store(0, Ordering::Relaxed);
            handle.shedding.store(false, Ordering::Relaxed);
            handle.frames_received.store(0, Ordering::Relaxed);
            handle.frames_analysed.store(0, Ordering::Relaxed);
        }

        if !handle.is_active() {
            continue;
        }

        tap.latest_frames(&mut view);
        let (annotations, shedding) = state.cycle(&view, &meta, rate_known());
        handle.shedding.store(shedding, Ordering::Relaxed);
        // D-125 Amendment 4 item 3: same `Relaxed` convention as
        // `track_count`/`shedding` above — the compute thread's
        // file-source backpressure wait (`fold_run`) only ever uses this
        // as an approximate throttling signal (how far behind is analysis,
        // roughly), never to gate access to other shared data, so there is
        // nothing here for a stronger ordering to protect.
        handle
            .frames_received
            .store(state.frames_received(), Ordering::Relaxed);
        handle
            .frames_analysed
            .store(state.frames_analysed(), Ordering::Relaxed);
        feed.set_load(AnalysisLoad {
            shedding,
            frames_received: state.frames_received(),
            frames_analysed: state.frames_analysed(),
        });
        handle
            .track_count
            .store(annotations.len(), Ordering::Relaxed);
        feed.push_annotations(&annotations);
    }
}

/// Map `phosphene-analyze`'s own `Annotation` — a structural mirror kept for
/// that crate's own offline testability (its `annotate` module docs) — onto
/// the `phosphene-render` frozen type `push_annotations` actually takes.
fn convert(a: phosphene_analyze::Annotation) -> phosphene_render::Annotation {
    phosphene_render::Annotation {
        track: phosphene_render::TrackId(a.track.0),
        span: phosphene_render::FreqSpanHz {
            low_hz: a.span.low_hz,
            high_hz: a.span.high_hz,
        },
        anchor_hz: a.anchor_hz,
        label: a.label,
        detail: a.detail,
        confidence: a.confidence,
        strength_db: a.strength_db,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phosphene_analyze::detect::{BlobBuilder, CfarConfig, NoiseFloor, NoiseFloorConfig};
    use phosphene_core::{Complex, SpectrumAnalyzer, WindowKind};
    use phosphene_sources::{
        FileSource, FileSourceConfig, HopOrder, HopperConfig, Input, IqFormat, NoiseConfig,
        ReplayPace, SampleSink, SampleSource, SigGen, SigGenConfig, SinkFlow,
    };

    /// D-125 item 1's GREEN control: the mirror of the RED control quoted
    /// on this lane's pre-fix base commit (0e0d17c), run here on today's
    /// full-coverage code (items 2/3/4/6 all landed). Same scene: a single
    /// isolated hopper dwell (`period_s` far longer than the whole run, so
    /// it never repeats) at the very start of the stream, over by frame 40,
    /// followed by nothing but noise for the rest of the first ~1000-frame
    /// cycle. On 0e0d17c the pre-fix worker's "newest `TAP_HISTORY_FRAMES`
    /// (256) frames only" view never contained the burst at all — frames
    /// 0..40 were long gone from the tail-256 window (frames 744..1000) by
    /// the time `cycle()` ran — so it was structurally invisible, not
    /// merely hard to catch (D-095: deterministically red, not usually
    /// red). Full coverage (item 6) gives every one of this cycle's frames
    /// to the tracker, including the burst, so it must be labelled.
    ///
    /// RED (0e0d17c, isolated worktree, same scene): `any_annotation_ever =
    /// false` over 2 cycles (500 ms) — quoted verbatim, `cargo test
    /// --release -p phosphene-app --features analyze --bin phosphene --
    /// --nocapture scratch_item1_red_control_ook_at_10msps_today`, under the
    /// Amendment 2 cap (peak RSS 598.8 MB, `MemoryMax=10G`), exit 0.
    #[test]
    fn item1_green_control_isolated_burst_is_labelled_on_full_coverage() {
        let rate = 2_048_000.0;
        let fft = 512usize;
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 7;
        cfg.hoppers.push(HopperConfig {
            offsets_hz: vec![300_000.0],
            dwell_s: 0.010,
            period_s: 5.0,
            level_dbfs: -20.0,
            order: HopOrder::Cycle,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);
        let cycle_frames = (CYCLE.as_secs_f64() / meta.frame_dt_s).round() as usize;

        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        let mut any_annotation_ever = false;
        for _ in 0..2 {
            // Full coverage (item 6): every frame of the cycle reaches the
            // view, not just the newest `TAP_HISTORY_FRAMES` of it.
            let mut view = FrameView::new(fft, cycle_frames);
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            let (anns, _shedding) = state.cycle(&view, &meta, true);
            if !anns.is_empty() {
                any_annotation_ever = true;
            }
        }
        assert!(
            any_annotation_ever,
            "GREEN control expected the isolated burst (frames 0..40) to be \
             labelled under full coverage — got none; item 6 no longer \
             gives the tracker every frame of the cycle"
        );
    }

    /// Isolates `AnalysisState::cycle` from the headless GPU/composer loop:
    /// a strong bursty signal, batched the same way headless batches spectra
    /// (many frames per `cycle()` call), must be detected, tracked and
    /// annotated within a modest number of cycles.
    ///
    /// A **continuous** (100% duty) tone at a single fixed bin is the one
    /// documented blind spot of the A0 per-bin noise-floor estimator — its
    /// own bin is indistinguishable from noise at its level by any per-bin
    /// temporal statistic (`detect::noise` module docs; asserted directly by
    /// `detect::seal_tests::robust_floor_is_not_pulled_up_by_strong_occupants`,
    /// whose `floor[tone_bin] > -60.0` assertion is exactly this behavior).
    /// A single-offset "hopper" (on/off at one fixed frequency) is bursty —
    /// below that estimator's ~50%-duty budget — and is what a live signal
    /// with normal on/off structure looks like to it.
    #[test]
    fn cycle_detects_a_strong_bursty_signal_when_batched() {
        let rate = 2_048_000.0;
        let fft = 512usize;
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 1;
        cfg.hoppers.push(HopperConfig {
            offsets_hz: vec![300_000.0],
            dwell_s: 0.01,
            period_s: 0.03,
            level_dbfs: -20.0,
            order: HopOrder::Cycle,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);

        let batch = 100usize;
        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        let mut annotations = Vec::new();
        for _cycle in 0..20 {
            let mut view = FrameView::new(fft, batch);
            for _ in 0..batch {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
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
            "no annotation produced for a strong steady tone after 2000 frames"
        );
        assert!(
            (annotations[0].anchor_hz - 300_000.0).abs() < 5_000.0,
            "annotation anchored at {} Hz, expected near 300 kHz",
            annotations[0].anchor_hz
        );
    }

    /// D-125 Amendment 4 item 3's live-source negative control: "a live
    /// source driven over the limit must report its shed frames in the
    /// count (a test that fails if frames are dropped without being
    /// counted)." Simulates exactly what a bounded `frame_ring` does under
    /// overload — the second `cycle()` call's `view` starts at a sequence
    /// number far past the first call's last one, as it would if 500 frames
    /// had been produced and silently overwritten before the tap was next
    /// polled. `frames_received` must count every one of them regardless;
    /// `frames_analysed` must not.
    #[test]
    fn a_sequence_gap_is_counted_honestly_not_silently_absorbed() {
        let bins = 64;
        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins,
            bin_hz: 16_000.0,
            frame_dt_s: 1.0 / 16_000.0,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);
        let spectrum = vec![-70.0f32; bins];

        // First cycle: ten consecutive frames, no gap yet.
        let mut view = FrameView::new(bins, 10);
        for seq in 0..10u64 {
            view.push(&spectrum, seq);
        }
        let (_, shedding) = state.cycle(&view, &meta, true);
        assert!(!shedding, "no gap yet: the first cycle must not shed");
        assert_eq!(state.frames_received(), 10);
        assert_eq!(state.frames_analysed(), 10);

        // Second cycle: the tap's own bound only retained the newest ten
        // frames of a much longer run — sequence 500..510, not 10..20 — so
        // 490 frames were produced and shed in between, exactly the shape
        // a bounded, overwrite-on-full `frame_ring` (NFR-A5) produces under
        // real overload.
        let mut view2 = FrameView::new(bins, 10);
        for seq in 500..510u64 {
            view2.push(&spectrum, seq);
        }
        let (_, shedding) = state.cycle(&view2, &meta, true);
        assert!(shedding, "a real sequence gap must report shedding");
        assert_eq!(
            state.frames_received(),
            10 + 500,
            "the shed frames must still be counted as received — a silent \
             drop would leave this at 20, not 510"
        );
        assert_eq!(
            state.frames_analysed(),
            10 + 10,
            "only the frames actually folded into the tracker count as analysed"
        );
    }

    /// D-125 item 4, re-wired on full coverage (item 6): previously only
    /// tracks alive at a cycle's *last* frame were published
    /// (`AnalysisState::cycle` read `tracker.tracks()` alone, never
    /// `retired()`). Here every analysis cycle spans exactly one hop
    /// period (30 ms), and the coast allowance (2 frames, 500 us at this
    /// rate) is far shorter than the 20 ms off time, so by construction the
    /// burst has always died before each cycle's last frame — the pre-fix
    /// engine would report no annotation from any cycle in this test, ever.
    #[test]
    fn a_track_confirmed_during_a_cycle_is_published_even_if_it_dies_within_it() {
        let rate = 2_048_000.0;
        let fft = 512usize;
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 3;
        cfg.hoppers.push(HopperConfig {
            offsets_hz: vec![250_000.0],
            dwell_s: 0.010,
            period_s: 0.030,
            level_dbfs: -20.0,
            order: HopOrder::Cycle,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);

        let period_frames = (0.030 / meta.frame_dt_s).round() as usize;
        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        let mut saw_annotation_from_a_dead_end_cycle = false;
        for _ in 0..8 {
            let mut view = FrameView::new(fft, period_frames);
            for _ in 0..period_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            let (anns, _shedding) = state.cycle(&view, &meta, true);
            if !anns.is_empty() {
                saw_annotation_from_a_dead_end_cycle = true;
                assert!(
                    (anns[0].anchor_hz - 250_000.0).abs() < 5_000.0,
                    "annotation anchored at {} Hz, expected near 250 kHz",
                    anns[0].anchor_hz
                );
            }
        }
        assert!(
            saw_annotation_from_a_dead_end_cycle,
            "no cycle published a track, though every cycle's last frame is \
             always mid-gap (the burst is always dead by then) — a track \
             confirmed and retired within one cycle must still be published"
        );
    }

    /// D-125 item 2's measurement: how long does one `cycle()` call take to
    /// fold every raw frame of a 250 ms `CYCLE` at 10 MS/s with FFT 1024,
    /// literally, no reduction — the simplest possible reading of "every
    /// frame analysed"?
    ///
    /// **Finding, not a fix (D-125 item 2's stop rule).** Measured on the
    /// lane host (a modest, shared CPU — see the AI-3 brief's Amendment 2),
    /// `cargo test --release`, 10 runs after 4 warm-up cycles:
    /// ```text
    /// cycle_frames = 2441
    /// cycle 0: 2595.315 ms   cycle 5: 2624.090 ms
    /// cycle 1: 2625.699 ms   cycle 6: 2625.430 ms
    /// cycle 2: 2595.518 ms   cycle 7: 2614.087 ms
    /// cycle 3: 2625.541 ms   cycle 8: 2623.414 ms
    /// cycle 4: 2592.239 ms   cycle 9: 2623.122 ms
    /// max: 2625.699 ms of 250 ms budget
    /// ```
    /// About **10.5x** the 250 ms `CYCLE` budget, even optimized — not a
    /// debug-build artifact (a debug-profile run of the same probe did not
    /// finish even one cycle inside 5 minutes, confirming the release
    /// number is the real one). Per item 2: *"If it exceeds the cycle on
    /// the lane host, STOP and tell the master. It is a performance
    /// finding, not a failure to hide."* This lane therefore does **not**
    /// wire the live worker's tap (`crate::analyze::TAP_HISTORY_FRAMES`,
    /// `PipelineSpectrumTap`) to this literal every-raw-frame design, and
    /// does not switch the lockstep capture
    /// (`headless::run_file_lockstep_frames`) or the goldens to it either —
    /// both would bake in the same design this measurement shows does not
    /// fit here. A cheaper design that still analyses every frame (the
    /// brief's own example: "a max-reduced analysis stream at a fixed time
    /// resolution") is a real option, but choosing and building one is a
    /// design decision for the master, not this lane to invent unilaterally
    /// after finding the naive approach does not fit — see the PR body /
    /// the branch's own stop report.
    #[test]
    #[ignore = "slow: a direct measurement of literal full-frame coverage against the 250ms NFR-A1 budget, not a fast unit test"]
    fn full_coverage_analysis_time_exceeds_the_cycle_budget_on_the_lane_host() {
        let rate = 10_000_000.0;
        let fft = 1024usize;
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 7;
        cfg.ook.push(phosphene_sources::siggen::OokConfig {
            offset_hz: 300_000.0,
            symbol_rate_bd: 2000.0,
            level_dbfs: -20.0,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);
        let cycle_frames = (CYCLE.as_secs_f64() / meta.frame_dt_s).round() as usize;
        eprintln!("cycle_frames = {cycle_frames}");

        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        // Warm up a few cycles first (noise floor spin-up), then time.
        for _ in 0..4 {
            let mut view = FrameView::new(fft, cycle_frames);
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            state.cycle(&view, &meta, true);
        }
        let mut times = Vec::new();
        for _ in 0..10 {
            let mut view = FrameView::new(fft, cycle_frames);
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            let started = std::time::Instant::now();
            state.cycle(&view, &meta, true);
            times.push(started.elapsed());
        }
        for (i, t) in times.iter().enumerate() {
            eprintln!("cycle {i}: {:.3} ms", t.as_secs_f64() * 1000.0);
        }
        let max = times.iter().max().unwrap();
        eprintln!("max: {:.3} ms of 250 ms budget", max.as_secs_f64() * 1000.0);
        // Pins the finding rather than only printing it: literal full-frame
        // coverage exceeds `CYCLE` here. If this one day stops failing (a
        // faster host, or an optimization), that is itself news — the
        // stop-and-surface call above should be revisited, not silently
        // dropped by deleting the assertion.
        assert!(
            *max > CYCLE,
            "expected literal full-frame coverage to exceed the {CYCLE:?} budget on this host \
             (D-125 item 2's stop condition) — it did not ({max:?}); the finding this test \
             records may no longer hold and should be re-reported"
        );
    }

    /// PM-requested profiling pass (bounded): where does the ~1.07 ms/frame
    /// this lane measured for full-frame coverage go? `perf`/flamegraph are
    /// unavailable on the lane host (`perf_event_paranoid` is 4 — no
    /// CAP_PERFMON for an unprivileged `systemd-run --user` scope, and
    /// raising it is a shared-host kernel setting this lane does not touch
    /// unilaterally), so this uses stage timers instead, exactly as offered.
    ///
    /// Reproduces `AnalysisState::cycle`'s exact per-frame call sequence
    /// (`Detector::push_frame`'s own body: `noise.push_frame` →
    /// `cfar.detect` → `blobs.push_frame`, then `Tracker::observe`) by
    /// driving the same public sub-stage types directly — `NoiseFloor`,
    /// `Cfar`, `BlobBuilder` are all `pub` and re-exported from
    /// `phosphene_analyze::detect` for exactly this kind of external
    /// measurement — rather than adding any timing to production code.
    /// Nothing here changes behaviour or coverage; it only measures the
    /// existing literal full-frame design at 7f20eb5.
    ///
    /// **Result, lane host, `cargo test --release`, 14 cycles (4 warm-up +
    /// 10 measured, 24,410 frames measured) of the same 10 MS/s / FFT 1024
    /// OOK scene the item 2 measurement used:**
    /// ```text
    /// noise.push_frame:  847.621 us/frame (79.6%)
    /// cfar.detect:       213.471 us/frame (20.1%)
    /// blobs.push_frame:    1.918 us/frame ( 0.2%)
    /// tracker.observe:     1.514 us/frame ( 0.1%)
    /// sum:              1064.524 us/frame
    /// ```
    /// Matches the item 2 measurement's own per-frame rate closely (2625.7ms
    /// / 2441 frames = 1075.7 us/frame there; 1064.5 us/frame here — the
    /// ~1% gap is this test's narrower per-frame timing not including the
    /// per-cycle `Detector::finish()` flush or the measure/annotate tail
    /// `AnalysisState::cycle` also does once per cycle, not per frame).
    ///
    /// **The noise floor estimator (`NoiseFloor::push_frame`,
    /// `detect/noise.rs`) is essentially the whole cost, at ~80%; CFAR is
    /// most of the rest, at ~20%; blob-building and tracking are noise (well
    /// under 1% combined).** Reading why, from that function's own code
    /// (not touched here — this is a read, not a change): for every one of
    /// the frame's `bins` bins it copies that bin's whole sliding window
    /// (`window_frames`, 64 by default) into a scratch buffer and
    /// `sort_unstable_by`s it, every frame — O(bins x window x
    /// log(window)) per frame, here 1024 x 64 x log2(64) ~ 390K comparisons,
    /// once per frame, 24,410 times in this run.
    #[test]
    #[ignore = "slow: a multi-minute stage-timing measurement, not a fast unit test"]
    fn full_coverage_per_stage_timing_breakdown() {
        use phosphene_analyze::detect::{BlobBuilder, Cfar, NoiseFloor};

        let rate = 10_000_000.0;
        let fft = 1024usize;
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 7;
        cfg.ook.push(phosphene_sources::siggen::OokConfig {
            offset_hz: 300_000.0,
            symbol_rate_bd: 2000.0,
            level_dbfs: -20.0,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let cycle_frames = (CYCLE.as_secs_f64() / meta.frame_dt_s).round() as usize;

        let det_cfg = DetectorConfig::new();
        let mut noise = NoiseFloor::new(fft, det_cfg.noise.clone()).unwrap();
        let mut cfar = Cfar::new(fft, det_cfg.cfar.clone()).unwrap();
        let mut blobs = BlobBuilder::new(fft, det_cfg.min_blob_cells).unwrap();
        let mut mask = vec![false; fft];
        let mut tracker = Tracker::new(TrackerConfig::default()).unwrap();

        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        let (mut noise_ns, mut cfar_ns, mut blob_ns, mut track_ns) = (0u128, 0u128, 0u128, 0u128);
        let mut frames_measured = 0u64;
        // 4 warm-up cycles (noise-floor spin-up, same as the sibling timing
        // probe), unmeasured, then 10 measured cycles.
        for pass in 0..14 {
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                if pass >= 4 {
                    let t0 = std::time::Instant::now();
                    noise.push_frame(&spectrum);
                    noise_ns += t0.elapsed().as_nanos();

                    let t1 = std::time::Instant::now();
                    cfar.detect(&spectrum, noise.floor_dbfs(), &mut mask);
                    cfar_ns += t1.elapsed().as_nanos();

                    let t2 = std::time::Instant::now();
                    let dets = blobs.push_frame(&spectrum, &mask, noise.floor_dbfs(), seq);
                    blob_ns += t2.elapsed().as_nanos();

                    let t3 = std::time::Instant::now();
                    tracker.observe(seq, &dets, &meta);
                    track_ns += t3.elapsed().as_nanos();

                    frames_measured += 1;
                } else {
                    noise.push_frame(&spectrum);
                    cfar.detect(&spectrum, noise.floor_dbfs(), &mut mask);
                    let dets = blobs.push_frame(&spectrum, &mask, noise.floor_dbfs(), seq);
                    tracker.observe(seq, &dets, &meta);
                }
                seq += 1;
            }
        }

        let per_frame = |ns: u128| ns as f64 / frames_measured as f64 / 1000.0; // us/frame
        let total_us =
            per_frame(noise_ns) + per_frame(cfar_ns) + per_frame(blob_ns) + per_frame(track_ns);
        eprintln!("frames measured: {frames_measured}");
        eprintln!(
            "noise.push_frame:  {:.3} us/frame ({:.1}%)",
            per_frame(noise_ns),
            100.0 * per_frame(noise_ns) / total_us
        );
        eprintln!(
            "cfar.detect:       {:.3} us/frame ({:.1}%)",
            per_frame(cfar_ns),
            100.0 * per_frame(cfar_ns) / total_us
        );
        eprintln!(
            "blobs.push_frame:  {:.3} us/frame ({:.1}%)",
            per_frame(blob_ns),
            100.0 * per_frame(blob_ns) / total_us
        );
        eprintln!(
            "tracker.observe:   {:.3} us/frame ({:.1}%)",
            per_frame(track_ns),
            100.0 * per_frame(track_ns) / total_us
        );
        eprintln!("sum:               {total_us:.3} us/frame");
    }

    // D-125 Amendment 4 item 1: `optimised_full_coverage_analysis_time_at_
    // 10_and_20_msps` and its `measure_optimised_cycle` helper (the
    // Amendment 3 single-thread, cached-CFAR measurement — 781.312 ms at
    // 10 MS/s, 1597.835 ms at 20 MS/s, reuse fraction 0.0060 / 0.0000, on
    // the lane host's i7-1360P) are removed here, with the cache itself:
    // their own numbers described code that no longer exists once the
    // cache is gone, so keeping the test would either silently start
    // measuring something else under the old name, or need `#[ignore]`d
    // dead weight. The commit that removed them (search this file's git
    // log for "Amendment 4 item 1") is the numbers' permanent home; D-125's
    // own decisions.md entry (main 9bd2533) already carries them too. A new
    // measurement, under the parallel design, is
    // `parallel_full_coverage_analysis_time_at_10_and_20_msps` below.

    fn measure_parallel_cycle(rate: f64, fft: usize) -> Vec<std::time::Duration> {
        let mut cfg = SigGenConfig::new(rate);
        cfg.seed = 7;
        cfg.ook.push(phosphene_sources::siggen::OokConfig {
            offset_hz: 300_000.0,
            symbol_rate_bd: 2000.0,
            level_dbfs: -20.0,
        });
        cfg.noise = Some(NoiseConfig { level_dbfs: -70.0 });
        let mut gen = SigGen::new(cfg).unwrap();
        let mut analyzer = SpectrumAnalyzer::new(fft, WindowKind::Hann).unwrap();

        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: rate,
            bins: fft,
            bin_hz: rate / fft as f64,
            frame_dt_s: fft as f64 / rate,
            cal_offset_db: 0.0,
        };
        let mut state = AnalysisState::new(&meta);
        let cycle_frames = (CYCLE.as_secs_f64() / meta.frame_dt_s).round() as usize;

        let mut chunk = vec![Complex::new(0.0f32, 0.0); fft];
        let mut spectrum = vec![0.0f32; fft];
        let mut seq = 0u64;
        for _ in 0..4 {
            let mut view = FrameView::new(fft, cycle_frames);
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            state.cycle(&view, &meta, true);
        }
        let mut times = Vec::new();
        for _ in 0..10 {
            let mut view = FrameView::new(fft, cycle_frames);
            for _ in 0..cycle_frames {
                gen.fill(&mut chunk);
                analyzer.process(&chunk, &mut spectrum);
                view.push(&spectrum, seq);
                seq += 1;
            }
            let started = std::time::Instant::now();
            state.cycle(&view, &meta, true);
            times.push(started.elapsed());
        }
        times
    }

    /// D-125 Amendment 4 item 2: re-measured, on the parallel per-bin
    /// design (items 1-2), same host, same scene and method as the
    /// Amendment 3 measurement this replaces (see the removal note above).
    ///
    /// **Result, lane host (13th-generation Intel Core i7-1360P, 12 cores /
    /// 16 threads), `cargo test --release`, rayon's global pool at its
    /// default size (`std::thread::available_parallelism()`, 16 on this
    /// host), 14 cycles (4 warm-up + 10 measured) per rate, under the
    /// Amendment 2 cap:**
    /// ```text
    /// --- 10 MS/s (FFT 1024), cycle_frames = 2441, threads = 16 ---
    /// max: 366.326 ms of 250 ms budget (7 of 10 cycles under budget)
    /// --- 20 MS/s (FFT 1024), cycle_frames = 4883, threads = 16 ---
    /// max: 610.949 ms of 250 ms budget
    /// ```
    /// ~2.1x faster than single-thread at 10 MS/s (781.312ms, commit
    /// eeed0af) and ~2.6x at 20 MS/s (1597.835ms) — well short of a 16x
    /// speedup from 16 threads (Amdahl: CFAR's and the noise floor's
    /// per-frame parallel regions each pay rayon dispatch overhead, and a
    /// 12-core/16-thread mobile CPU's hyperthreads share execution units,
    /// so 16-way numeric work does not scale linearly). 10 MS/s now mostly
    /// *keeps up* (7 of 10 measured cycles under the 250ms budget; the
    /// worst, 366ms, is 1.5x over). 20 MS/s stays clearly over (1.5x-2.4x).
    /// Per item 5: this is a real-time limit, not a core count — the
    /// measured figure the PM carries into D-125/NFR-A1/the README replaces
    /// the prior single-thread ~3 MS/s estimate with "keeps up to roughly
    /// 10 MS/s on this machine class, on multiple threads."
    #[test]
    #[ignore = "slow: a multi-minute release-build measurement, not a fast unit test"]
    fn parallel_full_coverage_analysis_time_at_10_and_20_msps() {
        // Mirrors `NoiseFloor::push_frame`/`Cfar::detect`'s own
        // `rayon::current_num_threads()` call: at FFT 1024 bins, far more
        // than any real thread count, the pool never has more work than
        // threads to give it, so this reports exactly what `AnalysisState::
        // cycle` below actually used.
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        for &rate in &[10_000_000.0, 20_000_000.0] {
            let times = measure_parallel_cycle(rate, 1024);
            let cycle_frames = (CYCLE.as_secs_f64() / (1024.0 / rate)).round() as usize;
            eprintln!(
                "--- {:.0} MS/s (FFT 1024), cycle_frames = {cycle_frames}, threads = {threads} ---",
                rate / 1e6
            );
            for (i, t) in times.iter().enumerate() {
                eprintln!("cycle {i}: {:.3} ms", t.as_secs_f64() * 1000.0);
            }
            let max = times.iter().max().unwrap();
            eprintln!("max: {:.3} ms of 250 ms budget", max.as_secs_f64() * 1000.0);
        }
    }

    // ---- AN-1/AN-3 (D-126): burst sensitivity and the false-alarm rate,
    // measured together on the real 913 MHz capture. -------------------

    /// A wall-clock-free reader over a real IQ capture (mirrors
    /// `crate::headless::FileFeed`'s approach, which is private to that
    /// module — this lane's territory is `analyze.rs`, not `headless.rs`, so
    /// this is its own small copy of the same idea: pull whole FFT frames
    /// out of `FileSource` with [`ReplayPace::Unthrottled`] and compute each
    /// one's dB power spectrum).
    struct CaptureFrames {
        source: FileSource,
        analyzer: SpectrumAnalyzer,
        fft_size: usize,
        pending: Vec<Complex<f32>>,
        spectrum: Vec<f32>,
        ended: bool,
    }

    struct CollectSink<'a> {
        buf: &'a mut Vec<Complex<f32>>,
        want: usize,
    }

    impl SampleSink for CollectSink<'_> {
        fn push(&mut self, samples: &[Complex<f32>]) -> SinkFlow {
            self.buf.extend_from_slice(samples);
            if self.buf.len() >= self.want {
                SinkFlow::Stop
            } else {
                SinkFlow::Continue
            }
        }
    }

    impl CaptureFrames {
        fn open(path: &std::path::Path, fft_size: usize, rate_hz: f64) -> Self {
            let cfg = FileSourceConfig {
                input: Input::Path(path.to_path_buf()),
                format: IqFormat::Cs16,
                sample_rate_hz: Some(rate_hz),
                center_freq_hz: None,
                pace: ReplayPace::Unthrottled,
                loop_replay: false,
            };
            let source = FileSource::new(cfg).expect("open the real 913 MHz capture");
            let analyzer =
                SpectrumAnalyzer::new(fft_size, WindowKind::Hann).expect("valid FFT size");
            CaptureFrames {
                source,
                analyzer,
                fft_size,
                pending: Vec::new(),
                spectrum: vec![0.0; fft_size],
                ended: false,
            }
        }

        /// The next whole FFT frame's dB power spectrum, or `None` once the
        /// file is exhausted (a trailing partial frame is discarded, same as
        /// `FileSource`'s own sub-sample truncation contract).
        fn next(&mut self) -> Option<&[f32]> {
            if self.pending.len() < self.fft_size && !self.ended {
                let want = self.fft_size;
                let mut sink = CollectSink {
                    buf: &mut self.pending,
                    want,
                };
                self.source.stream(&mut sink).expect("capture read failed");
                if self.pending.len() < want {
                    self.ended = true;
                }
            }
            if self.pending.len() < self.fft_size {
                return None;
            }
            let frame: Vec<Complex<f32>> = self.pending.drain(..self.fft_size).collect();
            self.analyzer.process(&frame, &mut self.spectrum);
            Some(&self.spectrum)
        }
    }

    /// Ground-truth occupancy margin (dB over the per-bin noise floor) for
    /// this lane's seal (D-126): the instrument that decides "a burst is
    /// present" is independent of the detector configuration under test —
    /// it reuses the same [`NoiseFloor`] estimator (its whole job is
    /// estimating the mean per-bin noise power, exactly what a ground-truth
    /// threshold needs) but its own decision margin sits well above CFAR's
    /// own boundary: `pfa` 1e-3's single-bin threshold is
    /// `10·log10(-ln(1e-3))` ≈ 8.4 dB over the floor, so marking ground
    /// truth at 15 dB with at least [`GT_MIN_BINS`] bins at once cannot
    /// simply reproduce the CFAR/tracker decision the candidates below are
    /// tuning. The same method, unchanged, scores every row of the table.
    const GT_MARGIN_DB: f32 = 15.0;
    /// Bins that must clear [`GT_MARGIN_DB`] **contiguously, within one
    /// frame** for that run to count toward a burst — several, so a single
    /// noise spike (or the odd leftover DC skirt bin) cannot mark a
    /// ground-truth burst on its own. Amendment 2: this now gates a
    /// per-frame, per-run width (see [`filter_runs_by_width`]) rather than a
    /// whole-frame bin count, so two emitters occupying the same frame are
    /// judged independently instead of pooled into one span.
    const GT_MIN_BINS: usize = 3;
    /// Frames a ground-truth gap may span and still belong to the same
    /// burst — about 1 ms at this capture's ~205 µs frame time, enough to
    /// bridge a brief fade without merging genuinely separate bursts.
    const GT_GAP_FRAMES: u64 = 5;
    /// A ground-truth burst this short (in whole frames) is exactly what
    /// D-126 measured the pre-fix `birth_m: 3` M-of-N rule as structurally
    /// unable to confirm, at any frame rate: a delivered blob is credited
    /// its own whole frame span at once (`Tracker::observe`'s own doc), so a
    /// blob spanning fewer than `birth_m` frames can never reach M hits by
    /// itself. `birth_m: 3` (this seal's baseline row) needs 3 frames — this
    /// is AN-1's own cutoff, not a separate guess, and review 1's blocker 2:
    /// the aggregate recall number can improve entirely on bursts already
    /// long enough to confirm under the old rule, so AN-1's actual claim —
    /// short bursts specifically — needs its own measurement.
    const SHORT_BURST_MAX_FRAMES: u64 = 2;

    /// Zero out every maximal run of `true` bins shorter than `min_width` —
    /// [`GT_MIN_BINS`] applied per contiguous run instead of per whole
    /// frame (Amendment 2), so two simultaneous emitters in the same frame
    /// are each judged on their own run's width rather than pooled into one
    /// frame-wide bin count.
    fn filter_runs_by_width(mask: &mut [bool], min_width: usize) {
        let n = mask.len();
        let mut k = 0;
        while k < n {
            if mask[k] {
                let lo = k;
                while k < n && mask[k] {
                    k += 1;
                }
                if k - lo < min_width {
                    for b in &mut mask[lo..k] {
                        *b = false;
                    }
                }
            } else {
                k += 1;
            }
        }
    }

    /// A ground-truth burst with both a time extent and a frequency extent
    /// (Amendment 2: the coupled oracle's core defect was associating
    /// ground truth to tracks by time alone, so an unrelated-frequency
    /// track could score a hit and dodge the false-alarm count at once).
    #[derive(Debug, Clone, Copy)]
    struct GtBurst {
        frame_lo: u64,
        frame_hi: u64,
        bin_lo: usize,
        bin_hi: usize,
    }

    impl GtBurst {
        /// Frequency edges in Hz, on the same `(bin - 0.5, bin + 0.5)`
        /// convention [`phosphene_analyze::detect::Detection::offset_span_hz`]
        /// uses, so a burst and a detection agree on what "the edge" means.
        fn freq_span_hz(&self, meta: &BandMeta) -> (f64, f64) {
            (
                meta.bin_offset_hz(self.bin_lo as f64 - 0.5),
                meta.bin_offset_hz(self.bin_hi as f64 + 0.5),
            )
        }
    }

    /// Two raw ground-truth blobs belong to the same physical burst only if
    /// they are close in time (within [`GT_GAP_FRAMES`], bridging a brief
    /// fade — unchanged value, Amendment 2) **and** agree in frequency
    /// within `tol_hz` (the derived association tolerance): a time-only
    /// bridge is exactly the defect Amendment 2 exists to remove, so both
    /// conditions are required, never either alone.
    fn gt_mergeable(
        a: &GtBurst,
        b: &GtBurst,
        gap_frames: u64,
        meta: &BandMeta,
        tol_hz: f64,
    ) -> bool {
        let time_ok =
            a.frame_hi + gap_frames >= b.frame_lo && b.frame_hi + gap_frames >= a.frame_lo;
        if !time_ok {
            return false;
        }
        let (a_lo, a_hi) = a.freq_span_hz(meta);
        let (b_lo, b_hi) = b.freq_span_hz(meta);
        a_lo - tol_hz <= b_hi && b_lo - tol_hz <= a_hi
    }

    /// Union-find `find`, with path compression, over a plain index parent
    /// array — merging raw ground-truth blobs into bursts is small enough
    /// (tens of blobs) that this need not be fancy, just correct and
    /// transitive (A merges with B, B with C: A and C end up together even
    /// when not directly [`gt_mergeable`]).
    fn uf_find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    fn uf_union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (uf_find(parent, a), uf_find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    /// A published track's own frequency extent, on the same convention as
    /// [`GtBurst::freq_span_hz`]: centre ± half the smoothed bandwidth.
    #[derive(Debug, Clone, Copy)]
    struct Published {
        born: u64,
        last_seen: u64,
        lo_hz: f64,
        hi_hz: f64,
    }

    fn time_overlaps(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> bool {
        a_lo <= b_hi && a_hi >= b_lo
    }

    fn freq_overlaps(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64, tol_hz: f64) -> bool {
        a_lo - tol_hz <= b_hi && b_lo - tol_hz <= a_hi
    }

    /// Amendment 2's association, factored out of the end-to-end test so
    /// Amendment 3's direct oracle test (round 7's blocker 1: the
    /// association was not load-bearing — forcing frequency overlap to
    /// always-true left the coupled seal green) can exercise it on
    /// constructed inputs, cheaply and deterministically, without the real
    /// capture. A burst is recalled only when some published track
    /// overlaps it in TIME **and** FREQUENCY (within `tol_hz`, the derived
    /// association tolerance).
    fn gt_recalled(gt: &GtBurst, published: &[Published], meta: &BandMeta, tol_hz: f64) -> bool {
        let (gt_lo, gt_hi) = gt.freq_span_hz(meta);
        published.iter().any(|p| {
            time_overlaps(p.born, p.last_seen, gt.frame_lo, gt.frame_hi)
                && freq_overlaps(p.lo_hz, p.hi_hz, gt_lo, gt_hi, tol_hz)
        })
    }

    /// The false-alarm half of Amendment 2's association (see
    /// [`gt_recalled`]): a published track is a false alarm unless some
    /// ground-truth burst contains both its birth frame **and** overlaps it
    /// in frequency — a track born inside a burst's time window but at an
    /// unrelated frequency is still a false alarm.
    fn track_is_false_alarm(
        p: &Published,
        gt_bursts: &[GtBurst],
        meta: &BandMeta,
        tol_hz: f64,
    ) -> bool {
        !gt_bursts.iter().any(|gt| {
            let (gt_lo, gt_hi) = gt.freq_span_hz(meta);
            p.born >= gt.frame_lo
                && p.born <= gt.frame_hi
                && freq_overlaps(p.lo_hz, p.hi_hz, gt_lo, gt_hi, tol_hz)
        })
    }

    /// Amendment 3 blocker 1: a direct, cheap, deterministic proof that the
    /// frequency association in [`gt_recalled`]/[`track_is_false_alarm`] is
    /// load-bearing — round 7's mandatory negative control (forcing the
    /// frequency test to always-true, recreating round 6's time-only
    /// association) left the coupled seal green (1 passed, 0 failed), which
    /// proved the end-to-end comparison alone cannot detect the
    /// association's removal: baseline, candidate and loosened all run
    /// through the same oracle, so its behaviour largely cancels out of a
    /// baseline-vs-candidate comparison. This test fails the moment the
    /// frequency test is disabled or inverted, on inputs built here, no
    /// capture needed.
    #[test]
    fn frequency_association_is_load_bearing_not_decorative() {
        let meta = BandMeta {
            center_freq_hz: None,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 1.0e-3,
            cal_offset_db: 0.0,
        };
        let tol_hz = 3000.0; // TrackerConfig::default().gate_min_bins (3.0) * bin_hz (1000.0)
        let burst = GtBurst {
            frame_lo: 10,
            frame_hi: 20,
            bin_lo: 500,
            bin_hi: 510,
        };

        // A track overlapping the burst in TIME (born..=last_seen 12..=18
        // sits inside 10..=20) but centred 100 bins (100,000 Hz) away in
        // FREQUENCY — far past `tol_hz`.
        let far_track = Published {
            born: 12,
            last_seen: 18,
            lo_hz: 95_000.0,
            hi_hz: 105_000.0,
        };
        assert!(
            !gt_recalled(&burst, std::slice::from_ref(&far_track), &meta, tol_hz),
            "a track overlapping in time but far in frequency must not count as recall"
        );
        assert!(
            track_is_false_alarm(&far_track, std::slice::from_ref(&burst), &meta, tol_hz),
            "a track overlapping in time but far in frequency must count as a false alarm"
        );

        // Positive control: the same time window, moved onto the burst's
        // own frequency, DOES recall it and is NOT a false alarm — proves
        // these assertions are not simply always-false/always-true.
        let (gt_lo, gt_hi) = burst.freq_span_hz(&meta);
        let near_track = Published {
            born: 12,
            last_seen: 18,
            lo_hz: gt_lo,
            hi_hz: gt_hi,
        };
        assert!(
            gt_recalled(&burst, std::slice::from_ref(&near_track), &meta, tol_hz),
            "a track overlapping in both time and frequency must count as recall"
        );
        assert!(
            !track_is_false_alarm(&near_track, std::slice::from_ref(&burst), &meta, tol_hz),
            "a track overlapping in both time and frequency must not count as a false alarm"
        );
    }

    /// Total frames covered by the union of `intervals` (each `(lo, hi)`
    /// inclusive) — Amendment 3 blocker 2: occupied time must be the union
    /// of ground-truth burst frame spans, not their sum, since two
    /// frequency-separated bursts (Amendment 2's own splitting requirement)
    /// can share frames, and summing double-counts that shared time.
    /// Overlapping and touching-but-disjoint spans both merge into one run
    /// before being counted, so shared or adjacent frames are counted
    /// exactly once either way.
    fn union_frame_count(intervals: &[(u64, u64)]) -> u64 {
        let mut sorted: Vec<(u64, u64)> = intervals.to_vec();
        sorted.sort_by_key(|&(lo, _)| lo);
        let mut total = 0u64;
        let mut cur: Option<(u64, u64)> = None;
        for (lo, hi) in sorted {
            cur = Some(match cur {
                None => (lo, hi),
                Some((clo, chi)) if lo <= chi + 1 => (clo, chi.max(hi)),
                Some((clo, chi)) => {
                    total += chi - clo + 1;
                    (lo, hi)
                }
            });
        }
        if let Some((clo, chi)) = cur {
            total += chi - clo + 1;
        }
        total
    }

    /// Amendment 4 blocker: restoring the prior summed-duration calculation
    /// produced the old (wrong) rates and the real-capture test **still
    /// passed 1/1** — the union correction was not regression-protected,
    /// exactly Amendment 3's blocker 1 lesson unapplied to the other half.
    /// This is the same shape as
    /// [`frequency_association_is_load_bearing_not_decorative`]: a fast,
    /// deterministic, non-`#[ignore]`d test on constructed intervals, with
    /// a positive control (disjoint spans sum normally) so the assertion
    /// cannot pass vacuously, that fails the moment [`union_frame_count`]
    /// is replaced by a plain sum.
    #[test]
    fn union_frame_count_counts_shared_frames_once() {
        // Positive control: two spans with a real gap between them (no
        // frame in common, not even touching) — the union equals the naive
        // sum, proving this test does not just always expect "less than
        // the sum" regardless of input.
        let disjoint = [(0u64, 4u64), (10u64, 14u64)];
        let disjoint_sum: u64 = disjoint.iter().map(|&(lo, hi)| hi - lo + 1).sum();
        assert_eq!(
            union_frame_count(&disjoint),
            disjoint_sum,
            "disjoint spans with a real gap must sum normally"
        );
        assert_eq!(disjoint_sum, 10, "sanity: 5 + 5 frames");

        // Touching (adjacent, zero-frame gap) spans: still no frame in
        // common, so the union also equals the naive sum here — but a
        // union computed as "sum minus overlap" with an off-by-one at the
        // boundary would get this wrong in either direction, so it is
        // asserted on its own rather than assumed to follow from the
        // disjoint case above.
        let touching = [(0u64, 4u64), (5u64, 9u64)];
        assert_eq!(
            union_frame_count(&touching),
            10,
            "touching, non-overlapping spans (frame 4 then frame 5) must total 10, not double-count \
             or drop the boundary"
        );

        // Overlapping spans: frames 5..=10 belong to both — summing double
        // counts that shared range (11 + 11 = 22); the union must count it
        // once (0..=15 = 16 frames).
        let overlapping = [(0u64, 10u64), (5u64, 15u64)];
        let overlapping_sum: u64 = overlapping.iter().map(|&(lo, hi)| hi - lo + 1).sum();
        assert_eq!(overlapping_sum, 22, "sanity: 11 + 11 frames");
        assert_eq!(
            union_frame_count(&overlapping),
            16,
            "overlapping spans must contribute their union once, not their sum"
        );
        assert!(
            union_frame_count(&overlapping) < overlapping_sum,
            "the union of overlapping spans must be strictly less than their sum — if this ever \
             holds with equality, union_frame_count has regressed to a plain sum"
        );

        // Spans sharing exactly one boundary frame (frame 5 belongs to
        // both): the classic off-by-one case a naive "sum minus overlap"
        // implementation gets wrong at the edges.
        let shared_boundary = [(0u64, 5u64), (5u64, 10u64)];
        assert_eq!(
            union_frame_count(&shared_boundary),
            11,
            "spans sharing exactly one boundary frame must count it once (0..=10 = 11 frames)"
        );
    }

    /// Find the real 913 MHz capture (D-126's scene, never committed to this
    /// public repo — 600 MB of a real recording). Review 2's blocker 2:
    /// different hosts keep it at different paths (the lane host under
    /// `~/captures/`, the review host directly under `~`), so this tries
    /// every known layout — `$PHOSPHENE_AN1_AN3_CAPTURE` first if set (an
    /// explicit override, tried alone, for any other host), then
    /// `$HOME/captures/cap_913MHz_5Msps.sc16`, then
    /// `$HOME/cap_913MHz_5Msps.sc16`.
    ///
    /// Returns `None`, after printing every path it tried, rather than
    /// panicking: the registry seal runs this test for **every** phosphene
    /// lane (round 4's finding), and almost none of them have anything to do
    /// with this specific capture, so a host without it must skip this one
    /// test cleanly, not redden the fleet-wide seal. The named-paths
    /// diagnostic stays — only the "and now fail" part changes.
    fn locate_capture() -> Option<std::path::PathBuf> {
        let mut tried: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(over) = std::env::var("PHOSPHENE_AN1_AN3_CAPTURE") {
            let p = std::path::PathBuf::from(over);
            if p.exists() {
                return Some(p);
            }
            tried.push(p);
        } else if let Ok(home) = std::env::var("HOME") {
            for rel in ["captures/cap_913MHz_5Msps.sc16", "cap_913MHz_5Msps.sc16"] {
                let p = std::path::PathBuf::from(&home).join(rel);
                if p.exists() {
                    return Some(p);
                }
                tried.push(p);
            }
        }
        println!(
            "SKIP an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture: the real 913 \
             MHz capture (D-126's scene, never substituted with a synthetic one) was not found \
             at any of the paths tried: {} — set PHOSPHENE_AN1_AN3_CAPTURE to its exact path, or \
             place it at one of those paths, to run this lane's coupled seal on this host",
            tried
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        None
    }

    /// AN-1/AN-3's ⛔ coupled seal (D-126, the federator's 2026-09-16 ruling
    /// at the top of the lane doc): burst recall and the false-alarm rate,
    /// asserted TOGETHER, on the same real 913 MHz capture, in the same run
    /// — never in separate seals that could each pass while the pair
    /// regresses. Three configurations are scored in one pass over the same
    /// frames against the same ground truth:
    ///
    /// * **baseline** — the pre-fix defaults this lane found (D-126):
    ///   `birth_m` 3, `min_blob_cells` 1, `pfa` 1e-3.
    /// * **candidate** — this lane's fix, the shipped defaults verbatim
    ///   (`birth_m` 1, `min_blob_cells` 5, `pfa` 1e-4): a blob now publishes
    ///   on its first detection, no longer asked to persist across several
    ///   frames the way the birth window required, with the false-alarm
    ///   pressure that would otherwise create carried instead by a stricter
    ///   blob-size gate and a tighter Pfa.
    /// * **loosened** — a red control: keeps the candidate's `birth_m` 1 but
    ///   reverts `min_blob_cells`/`pfa` to the pre-fix values, which must
    ///   raise the false-alarm rate well past the candidate's (the negative
    ///   control D-126's coupled seal asks for: loosen deliberately and show
    ///   the false-alarm rate climb — proving the compensating gates are
    ///   load-bearing, not incidental).
    ///
    /// The **revert** control is the baseline row itself: it must show lower
    /// recall than the candidate (the short bursts this lane exists to
    /// catch reappearing once the fix is undone).
    ///
    /// Review 1 (no-merge) found two measurement defects, both fixed here:
    /// * **Blocker 2** — the aggregate recall number (over every ground-truth
    ///   burst, long and short) can improve while AN-1's actual claim, SHORT
    ///   bursts specifically, stays at zero: a fix that only helps bursts
    ///   already long enough to confirm under the old rule would pass the
    ///   old assertion. [`SHORT_BURST_MAX_FRAMES`] buckets the ground truth
    ///   by D-126's own birth-rule cutoff, and short-burst recall is
    ///   asserted on its own.
    /// * **Blocker 3** — the false-alarm oracle excluded any published track
    ///   whose *interval* overlapped any ground-truth burst, which let a
    ///   track born in a noise-only span dodge the count if it coasted long
    ///   enough to later overlap a real burst's window. False alarms are now
    ///   counted by where a track *originates* (`born_frame`): a track
    ///   published from a span with no ground-truth burst under it is a
    ///   false alarm, whatever it does afterward.
    /// * **Blocker 1** — a comment naming the command does not prove it was
    ///   run: this test stays `#[ignore]`d (the capture is 600 MB, real, and
    ///   never committed to this public repo — no host runs it by accident),
    ///   but this doc comment now carries an actual, reproducible transcript
    ///   instead of only a prescription to go run one.
    ///
    /// **Result** (lane host, `cargo test --release -p phosphene-app \
    /// --features analyze --bin phosphene -- --ignored --nocapture \
    /// an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture`, this
    /// commit, [`locate_capture`] resolving `~/captures/cap_913MHz_5Msps.sc16`,
    /// Amendment 3's union-corrected quiet time):
    /// ```text
    /// AN-1/AN-3 coupled seal (D-126): 52 ground-truth bursts (11 SHORT, <= 2 frames) over 30.0s
    /// (146484 frames at 1024-pt FFT / 5.0 MS/s, 29.8s quiet)
    ///   baseline (pre-fix, D-126)            recall 51/52 (98.1%)  SHORT recall 10/11 (90.9%)
    ///     false alarms  1763 (59.121/s, 1.20e-2/frame)
    ///   candidate (this lane's fix, shipped defaults) recall 52/52 (100.0%)  SHORT recall 11/11
    ///     (100.0%)  false alarms  1062 (35.614/s, 7.25e-3/frame)
    ///   loosened (red control: revert the compensating gates) recall 52/52 (100.0%)  SHORT
    ///     recall 11/11 (100.0%)  false alarms 144183 (4835.093/s, 9.84e-1/frame)
    /// test analyze::tests::an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture ... ok
    /// test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 103 filtered out; finished in 42.18s
    /// ```
    ///
    /// Round 6 (no-merge, Amendment 2): the coupled oracle associated ground
    /// truth to tracks *by time alone* — `gt_bursts` carried no frequency, so
    /// an unrelated-frequency track scored a recall hit and was exempt from
    /// the false-alarm count at once. Ground truth is now built the same way
    /// the product's own detector is: [`BlobBuilder`] over the ground-truth
    /// occupancy mask, so two simultaneous emitters in the same frames stay
    /// two bursts (never one min/max span — that hazard re-admits the
    /// defect), each carrying a real bin range. Recall now requires a
    /// published track to overlap a burst in time **and** frequency; a false
    /// alarm requires no burst to contain both the track's birth frame *and*
    /// its frequency. The association tolerance (`assoc_tol_hz`, computed
    /// inline as `TrackerConfig::default().gate_min_bins × meta.bin_hz`) is
    /// derived from the tracker's own centre-frequency gate, not tuned to
    /// make the assertions pass. Honest
    /// association finds *more* ground-truth bursts here (52 vs the
    /// time-only oracle's 40, 11 SHORT vs 7 — the merged multi-emitter spans
    /// the old oracle pooled together now correctly split), and the fix
    /// still improves all three numbers together; per Amendment 2, that
    /// outcome is reported as measured, not assumed in advance.
    ///
    /// Round 7 (no-merge, Amendment 3) found two further defects, both fixed
    /// here:
    /// * **Blocker 1** — the mandatory negative control (forcing
    ///   `freq_overlaps` to always return `true`, recreating round 6's
    ///   time-only association) left this end-to-end test green: baseline,
    ///   candidate and loosened all score through the same oracle, so its
    ///   behaviour largely cancels out of a baseline-vs-candidate
    ///   comparison, and no amount of tightening these assertions can fix
    ///   that. `gt_recalled`/`track_is_false_alarm` are now factored out
    ///   into their own functions, and
    ///   [`frequency_association_is_load_bearing_not_decorative`] (a fast,
    ///   deterministic, non-`#[ignore]`d unit test on constructed inputs, no
    ///   capture needed) fails the moment that association is disabled or
    ///   inverted — verified directly by making that same edit and watching
    ///   it fail before reverting.
    /// * **Blocker 2** — `quiet_s` subtracted the *sum* of every burst's
    ///   duration, so two frequency-separated bursts sharing frames
    ///   (Amendment 2's own splitting requirement) had their shared time
    ///   counted twice, overstating occupied time and understating quiet
    ///   time. Occupied time is now the *union* of burst frame intervals
    ///   (adjacent/overlapping spans merged, frequency ignored — this is a
    ///   time-only measure of "was anything ground-truth-occupied here").
    ///   On this capture the correction is small (59.128 → 59.121/s
    ///   baseline, 35.618 → 35.614/s candidate, 4835.691 → 4835.093/s
    ///   loosened) — simultaneous same-frame, different-frequency bursts are
    ///   rare here — but the bug was real and the fix is unconditional, not
    ///   fitted to leave the story unchanged.
    ///
    /// Round 8 (no-merge, Amendment 4): the same lesson as round 7's blocker
    /// 1, unapplied to the other half — restoring the prior summed-duration
    /// calculation produced the old (wrong) rates and this end-to-end test
    /// **still passed 1/1**, so the union correction was not
    /// regression-protected. The union is now [`union_frame_count`], its own
    /// function with [`union_frame_count_counts_shared_frames_once`] (fast,
    /// deterministic, not `#[ignore]`d) exercising it directly: overlapping
    /// spans and spans sharing exactly one boundary frame must contribute
    /// their union once (strictly less than their sum), with a positive
    /// control (disjoint and touching-but-non-overlapping spans sum
    /// normally) so the assertion cannot pass vacuously. Verified the same
    /// way as [`frequency_association_is_load_bearing_not_decorative`]:
    /// temporarily replaced the function body with a plain sum, watched the
    /// new test fail (`left: 22, right: 16`, the overlapping-spans
    /// assertion), reverted. The end-to-end numbers above are unchanged by
    /// this round — the union arithmetic was already correct, only its
    /// regression protection was missing.
    ///
    /// Review 2 (no-merge) also found the capture path was hardcoded to one
    /// layout (`~/captures/cap_913MHz_5Msps.sc16`); the review host keeps it
    /// directly under `~` instead. [`locate_capture`] now tries both, plus
    /// an env override, and names every path it tried if none exist.
    ///
    /// Round 4 (no-merge, engine-side): the registry seal runs this test for
    /// **every** phosphene lane, almost none of which have any reason to
    /// carry this capture, so a missing capture must skip this one test
    /// cleanly rather than fail it — the fix is [`locate_capture`] returning
    /// `None` (this function then returning immediately) instead of
    /// panicking, still naming every path it tried. Reproduced by setting
    /// `PHOSPHENE_AN1_AN3_CAPTURE` to a path that does not exist and
    /// rerunning the same command above — no panic, `ok`, and this line on
    /// stdout:
    /// ```text
    /// SKIP an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture: the real 913 MHz
    /// capture (D-126's scene, never substituted with a synthetic one) was not found at any of
    /// the paths tried: /nonexistent/bogus.sc16 — set PHOSPHENE_AN1_AN3_CAPTURE to its exact
    /// path, or place it at one of those paths, to run this lane's coupled seal on this host
    /// ```
    #[test]
    #[ignore = "slow: needs the real ~913 MHz capture (D-126), locate_capture()'s own doc has the \
                paths tried, and takes several minutes to score three configurations at once \
                over 30 s of real IQ — not a fast unit test, and this lane never substitutes a \
                synthetic scene for it"]
    fn an1_an3_burst_recall_and_false_alarm_rate_on_the_real_capture() {
        // Round 4/5: this test's own #[ignore] does not stop the registry
        // seal from running it for every phosphene lane (only `--ignored`
        // does that, and the registry's command is not this lane's to
        // change), so a host with nothing to do with this capture must
        // exit clean, not red the fleet-wide seal.
        let Some(path) = locate_capture() else {
            return;
        };

        const RATE_HZ: f64 = 5_000_000.0;
        const FFT: usize = 1024;
        let meta = BandMeta {
            center_freq_hz: Some(913_000_000.0),
            span_hz: RATE_HZ,
            bins: FFT,
            bin_hz: RATE_HZ / FFT as f64,
            frame_dt_s: FFT as f64 / RATE_HZ,
            cal_offset_db: 0.0,
        };

        struct Candidate {
            name: &'static str,
            detector_cfg: DetectorConfig,
            tracker_cfg: TrackerConfig,
        }
        // `DetectorConfig::new()`/`TrackerConfig::default()` now return this
        // lane's *fixed* values (`min_blob_cells: 5`, `cfar.pfa: 1e-4`,
        // `birth_m: 1`) — see their own doc comments for the full measured
        // rationale. So the **baseline** row below spells out the pre-fix
        // numbers explicitly (D-126's measured starting point: `birth_m: 3`,
        // `min_blob_cells: 1`, `pfa: 1e-3`), the **candidate** row is the
        // shipped defaults verbatim, and the **loosened** row is a red
        // control: keep the candidate's `birth_m: 1` (a blob confirms on its
        // first detection) but revert the compensating blob-size/Pfa gates
        // that make that safe, to show the false-alarm rate this lane's
        // whole fix depends on those gates to prevent.
        let candidates: Vec<Candidate> = vec![
            Candidate {
                name: "baseline (pre-fix, D-126)",
                detector_cfg: DetectorConfig {
                    min_blob_cells: 1,
                    cfar: CfarConfig {
                        pfa: 1e-3,
                        ..CfarConfig::new()
                    },
                    ..DetectorConfig::new()
                },
                tracker_cfg: TrackerConfig {
                    birth_m: 3,
                    ..TrackerConfig::default()
                },
            },
            Candidate {
                name: "candidate (this lane's fix, shipped defaults)",
                detector_cfg: DetectorConfig::new(),
                tracker_cfg: TrackerConfig::default(),
            },
            Candidate {
                name: "loosened (red control: revert the compensating gates)",
                detector_cfg: DetectorConfig {
                    min_blob_cells: 1,
                    cfar: CfarConfig {
                        pfa: 1e-3,
                        ..CfarConfig::new()
                    },
                    ..DetectorConfig::new()
                },
                tracker_cfg: TrackerConfig {
                    birth_m: 1,
                    ..TrackerConfig::default()
                },
            },
        ];

        let mut detectors: Vec<Detector> = candidates
            .iter()
            .map(|c| Detector::new(FFT, c.detector_cfg.clone()).unwrap())
            .collect();
        let mut trackers: Vec<Tracker> = candidates
            .iter()
            .map(|c| Tracker::new(c.tracker_cfg.clone()).unwrap())
            .collect();

        // Amendment 2: the association tolerance is derived from the
        // tracker's own centre-frequency gate (`gate_min_bins × bin_hz`),
        // never a number tuned to make assertions pass — the same width the
        // product itself uses to decide whether a detection belongs to a
        // track, so ground truth never demands tighter frequency agreement
        // than the tracker's own association logic provides for.
        let assoc_tol_hz = TrackerConfig::default().gate_min_bins * meta.bin_hz;

        let mut gt_noise = NoiseFloor::new(FFT, NoiseFloorConfig::new()).unwrap();
        // Amendment 2: ground truth must carry frequency, and two emitters
        // live in the same frames must stay two bursts, not one — so this
        // reuses the product's own connected-component grouping
        // (`BlobBuilder`), fed the ground-truth occupancy mask (the same
        // `GT_MARGIN_DB`/`GT_MIN_BINS` criteria as before, `GT_MIN_BINS` now
        // a per-run width via `filter_runs_by_width`) instead of the CFAR
        // mask.
        let mut gt_blobs = BlobBuilder::new(FFT, GT_MIN_BINS).unwrap();
        let mut gt_raw: Vec<GtBurst> = Vec::new();
        let dc_center = FFT / 2;
        let dc_guard = DetectorConfig::new().dc_guard_bins;

        let mut feed = CaptureFrames::open(&path, FFT, RATE_HZ);
        let mut seq = 0u64;
        let mut gt_mask = vec![false; FFT];
        while let Some(spectrum) = feed.next() {
            gt_noise.push_frame(spectrum);
            let floor = gt_noise.floor_dbfs();
            for k in 0..FFT {
                gt_mask[k] =
                    k.abs_diff(dc_center) > dc_guard && spectrum[k] > floor[k] + GT_MARGIN_DB;
            }
            filter_runs_by_width(&mut gt_mask, GT_MIN_BINS);
            for d in gt_blobs.push_frame(spectrum, &gt_mask, floor, seq) {
                gt_raw.push(GtBurst {
                    frame_lo: d.frame_lo,
                    frame_hi: d.frame_hi,
                    bin_lo: d.bin_lo,
                    bin_hi: d.bin_hi,
                });
            }

            for (i, det) in detectors.iter_mut().enumerate() {
                let dets = det.push_frame(spectrum, seq);
                trackers[i].observe(seq, &dets, &meta);
            }
            seq += 1;
        }
        let total_frames = seq;
        assert!(total_frames > 0, "the capture produced no whole FFT frames");
        for d in gt_blobs.flush() {
            gt_raw.push(GtBurst {
                frame_lo: d.frame_lo,
                frame_hi: d.frame_hi,
                bin_lo: d.bin_lo,
                bin_hi: d.bin_hi,
            });
        }
        for (i, det) in detectors.iter_mut().enumerate() {
            let flushed = det.finish();
            trackers[i].observe(total_frames, &flushed, &meta);
        }

        // Bridge brief fades (GT_GAP_FRAMES) between raw blobs that agree in
        // frequency (assoc_tol_hz) into one ground-truth burst; two blobs at
        // unrelated frequencies stay separate however close in time
        // (Amendment 2's hazard: one min/max extent per burst re-admits the
        // defect). Union-find makes this transitive.
        let gt_bursts: Vec<GtBurst> = {
            let n = gt_raw.len();
            let mut parent: Vec<usize> = (0..n).collect();
            for i in 0..n {
                for j in (i + 1)..n {
                    if gt_mergeable(&gt_raw[i], &gt_raw[j], GT_GAP_FRAMES, &meta, assoc_tol_hz) {
                        uf_union(&mut parent, i, j);
                    }
                }
            }
            let mut groups: std::collections::HashMap<usize, GtBurst> =
                std::collections::HashMap::new();
            for (i, r) in gt_raw.iter().enumerate() {
                let root = uf_find(&mut parent, i);
                groups
                    .entry(root)
                    .and_modify(|g| {
                        g.frame_lo = g.frame_lo.min(r.frame_lo);
                        g.frame_hi = g.frame_hi.max(r.frame_hi);
                        g.bin_lo = g.bin_lo.min(r.bin_lo);
                        g.bin_hi = g.bin_hi.max(r.bin_hi);
                    })
                    .or_insert(*r);
            }
            let mut merged: Vec<GtBurst> = groups.into_values().collect();
            merged.sort_by_key(|b| b.frame_lo);
            merged
        };

        assert!(
            !gt_bursts.is_empty(),
            "ground truth found no bursts at all in the real capture — the GT_* constants or \
             the capture itself need a look, not the candidates below"
        );
        // Blocker 2: AN-1's own claim is about SHORT bursts specifically —
        // the ones D-126 measured the old birth_m:3 rule as structurally
        // unable to confirm. Bucket the ground truth so recall on that
        // subset is measured and asserted on its own, not inferred from the
        // aggregate.
        let short_bursts: Vec<GtBurst> = gt_bursts
            .iter()
            .copied()
            .filter(|b| (b.frame_hi - b.frame_lo + 1) <= SHORT_BURST_MAX_FRAMES)
            .collect();
        assert!(
            !short_bursts.is_empty(),
            "ground truth found no SHORT (<= {SHORT_BURST_MAX_FRAMES} frames) bursts at all — \
             AN-1's own claim has nothing to measure against on this capture; that is a finding \
             to surface, not a threshold to quietly loosen"
        );

        let frame_dt = meta.frame_dt_s;
        let total_s = total_frames as f64 * frame_dt;
        // Amendment 3 blocker 2: occupied time is the UNION of burst frame
        // intervals, not the sum of each burst's own duration — see
        // `union_frame_count`'s own doc for why and its unit tests
        // (Amendment 4) for the load-bearing proof.
        let occupied_frames: Vec<(u64, u64)> =
            gt_bursts.iter().map(|b| (b.frame_lo, b.frame_hi)).collect();
        let occupied_s = union_frame_count(&occupied_frames) as f64 * frame_dt;
        let quiet_s = (total_s - occupied_s).max(frame_dt);

        struct Row {
            name: &'static str,
            recall_hits: usize,
            short_recall_hits: usize,
            fa_count: u64,
            fa_per_s: f64,
            fa_per_frame: f64,
        }

        let mut rows = Vec::with_capacity(candidates.len());
        for (i, c) in candidates.iter().enumerate() {
            let published: Vec<Published> = trackers[i]
                .retired()
                .iter()
                .chain(trackers[i].tracks())
                .map(|t| Published {
                    born: t.born_frame,
                    last_seen: t.last_seen_frame,
                    lo_hz: t.center_offset_hz - t.bandwidth_hz / 2.0,
                    hi_hz: t.center_offset_hz + t.bandwidth_hz / 2.0,
                })
                .collect();
            // Amendment 2/3: recall and false-alarm membership both go
            // through the same load-bearing association helpers
            // (`gt_recalled`/`track_is_false_alarm`) that
            // `frequency_association_is_load_bearing_not_decorative`
            // exercises directly.
            let recall_hits = gt_bursts
                .iter()
                .filter(|b| gt_recalled(b, &published, &meta, assoc_tol_hz))
                .count();
            let short_recall_hits = short_bursts
                .iter()
                .filter(|b| gt_recalled(b, &published, &meta, assoc_tol_hz))
                .count();
            let fa_count = published
                .iter()
                .filter(|p| track_is_false_alarm(p, &gt_bursts, &meta, assoc_tol_hz))
                .count() as u64;
            rows.push(Row {
                name: c.name,
                recall_hits,
                short_recall_hits,
                fa_count,
                fa_per_s: fa_count as f64 / quiet_s,
                fa_per_frame: fa_count as f64 / total_frames as f64,
            });
        }

        println!(
            "AN-1/AN-3 coupled seal (D-126): {} ground-truth bursts ({} SHORT, <= \
             {SHORT_BURST_MAX_FRAMES} frames) over {total_s:.1}s ({total_frames} frames at \
             {FFT}-pt FFT / {:.1} MS/s, {quiet_s:.1}s quiet)",
            gt_bursts.len(),
            short_bursts.len(),
            RATE_HZ / 1e6
        );
        for r in &rows {
            println!(
                "  {:<36} recall {:>3}/{:<3} ({:5.1}%)  SHORT recall {:>2}/{:<2} ({:5.1}%)  \
                 false alarms {:>5} ({:.3}/s, {:.2e}/frame)",
                r.name,
                r.recall_hits,
                gt_bursts.len(),
                100.0 * r.recall_hits as f64 / gt_bursts.len() as f64,
                r.short_recall_hits,
                short_bursts.len(),
                100.0 * r.short_recall_hits as f64 / short_bursts.len() as f64,
                r.fa_count,
                r.fa_per_s,
                r.fa_per_frame,
            );
        }

        // ⛔ The coupled seal (D-126, the federator's 2026-09-16 ruling): both
        // numbers asserted together, on the same capture, in the same run.
        let (base, cand, loose) = (&rows[0], &rows[1], &rows[2]);
        assert!(
            cand.recall_hits > base.recall_hits,
            "AN-1: the candidate must catch more of the real bursts than the pre-fix \
             baseline — {}/{} vs {}/{}",
            cand.recall_hits,
            gt_bursts.len(),
            base.recall_hits,
            gt_bursts.len()
        );
        // Blocker 2: the aggregate number above is not AN-1's claim — SHORT
        // bursts are. Assert that subset directly, so a fix that only helps
        // bursts already long enough to confirm under the old rule cannot
        // pass by improving the aggregate alone.
        assert!(
            cand.short_recall_hits > base.short_recall_hits,
            "AN-1's own claim: the candidate must catch more SHORT bursts (<= \
             {SHORT_BURST_MAX_FRAMES} frames, D-126's own birth_m:3 cutoff) than the pre-fix \
             baseline — {}/{} vs {}/{} short bursts. The aggregate recall improving while this \
             stays flat is exactly review 1's blocker 2: fixing the easy (long) bursts while \
             the ones the owner actually complained about stay missed",
            cand.short_recall_hits,
            short_bursts.len(),
            base.short_recall_hits,
            short_bursts.len()
        );
        assert!(
            cand.fa_per_s < base.fa_per_s,
            "AN-3: the candidate must lower the false-alarm rate versus the pre-fix \
             baseline, not just avoid worsening it — {:.3}/s vs {:.3}/s",
            cand.fa_per_s,
            base.fa_per_s
        );
        // Revert control: undoing the fix must bring the short-burst misses
        // back — this is exactly the baseline row above, restated as an
        // explicit comparison so a future edit to the row order cannot
        // silently drop the control.
        assert!(
            base.recall_hits < cand.recall_hits,
            "negative control (revert): the baseline must show the recall this lane's \
             fix recovers — if it does not, the baseline row no longer reverts anything"
        );
        // Loosen-further control: keeping birth_m at 1 but reverting the
        // compensating blob-size/Pfa gates must blow the false-alarm rate up
        // far past the candidate's — proving those gates are load-bearing,
        // not incidental.
        assert!(
            loose.fa_per_s > cand.fa_per_s * 2.0,
            "negative control (loosen further): reverting min_blob_cells/pfa while keeping \
             birth_m at 1 must raise the false-alarm rate well past the candidate's \
             ({:.3}/s vs the candidate's {:.3}/s) — if it does not, this capture no longer \
             demonstrates why those gates are needed",
            loose.fa_per_s,
            cand.fa_per_s
        );
    }
}
