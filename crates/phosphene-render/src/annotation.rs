// SPDX-License-Identifier: MIT

//! The annotation-overlay feed (nextgen spec §2.2/§2.3, D-017):
//! [`AnnotationFeed::push_annotations`] and the [`Annotation`] it carries.
//!
//! The render seam is a *feed*, not scattered edits: analysis publishes a
//! complete annotation set via `push_annotations(&[Annotation])` — the exact
//! API name D-017 freezes — and render takes the latest set as an atomic
//! snapshot at its own cadence, like v1's accumulator swap. **This lane
//! (AI-1) fills the feed in: it stores the latest set and the D-098 §7.1
//! load side-channel; [`crate::overlay`] is what actually draws them.**
//!
//! [`Annotation`] mirrors the definition `phosphene-analyze` builds against
//! (its `annotate` module, per nextgen spec §5.2/§5.4): semantic content
//! only — what to label and how strong/confident it is. Pixels, placement,
//! and declutter policy are the renderer's ([`crate::overlay`]).
//!
//! **`FreqSpanHz`/`Annotation::anchor_hz` unit (D-098 §7.3, doc-comment-only
//! clarification — no field change):** these `f64`s follow the same
//! rate-optional convention as [`crate::axis::FreqValue`] — hertz when the
//! producing source declared a sample rate, cycles/sample when it did not.
//! The frozen type carries no unit tag of its own (unlike `FreqValue`'s
//! `Hz`/`Normalized` variants), so a consumer must already know which
//! convention is in force from the same place it gets `DisplayParams` — the
//! producer and the renderer agree on this out of band, exactly as
//! `phosphene-analyze`'s `annotate` module and the live-integration wiring
//! do (its `rate_known` parameter).

use std::sync::{Arc, Mutex};

/// Stable per-session identity of the track an annotation describes (nextgen
/// FR-AD4). Mirrors `phosphene-analyze`'s `TrackId`; the producer assigns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(pub u64);

/// A frequency interval, Hz.
///
/// Frame of reference: absolute frequencies when the band center is known,
/// otherwise offsets from the (unknown) center — the same convention the
/// producing measurement set used, and matching what the display shows on its
/// axis (v1 FR-C3). See the module docs for the rate-less convention (D-098
/// §7.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreqSpanHz {
    /// Lower edge, Hz.
    pub low_hz: f64,
    /// Upper edge, Hz.
    pub high_hz: f64,
}

/// One in-place annotation: the label riding above a detected signal, over a
/// vertical color-shade band marking its occupied bandwidth (nextgen FR-AD6,
/// FR-AM6).
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// The track this annotation describes.
    pub track: TrackId,
    /// Occupied frequency extent — the FR-AD6 color-shade band.
    pub span: FreqSpanHz,
    /// Where the label anchors (the signal's center), Hz.
    pub anchor_hz: f64,
    /// Concise headline shown in place (FR-AS3: measured and inferred
    /// content clearly separated, "consistent with", never "is").
    pub label: String,
    /// Detail-panel lines for select/hover drill-in (FR-AU3); also the
    /// copy-as-text content (FR-AU4).
    pub detail: Vec<String>,
    /// Overall confidence of the headline, `0.0..=1.0` (NFR-A3).
    pub confidence: f32,
    /// Signal strength (SNR, dB) — with `confidence`, the declutter ranking
    /// key when a band is crowded (FR-AU2, NFR-A5).
    pub strength_db: f32,
}

/// Analysis-side overload signal (D-098 §7.1, NFR-A5, annotation design
/// §2.4): additive to [`Annotation`]/`push_annotations`, published once per
/// analysis cycle. Lets the display say "analysis is shedding work" without
/// guessing from a track count alone — a quiet band and an overloaded one
/// both push a small `Annotation` slice, and only this field tells them
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnalysisLoad {
    /// The analysis pipeline is shedding tracks under load this cycle.
    pub shedding: bool,
    /// D-125 Amendment 4 item 3, the owner's ruling: "show an honest
    /// coverage count: frames analysed out of frames received, never a
    /// silent drop." Cumulative since the last retune reset (matching
    /// `dropped_batches`/`processed_pct`'s own D-013 idiom): every raw
    /// frame the source has produced, whether or not it survived to be
    /// analysed.
    pub frames_received: u64,
    /// D-125 Amendment 4 item 3: cumulative frames actually folded into the
    /// detector/tracker since the last retune reset — always `<=
    /// frames_received`. Equal to it exactly when nothing has ever been
    /// shed.
    pub frames_analysed: u64,
}

/// The annotation feed: single producer (the analysis pipeline), snapshot
/// consumers (the render overlay).
///
/// Storage is a mutex-guarded [`Arc`] swap: the lock is held only to exchange
/// a pointer, never while building or drawing. If profiling ever shows
/// contention, the implementation can move to a lock-free swap without
/// changing this API.
#[derive(Debug, Default)]
pub struct AnnotationFeed {
    latest: Mutex<Arc<Vec<Annotation>>>,
    load: Mutex<AnalysisLoad>,
}

impl AnnotationFeed {
    /// An empty feed, load nominal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a complete annotation set, replacing the previous one — the
    /// D-017 render-seam API. The set is complete, not incremental: a track
    /// with no annotation this cycle is no longer annotated.
    pub fn push_annotations(&self, annotations: &[Annotation]) {
        let next = Arc::new(annotations.to_vec());
        *self.latest.lock().expect("annotation feed poisoned") = next;
    }

    /// The most recently published set (cheap: clones an [`Arc`], not the
    /// annotations).
    pub fn snapshot(&self) -> Arc<Vec<Annotation>> {
        Arc::clone(&self.latest.lock().expect("annotation feed poisoned"))
    }

    /// Publish this cycle's analysis-side load (D-098 §7.1) — additive to,
    /// and independent of, [`push_annotations`].
    pub fn set_load(&self, load: AnalysisLoad) {
        *self.load.lock().expect("annotation feed poisoned") = load;
    }

    /// The most recently published load. Nominal (`shedding: false`) before
    /// the first [`set_load`](Self::set_load) call.
    pub fn load(&self) -> AnalysisLoad {
        *self.load.lock().expect("annotation feed poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ann(id: u64, label: &str) -> Annotation {
        Annotation {
            track: TrackId(id),
            span: FreqSpanHz {
                low_hz: 915_000_000.0,
                high_hz: 915_200_000.0,
            },
            anchor_hz: 915_100_000.0,
            label: label.to_string(),
            detail: vec!["duty ≈ 40%".to_string()],
            confidence: 0.8,
            strength_db: 22.0,
        }
    }

    #[test]
    fn feed_replaces_the_set_atomically() {
        let feed = AnnotationFeed::new();
        assert!(feed.snapshot().is_empty());

        feed.push_annotations(&[ann(1, "FSK"), ann(2, "chirp")]);
        let first = feed.snapshot();
        assert_eq!(first.len(), 2);

        feed.push_annotations(&[ann(3, "OFDM")]);
        // The consumer's earlier snapshot is unchanged; a new snapshot sees
        // the replacement set.
        assert_eq!(first.len(), 2);
        let second = feed.snapshot();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].track, TrackId(3));
    }

    #[test]
    fn load_is_nominal_until_set_then_independent_of_annotations() {
        let feed = AnnotationFeed::new();
        assert_eq!(feed.load(), AnalysisLoad::default());

        let shedding_load = AnalysisLoad {
            shedding: true,
            frames_received: 300,
            frames_analysed: 256,
        };
        feed.set_load(shedding_load);
        assert_eq!(feed.load(), shedding_load);
        // Independent of the annotation set (7.1: additive, not entangled).
        assert!(feed.snapshot().is_empty());

        feed.push_annotations(&[ann(1, "OOK")]);
        assert_eq!(feed.load(), shedding_load);
    }
}
