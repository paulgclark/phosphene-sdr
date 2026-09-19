// SPDX-License-Identifier: MIT

//! [`Annotation`] and the feed shape `phosphene-render` will consume
//! (nextgen spec §5.2/§5.4, D-017).
//!
//! The render seam is a *feed*, not scattered edits: analysis publishes a
//! complete annotation set via [`AnnotationFeed::push_annotations`] (the
//! exact API name D-017 freezes) and render takes the latest set as an
//! atomic snapshot at its own cadence, like v1's accumulator swap (spec
//! §8.2). Annotations carry semantic content only — what to label and how
//! strong/confident it is; pixels, placement and declutter policy are the
//! renderer's, per the §9 A0 design phase.
//!
//! `phosphene-render` is held by in-flight v1 lanes, so the feed lives here
//! for now; the render-side consumption is wired by its own late-integration
//! lane (D-017).

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::track::TrackId;

mod build;

pub use build::build_annotation;

/// A frequency interval, Hz.
///
/// Frame of reference: absolute frequencies when the band center is known,
/// otherwise offsets from the (unknown) center — the same convention the
/// producing [`crate::measure::MeasurementSet`] used, and matching what the
/// display shows on its axis (v1 FR-C3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FreqSpanHz {
    /// Lower edge, Hz.
    pub low_hz: f64,
    /// Upper edge, Hz.
    pub high_hz: f64,
}

/// One in-place annotation: the label riding above a detected signal, over a
/// vertical color-shade band marking its occupied bandwidth (FR-AD6,
/// FR-AM6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// The annotation feed: single producer (the analysis pipeline), snapshot
/// consumers (the render overlay).
///
/// The skeleton implements the *shape* — publish a full set, read the latest
/// set atomically — with a mutex-guarded [`Arc`] swap; the lock is held only
/// to exchange a pointer, never while building or drawing. If profiling ever
/// shows contention, the implementation can move to a lock-free swap without
/// changing this API.
#[derive(Debug, Default)]
pub struct AnnotationFeed {
    latest: Mutex<Arc<Vec<Annotation>>>,
}

impl AnnotationFeed {
    /// An empty feed.
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
}
