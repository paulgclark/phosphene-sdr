// SPDX-License-Identifier: MIT

//! D-120 (brief amendment 10): a negative control for the overlay's
//! draw-order fix (D-110, `AnnotationOverlayState::visible`'s track-ID
//! sort). The lockstep capture's byte-identical determinism tests only ever
//! render a scene with one active track at a time, so reverting the sort
//! changes nothing there — they cannot catch this regression. This test
//! builds many simultaneously visible tracks instead: `self.tracks` is a
//! `HashMap`, and iterating that many keys unsorted comes out in a
//! process-randomised order essentially every run, so asserting sorted
//! order here goes red the moment the sort is reverted.

use phosphene_render::{Annotation, AnnotationFeed, AnnotationOverlayState, FreqSpanHz, TrackId};

/// Comfortably above the point a `HashMap`'s randomised iteration order
/// becomes observable in practice (D-120: "at least 50").
const TRACK_COUNT: u64 = 50;

fn annotation(id: u64) -> Annotation {
    Annotation {
        track: TrackId(id),
        span: FreqSpanHz {
            low_hz: 1_000.0 * id as f64,
            high_hz: 1_000.0 * id as f64 + 500.0,
        },
        anchor_hz: 1_000.0 * id as f64 + 250.0,
        label: format!("SIGNAL {id}"),
        detail: Vec::new(),
        confidence: 0.9,
        strength_db: -10.0,
    }
}

#[test]
fn visible_returns_many_tracks_in_track_id_order() {
    let feed = AnnotationFeed::new();
    let annotations: Vec<Annotation> = (0..TRACK_COUNT).map(annotation).collect();
    feed.push_annotations(&annotations);

    let mut state = AnnotationOverlayState::new();
    state.update(&feed, TRACK_COUNT as usize, 0.0);

    // Past FADE_S so every track has fully faded in (alpha == 1.0) and none
    // is filtered out of `visible()`.
    let visible = state.visible(1.0);

    assert_eq!(
        visible.len(),
        TRACK_COUNT as usize,
        "expected all {TRACK_COUNT} tracks visible"
    );

    let ids: Vec<u64> = visible.iter().map(|v| v.track.0).collect();
    let mut sorted_ids = ids.clone();
    sorted_ids.sort_unstable();
    assert_eq!(
        ids, sorted_ids,
        "visible() must return tracks in track-ID order (D-110) so that \
         alpha-blended draw order is deterministic across runs; got {ids:?}, \
         which is not sorted — the byte-identical capture tests cannot see \
         this regression because they never have more than one active track"
    );
}
