// SPDX-License-Identifier: MIT

//! AN-2 (D-126) evidence: which of the two candidate mechanisms behind "no
//! label text ever appears" actually fires on real data — with counts this
//! test computes and asserts itself, every run, never hard-coded prose
//! (Amendment 2 item 3).
//!
//! `fixtures/an2_real_capture_annotations.tsv` is the exact sequence of
//! `Annotation`s the production detect -> track -> measure -> annotate
//! pipeline produced while replaying the owner's own 913 MHz capture (see
//! that file's own header for the full provenance, the regeneration recipe,
//! and Amendment 1 item 4's trim note — D-122/D-113: evidence lives in the
//! branch, not only in a PR description).
//!
//! **Amendment 2 ruling: `annotation-design` §2.3 / AC-3 / §2.4 stand.**
//! `overlay.rs` ships this lane unchanged from `main` — no
//! `backfill_collisions`, no overflow-candidate split, `text_of` still
//! gated on `AnnotationOverlayState`'s hysteretic `labelled` pool. This
//! lane's deliverable is the diagnosis and its evidence, not a fix to the
//! placement path — the root cause (CFAR false-alarm volume flooding the
//! declutter-admitted pool with candidates that collide with EACH OTHER)
//! is AN-1/AN-3's territory.
//!
//! **Drives the actual production entry point, checked against an
//! independent prediction.** Each frame computes a "shadow" placement —
//! the same `declutter_rank` + `resolve_collisions` pipeline `draw` runs
//! internally on the declutter-membership-admitted pool, called directly
//! here — and separately renders through the REAL `overlay::draw`,
//! counting the `egui::Shape::Text` runs it actually painted.
//! `an2_shadow_prediction_matches_production_rendering` asserts the two
//! agree on every frame, so it goes red if `draw`'s wiring ever changes —
//! for example if a future change reintroduces a call that shows text for
//! a track outside the labelled pool (exactly the shape this lane's own
//! reverted attempt took), the render's text count would exceed the
//! shadow's labelled-only prediction and this assertion would fail.
//!
//! **The two candidates:**
//! 1. The fade-in (`FADE_S`) never completing under track-id churn —
//!    `an2_fade_in_mechanism_is_ruled_out_on_real_data` asserts every
//!    distinct track in the fixture completes its fade-in: D-125's fix
//!    (frame-count allowances -> seconds, full coverage) keeps this ruled
//!    out.
//! 2. The placement ladder degrading admitted tracks to no text under
//!    collision pressure — `an2_placement_ladder_degrades_most_admitted_labels`
//!    computes and asserts the current degrade rate directly from this
//!    fixture, confirming the mechanism without touching the contract that
//!    causes it.

use std::collections::{HashMap, HashSet};

use egui::{FontId, Pos2, Rect};
use phosphene_render::annotation::{Annotation, AnnotationFeed, FreqSpanHz, TrackId};
use phosphene_render::layout::{DisplayParams, Layout};
use phosphene_render::overlay::{
    self, AnnotationLevel, AnnotationOverlayState, LabelCandidate, LabelStyle, VisibleTrack,
};
use phosphene_render::theme::Theme;
use phosphene_render::zoom::ZoomSpan;

const CYCLE_S: f64 = 0.25;
const CAP: usize = 8;
/// Fine enough to resolve `overlay::FADE_S` (0.2s) within one 250ms cycle.
const MICRO_STEPS: usize = 25;

/// The same pixel-x-from-view-fraction arithmetic `overlay::draw` itself
/// uses internally (a private helper there) — duplicated here rather than
/// widening that module's public surface just for this test, since it is
/// one line and the two must simply agree on the same convention `overlay`
/// documents on `freq_to_view_fraction`: `grid.min.x + grid.width() * t`.
fn x_for_fraction(t: f64, grid: Rect) -> f32 {
    grid.min.x + grid.width() * t as f32
}

fn parse_fixture() -> HashMap<usize, Vec<Annotation>> {
    let raw = include_str!("fixtures/an2_real_capture_annotations.tsv");
    let mut by_cycle: HashMap<usize, Vec<Annotation>> = HashMap::new();
    for line in raw.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 9, "malformed fixture row: {line:?}");
        let cycle: usize = f[0].parse().expect("cycle index");
        let ann = Annotation {
            track: TrackId(f[2].parse().expect("track id")),
            span: FreqSpanHz {
                low_hz: f[3].parse().expect("low_hz"),
                high_hz: f[4].parse().expect("high_hz"),
            },
            anchor_hz: f[5].parse().expect("anchor_hz"),
            label: f[8].to_string(),
            detail: vec![],
            confidence: f[6].parse().expect("confidence"),
            strength_db: f[7].parse().expect("strength_db"),
        };
        by_cycle.entry(cycle).or_default().push(ann);
    }
    by_cycle
}

/// Count `egui::Shape::Text` leaves, flattening `Shape::Vec` nesting —
/// with `LabelStyle::SingleLine` (used throughout this replay) `draw` emits
/// exactly one per successfully placed label (see `build_line_job`: a
/// `~signature` suffix would split into two runs, but no label in this
/// fixture carries one).
fn count_text_shapes(shapes: &[egui::Shape]) -> usize {
    fn walk(shape: &egui::Shape, n: &mut usize) {
        match shape {
            egui::Shape::Text(_) => *n += 1,
            egui::Shape::Vec(nested) => {
                for s in nested {
                    walk(s, n);
                }
            }
            _ => {}
        }
    }
    let mut n = 0;
    for s in shapes {
        walk(s, &mut n);
    }
    n
}

/// One micro-step's placement, in both forms: the "shadow" (this test's own
/// direct call to the same `declutter_rank` -> `resolve_collisions`
/// pipeline `draw` runs on the declutter-membership-admitted pool, giving
/// an exact per-track text/no-text prediction under the CURRENT, unchanged
/// `overlay.rs` contract) and the real render's total text-shape count
/// (from actually calling `overlay::draw`, painted through a real egui
/// context and font — see the module doc for why both exist).
struct Frame {
    labelled: HashSet<TrackId>,
    /// Track -> whether the shadow computation gives it real text. Only
    /// ever contains labelled-pool tracks (§2.3/AC-3: nothing outside the
    /// cap is ever offered a placement attempt).
    shadow_text: HashMap<TrackId, bool>,
    rendered_text_count: usize,
    full_alpha: HashSet<TrackId>,
}

/// Step the real `AnnotationOverlayState` through every cycle in
/// `by_cycle`, `MICRO_STEPS` sub-steps per 250ms cycle (fine enough to
/// resolve `FADE_S`), computing one [`Frame`] per sub-step.
fn drive(by_cycle: &HashMap<usize, Vec<Annotation>>) -> Vec<Frame> {
    let max_cycle = by_cycle.keys().copied().max().unwrap_or(0);
    let ctx = egui::Context::default();
    let theme = Theme::default();
    theme.install(&ctx);
    let grid = Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 720.0));
    let font = FontId::monospace(11.0);
    let params = DisplayParams {
        sample_rate: Some(5_000_000.0),
        center: Some(913_000_000.0),
        zoom: ZoomSpan::default(),
        ..Default::default()
    };
    let layout = Layout { grid };

    let feed = AnnotationFeed::new();
    let mut overlay_state = AnnotationOverlayState::new();
    let mut frames = Vec::new();

    for cycle in 0..=max_cycle {
        let empty = Vec::new();
        let anns = by_cycle.get(&cycle).unwrap_or(&empty);
        feed.push_annotations(anns);
        let cycle_t0 = cycle as f64 * CYCLE_S;
        overlay_state.update(&feed, CAP, cycle_t0);

        for step in 0..MICRO_STEPS {
            let now = cycle_t0 + step as f64 * (CYCLE_S / MICRO_STEPS as f64);
            overlay_state.update(&feed, CAP, now); // no-op unless a new push happened
            let visible = overlay_state.visible(now);
            if visible.is_empty() {
                frames.push(Frame {
                    labelled: HashSet::new(),
                    shadow_text: HashMap::new(),
                    rendered_text_count: 0,
                    full_alpha: HashSet::new(),
                });
                continue;
            }

            let to_ann = |v: &VisibleTrack| Annotation {
                track: v.track,
                span: v.span,
                anchor_hz: v.anchor_hz,
                label: v.label.clone(),
                detail: vec![],
                confidence: v.confidence,
                strength_db: v.strength_db,
            };
            let rank_input: Vec<Annotation> = visible.iter().map(to_ann).collect();
            let order = overlay::declutter_rank(&rank_input);
            // §2.3: collision processing is scoped to "the currently-
            // labelled tracks (§2.4's top-N membership)" — the shadow
            // candidate pool is exactly that, nothing beyond it.
            let labelled_candidates: Vec<LabelCandidate> = order
                .iter()
                .copied()
                .filter(|&i| visible[i].labelled)
                .map(|i| {
                    let a = &rank_input[i];
                    LabelCandidate {
                        track: a.track,
                        anchor_x: x_for_fraction(
                            overlay::freq_to_view_fraction(a.anchor_hz, &params),
                            grid,
                        ),
                        full_label: a.label.clone(),
                    }
                })
                .collect();

            let input = egui::RawInput {
                screen_rect: Some(grid),
                ..Default::default()
            };
            let mut shadow_text = HashMap::new();
            let output = ctx.run_ui(input, |ui| {
                let painter = ui.painter();
                let width_of = |s: &str| {
                    painter
                        .layout_no_wrap(s.to_owned(), font.clone(), theme.text)
                        .size()
                        .x
                };
                // The shadow: the same call `draw` makes internally on the
                // labelled pool, called directly here so this test has an
                // exact prediction independent of parsing rendered pixels.
                for p in overlay::resolve_collisions(&labelled_candidates, &width_of) {
                    shadow_text.insert(p.track, p.text.is_some());
                }

                // The real thing: paint through the actual production
                // entry point on the SAME `visible`, and count what it
                // actually drew.
                overlay::draw(
                    painter,
                    &layout,
                    &params,
                    &theme,
                    1.0,
                    AnnotationLevel::Full,
                    LabelStyle::SingleLine,
                    &visible,
                );
            });
            let rendered_text_count = count_text_shapes(
                &output
                    .shapes
                    .iter()
                    .map(|c| c.shape.clone())
                    .collect::<Vec<_>>(),
            );
            output.drop_without_applying_deltas();

            frames.push(Frame {
                labelled: visible
                    .iter()
                    .filter(|v| v.labelled)
                    .map(|v| v.track)
                    .collect(),
                shadow_text,
                rendered_text_count,
                full_alpha: visible
                    .iter()
                    .filter(|v| v.alpha >= 0.999)
                    .map(|v| v.track)
                    .collect(),
            });
        }
    }
    frames
}

#[test]
fn an2_shadow_prediction_matches_production_rendering() {
    // The shadow computation calls the public `resolve_collisions`
    // directly on the declutter-membership-admitted pool, so it predicts
    // `draw`'s §2.3/AC-3-compliant behaviour independently of what `draw`
    // itself does. If `draw` ever stops matching this prediction — for
    // example by showing text for a track outside the labelled pool, the
    // exact shape this lane's own reverted `backfill_collisions` attempt
    // took — the real render's text count diverges from the shadow's and
    // this assertion is what turns red.
    let by_cycle = parse_fixture();
    let frames = drive(&by_cycle);
    let mut mismatches = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let shadow_count = f.shadow_text.values().filter(|&&t| t).count();
        if shadow_count != f.rendered_text_count {
            mismatches.push((i, shadow_count, f.rendered_text_count));
        }
    }
    assert!(
        mismatches.is_empty(),
        "production `draw` rendered a different text-shape count than the shadow prediction on \
         {} of {} frames (first few (frame, shadow, rendered): {:?}) — draw() is not wired to \
         the same, contract-compliant placement pipeline this test predicts from",
        mismatches.len(),
        frames.len(),
        &mismatches[..mismatches.len().min(5)]
    );
}

#[test]
fn an2_fade_in_mechanism_is_ruled_out_on_real_data() {
    let by_cycle = parse_fixture();
    let total_distinct: HashSet<TrackId> = by_cycle.values().flatten().map(|a| a.track).collect();
    let frames = drive(&by_cycle);

    let mut ever_full_alpha = HashSet::new();
    for f in &frames {
        ever_full_alpha.extend(f.full_alpha.iter().copied());
    }
    // Every distinct track that ever appeared eventually reaches full
    // opacity: D-125's fix (frame-count -> seconds allowances, full
    // coverage) keeps every analysis cycle at least CYCLE=250ms apart,
    // comfortably longer than FADE_S=0.2s, so a track published even once
    // has time to finish fading in before it can be marked removed. If
    // this regresses, the churn D-125 fixed has come back.
    assert_eq!(
        ever_full_alpha.len(),
        total_distinct.len(),
        "every one of the {} distinct tracks in the real capture should complete its fade-in \
         (mechanism 2 must stay ruled out)",
        total_distinct.len()
    );
}

#[test]
fn an2_placement_ladder_degrades_most_admitted_labels() {
    // Mechanism 1, confirmed: of the tracks the declutter-membership pool
    // admits (§2.4's cap), most degrade all the way to a bare anchor tick
    // under collision — because CFAR's false-alarm volume (AN-1/AN-3's
    // territory, not touched here) floods the admitted pool with
    // candidates that collide with EACH OTHER, not because the placement
    // logic itself is broken. Every number below is computed fresh from
    // the fixture in this run — Amendment 2 item 3: no prose baseline.
    let by_cycle = parse_fixture();
    let frames = drive(&by_cycle);

    let mut ever_labelled: HashSet<TrackId> = HashSet::new();
    let mut ever_shown_text: HashSet<TrackId> = HashSet::new();
    let mut slots = 0usize;
    let mut filled = 0usize;
    for f in &frames {
        ever_labelled.extend(f.labelled.iter().copied());
        slots += f.labelled.len();
        filled += f.shadow_text.values().filter(|&&shown| shown).count();
        for (&track, &shown) in &f.shadow_text {
            if shown {
                ever_shown_text.insert(track);
            }
        }
    }
    let degrade_rate = 1.0 - filled as f64 / slots.max(1) as f64;
    println!(
        "AN-2 real-capture diagnosis (production overlay::draw, cross-checked against the \
         shadow prediction, trimmed 46-cycle fixture): {slots} declutter-cap-admitted \
         track-frame slots, {filled} ({:.1}%) kept real text at their own admitted position, \
         {:.1}% degraded to a bare anchor tick; {} distinct tracks ever admitted to the cap, {} \
         of those ever shown real text",
        100.0 * filled as f64 / slots.max(1) as f64,
        degrade_rate * 100.0,
        ever_labelled.len(),
        ever_shown_text.len(),
    );

    // Reject the degenerate case explicitly (Amendment 2 item 3): a
    // measurement pipeline that silently broke would plausibly report
    // either "everything admitted" (slots == filled, no collision ever
    // modelled) or "nothing ever shown" (filled == 0, no placement ever
    // succeeds) — both are checked, separately from the rate assertion
    // below, so neither can hide inside it.
    assert!(
        slots > 0,
        "no declutter-admitted slots were ever observed on this fixture"
    );
    assert!(
        filled > 0,
        "no admitted track ever kept real text on this fixture — the placement pipeline itself \
         may be broken, not merely collision-heavy"
    );
    assert!(
        ever_shown_text.len() < ever_labelled.len(),
        "expected the placement ladder to degrade at least one admitted track's label \
         (ever_shown_text {} should be strictly fewer than ever_labelled {}) — if every \
         admitted track keeps its text now, the collision mechanism this lane diagnosed may \
         have gone away and this assertion (and the lane's own diagnosis) need revisiting",
        ever_shown_text.len(),
        ever_labelled.len()
    );
    // The dominant-degradation claim itself: most of the capacity the
    // declutter cap commits to goes unused (a bare anchor tick, nobody's
    // text), computed and asserted directly rather than only printed.
    assert!(
        degrade_rate > 0.5,
        "expected the placement ladder to degrade more than half of admitted track-frame \
         slots on this real, collision-heavy capture window, got {:.1}%",
        degrade_rate * 100.0
    );
}
