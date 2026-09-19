// SPDX-License-Identifier: MIT

//! The annotation overlay (annotation design
//! `specs/002-nextgen-analyze/annotation-design.md`): the render-owned state
//! that reads [`crate::annotation::AnnotationFeed`]'s latest set each frame
//! and derives interpolated geometry (§2.5), declutter membership (§2.4),
//! fade phase (§2.5) and collision-resolved label placement (§2.3) — none of
//! which round-trips back into [`Annotation`] itself.
//!
//! Everything reachable without an egui context is a pure function or a
//! plain-data method, deliberately, so the placement/declutter/collision
//! logic is unit-testable (design D-092 Amendment 1's `[Unit]` evidence
//! tag) without a rendered frame. [`draw`] is the one function that actually
//! paints, called from [`crate::chrome::draw_tunable`] alongside
//! `axis_labels`.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke};

use crate::annotation::{Annotation, AnnotationFeed, FreqSpanHz, TrackId};
use crate::chrome::snap;
use crate::layout::{DisplayParams, Layout};
use crate::theme::Theme;

/// Annotation design §2.7 `N` key: full (labels + shade) or shade-only
/// (bands + anchor ticks, labels hidden, detail-on-click unaffected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationLevel {
    /// Labels and shade bands both draw.
    Full,
    /// Shade bands and anchor ticks only; no label text.
    ShadeOnly,
}

impl AnnotationLevel {
    /// The other state — FR-AU2's "dial down" is a two-state toggle, not a
    /// cycle, matching the requirement's own wording.
    pub fn toggled(self) -> Self {
        match self {
            AnnotationLevel::Full => AnnotationLevel::ShadeOnly,
            AnnotationLevel::ShadeOnly => AnnotationLevel::Full,
        }
    }
}

/// D-102 item 1 / D-105: the two label styles the look-review command can
/// produce. **Stacked is the default** (D-105, owner ruling: signals rarely
/// reach full scale, so the top of the plot usually has room for the taller
/// stack, and the narrower per-line width fits more labels before declutter
/// reduces one to a tick). Single line stays in the code as the sole
/// alternative, reachable only from the look-review command — not a
/// keymap-bound, user-facing setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabelStyle {
    /// The same grammar's fields stacked on up to three lines (identity;
    /// bandwidth; then power/flag/signature together) — see [`stack_label`].
    #[default]
    Stacked,
    /// The §2.2 headline grammar on one line: `SIGNAL · 12.5K · 22dB`.
    /// Look-review-only (D-105 superseded this being the default).
    SingleLine,
}

/// Split a §2.2-grammar label into up to three stacked lines (D-102 item 1):
/// identity on its own line, bandwidth on its own line, then every remaining
/// ` · `-separated segment (power, the optional flag, the optional
/// `~<signature>` run) joined back onto one third line. Mechanical, on the
/// grammar's own separator — like [`degrade_once`], this never re-parses
/// what a segment means, only how many of them there are.
pub fn stack_label(label: &str) -> Vec<String> {
    let parts: Vec<&str> = label.split(" · ").collect();
    match parts.len() {
        0 => Vec::new(),
        1 => vec![parts[0].to_string()],
        2 => vec![parts[0].to_string(), parts[1].to_string()],
        _ => vec![
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2..].join(" · "),
        ],
    }
}

/// Fade-**in** duration on first appearance, seconds (§2.5): cheap insurance
/// against a single missed analysis cycle reading as a flicker. D-125 item 5
/// leaves this unchanged — only the fade-**out** after a track leaves the
/// feed changes, below.
pub const FADE_S: f64 = 0.2;

/// D-125 item 5, the owner's ruling verbatim: "10 seconds but with a long
/// slow fade for the last 5." After a track leaves the feed its label stays
/// **fully visible** for this long, seconds, before the fade begins.
pub const LABEL_HOLD_S: f64 = 5.0;

/// D-125 item 5: after [`LABEL_HOLD_S`]'s full-visibility hold, the label
/// fades over this many more seconds, then is gone — [`LABEL_HOLD_S`] +
/// `LABEL_FADE_S` = 10 s total, the owner's number. The curve is **linear**
/// in opacity over elapsed time past the hold (the same shape [`FADE_S`]'s
/// fade-in already uses), chosen for consistency with that existing curve
/// and because the owner's wording ("a long slow fade") asks for a duration,
/// not a particular easing.
pub const LABEL_FADE_S: f64 = 5.0;

/// Hover-preview dwell before the detail panel opens as an ephemeral peek
/// (§2.6), seconds.
pub const HOVER_DWELL_S: f64 = 0.4;

/// Minimum rendered shade-band width, **device pixels** (§2.9): a signal
/// narrower than the current bin width is never sub-pixel-invisible.
pub const MIN_BAND_DEVICE_PX: f32 = 3.0;

/// §2.1 halo alphas (0..=255): the low-alpha fill between a band's edges.
const HALO_FILL_ALPHA: f32 = 16.0;
/// §2.1 halo alphas: the near-black casing stroke.
const HALO_CASING_ALPHA: f32 = 140.0;
/// §2.1 halo alphas: the near-white core stroke.
const HALO_CORE_ALPHA: f32 = 210.0;

/// Label backing-plate padding beyond the text's own bounds, points (D-102
/// item 7): "just larger than the text".
const LABEL_BACKING_PAD_X: f32 = 3.0;
const LABEL_BACKING_PAD_Y: f32 = 1.5;
/// Label backing-plate corner radius, points.
const LABEL_BACKING_ROUNDING: f32 = 3.0;

/// Label backing-plate opacity (D-102 item 7, fix pass 1). A deliberately
/// **separate** constant from the §2.1 halo's own casing alpha (140/255,
/// unchanged — halo values are not in this lane, D-102): the plate's job is
/// to actually hide whatever is behind the text, not to read as a halo
/// stroke. 140/255 demonstrably did not: measured directly against a
/// rendered PNG, the band's own near-white core stroke composited at
/// 140/255 over it still read at ~147/255 — clearly visible, not "hidden."
/// egui/wgpu blends in linear light, so the naive "alpha/255 of the way to
/// black" arithmetic understates how much of a bright pixel survives
/// underneath; 250/255 was chosen by rendering candidates and reading the
/// actual composited pixel back (see the PR for the before/after values),
/// landing at 30/255 for that same worst-case core-stroke crossing —
/// comfortably under the 32/255 "meaningfully different" bound this
/// codebase's own golden-diff outlier threshold already uses. Still
/// nominally "partly transparent" (not the literal 255 the fully-opaque
/// help-overlay panel uses elsewhere), while being effectively opaque to
/// the eye against anything this display can actually show.
const LABEL_BACKING_ALPHA: f32 = 250.0;

/// Vertical gap between stacked label lines, points (D-102 item 1).
const STACKED_LINE_GAP: f32 = 1.0;
/// Fixed label-lane height for the stacked style, points: the single-line
/// style's lane is ≈16pt (one 11pt line plus padding, §2.3); three such
/// lines plus two [`STACKED_LINE_GAP`]s need noticeably more room. Generous
/// rather than exact — this style is a look-review-only option (D-102), not
/// a pixel-tuned production default.
const STACKED_LANE_HEIGHT_PT: f32 = 46.0;

// ---------------------------------------------------------------------
// Frequency → pixel mapping (§1: "the same DisplayParams/ZoomSpan machinery
// crate::axis::freq_at uses — an annotation's geometry is computed with the
// INVERSE of that mapping").
// ---------------------------------------------------------------------

/// The view-fraction (0 = left grid edge, 1 = right; may fall outside `0..1`
/// for an off-screen frequency) that [`crate::axis::freq_at`] would map back
/// to `hz` — the exact inverse of that function, so annotation geometry
/// cannot disagree with the axis (AC-7).
pub fn freq_to_view_fraction(hz: f64, params: &DisplayParams) -> f64 {
    let u = match params.sample_rate {
        Some(rate) if rate != 0.0 => (hz - params.center.unwrap_or(0.0)) / rate + 0.5,
        // Rate-less: `hz` already carries the normalized cycles/sample value
        // (D-098 §7.3's convention), the same domain `axis::freq_at` returns
        // as `FreqValue::Normalized(u - 0.5)`.
        _ => hz + 0.5,
    };
    (u - params.zoom.lo()) / params.zoom.width()
}

/// Pixel x (egui points) for a view fraction `t` across `grid`.
fn x_for_fraction(t: f64, grid: Rect) -> f32 {
    grid.min.x + grid.width() * t as f32
}

/// Minimum horizontal hit-region width, egui points (§2.6): a sub-bin-wide
/// signal is still clickable regardless of its rendered band's true width.
const MIN_HIT_PT: f32 = 10.0;

/// Which visible track (if any) the pointer's x sits over, using each
/// track's own claimed band interval widened to at least [`MIN_HIT_PT`]
/// (§2.6). Callers gate this on the pointer's y already being inside the
/// grid — the band decorates the full plot height, so x alone decides here.
pub fn hit_test(
    visible: &[VisibleTrack],
    params: &DisplayParams,
    grid: Rect,
    pointer_x: f32,
) -> Option<TrackId> {
    for v in visible {
        let t_lo = freq_to_view_fraction(v.span.low_hz, params);
        let t_hi = freq_to_view_fraction(v.span.high_hz, params);
        let anchor_t = freq_to_view_fraction(v.anchor_hz, params);
        let Some(band) = clip_band(t_lo, t_hi, anchor_t) else {
            continue;
        };
        let (mut lo, mut hi) = (
            x_for_fraction(band.t_lo, grid),
            x_for_fraction(band.t_hi, grid),
        );
        if hi - lo < MIN_HIT_PT {
            let c = (lo + hi) / 2.0;
            lo = c - MIN_HIT_PT / 2.0;
            hi = c + MIN_HIT_PT / 2.0;
        }
        if pointer_x >= lo && pointer_x <= hi {
            return Some(v.track);
        }
    }
    None
}

// ---------------------------------------------------------------------
// §2.3/§2.4: the declutter total order
// ---------------------------------------------------------------------

/// The declutter total order (§2.3/§2.4): `(strength_db desc, confidence
/// desc, anchor_hz asc, TrackId asc)`. Because `TrackId` is unique, this key
/// can never tie — a strict total order over any set of tracks, independent
/// of the order `annotations` happen to arrive in.
pub fn declutter_rank(annotations: &[Annotation]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..annotations.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (&annotations[a], &annotations[b]);
        y.strength_db
            .partial_cmp(&x.strength_db)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                y.confidence
                    .partial_cmp(&x.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                x.anchor_hz
                    .partial_cmp(&y.anchor_hz)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| x.track.0.cmp(&y.track.0))
    });
    idx
}

/// Per-track membership bookkeeping for the §2.4 hysteresis rule: a track
/// already labelled stays labelled unless it ranks more than one position
/// outside the cap for **two consecutive** snapshots.
#[derive(Debug, Default)]
pub struct DeclutterMembership {
    state: HashMap<TrackId, MemberState>,
}

#[derive(Debug, Clone, Copy)]
struct MemberState {
    labelled: bool,
    over_streak: u32,
}

impl DeclutterMembership {
    /// A fresh membership tracker — nothing labelled yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update membership from one snapshot's declutter rank order (best
    /// first) and the current cap. Call once per **new** `push_annotations`
    /// snapshot, never per render frame — the rule is defined over
    /// consecutive snapshots, not consecutive frames.
    pub fn update(&mut self, ranked_ids: &[TrackId], cap: usize) {
        let present: std::collections::HashSet<TrackId> = ranked_ids.iter().copied().collect();
        self.state.retain(|id, _| present.contains(id));
        for (rank, &id) in ranked_ids.iter().enumerate() {
            let st = self.state.entry(id).or_insert(MemberState {
                labelled: false,
                over_streak: 0,
            });
            if rank < cap {
                st.labelled = true;
                st.over_streak = 0;
            } else if rank == cap {
                // Exactly one position outside the cap: grace, no flicker.
                st.over_streak = 0;
            } else if st.labelled {
                st.over_streak += 1;
                if st.over_streak >= 2 {
                    st.labelled = false;
                    st.over_streak = 0;
                }
            }
        }
    }

    /// Whether `id` currently shows a label (as of the last
    /// [`update`](Self::update)).
    pub fn is_labelled(&self, id: TrackId) -> bool {
        self.state.get(&id).is_some_and(|s| s.labelled)
    }
}

// ---------------------------------------------------------------------
// §2.3: collision resolution — one deterministic pass over the total order
// ---------------------------------------------------------------------

/// One label to place, in declutter-rank order (best first).
#[derive(Debug, Clone)]
pub struct LabelCandidate {
    /// The track this label belongs to.
    pub track: TrackId,
    /// Pixel x of the label's anchor (egui points).
    pub anchor_x: f32,
    /// The label's fullest-form text (§2.2's complete grammar).
    pub full_label: String,
}

/// A placed label: the resulting (possibly degraded) text and the horizontal
/// interval it claims in the lane. `text: None` means it degraded all the
/// way to no text — only the anchor tick remains (§2.3 step 2's last rung).
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedLabel {
    /// The track this placement belongs to.
    pub track: TrackId,
    /// The label text after any collision-driven degradation, or `None` if
    /// it degraded to nothing.
    pub text: Option<String>,
    /// The claimed horizontal interval (egui points), or a zero-width point
    /// at the anchor once `text` is `None`.
    pub interval: (f32, f32),
}

/// Drop the label's rightmost ` · `-separated segment — the mechanical
/// operation the §2.3 abbreviation ladder actually is (`~<signature>` first,
/// then `flag`, then `snr`, then `bw`, since the grammar always appends them
/// in that order: `<identity> · <bw> · <snr>dB[ · <flag>][ · ~<signature>]`).
/// Render never needs to know *what* a segment means to drop it — only the
/// grammar's own separator convention. `None` once the identity segment
/// itself is gone (the label is empty).
fn degrade_once(label: &str) -> Option<String> {
    let mut parts: Vec<&str> = label.split(" · ").collect();
    parts.pop();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}

fn intervals_overlap(a: (f32, f32), b: (f32, f32)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Resolve collisions for labels already in declutter-rank order (best
/// first): a single deterministic pass, each candidate only ever contending
/// with intervals already claimed by a **higher-ranked** label — never its
/// peers, never anything not yet processed. `width_of` measures a candidate
/// string's rendered width (egui text layout in production; a synthetic
/// closure in tests, which is what makes this pure-logic testable, AC-19).
pub fn resolve_collisions(
    ranked: &[LabelCandidate],
    width_of: &dyn Fn(&str) -> f32,
) -> Vec<PlacedLabel> {
    let mut claimed: Vec<(f32, f32)> = Vec::new();
    let mut out = Vec::with_capacity(ranked.len());
    for cand in ranked {
        let mut text = Some(cand.full_label.clone());
        loop {
            match text {
                Some(t) => {
                    let w = width_of(&t);
                    let interval = (cand.anchor_x - w / 2.0, cand.anchor_x + w / 2.0);
                    if claimed.iter().any(|&c| intervals_overlap(c, interval)) {
                        text = degrade_once(&t);
                        continue;
                    }
                    claimed.push(interval);
                    out.push(PlacedLabel {
                        track: cand.track,
                        text: Some(t),
                        interval,
                    });
                    break;
                }
                None => {
                    out.push(PlacedLabel {
                        track: cand.track,
                        text: None,
                        interval: (cand.anchor_x, cand.anchor_x),
                    });
                    break;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// §2.5: interpolation, fade, and per-track visual state
// ---------------------------------------------------------------------

/// The interpolatable geometry/ranking fields of an [`Annotation`] (§2.5:
/// everything except the label/detail text, which snaps on arrival).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Geom {
    low_hz: f64,
    high_hz: f64,
    anchor_hz: f64,
    confidence: f32,
    strength_db: f32,
}

impl From<&Annotation> for Geom {
    fn from(a: &Annotation) -> Self {
        Geom {
            low_hz: a.span.low_hz,
            high_hz: a.span.high_hz,
            anchor_hz: a.anchor_hz,
            confidence: a.confidence,
            strength_db: a.strength_db,
        }
    }
}

fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn lerp_f32(a: f32, b: f32, t: f64) -> f32 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t) as f32
}

fn interpolate(from: Geom, to: Geom, t: f64) -> Geom {
    Geom {
        low_hz: lerp_f64(from.low_hz, to.low_hz, t),
        high_hz: lerp_f64(from.high_hz, to.high_hz, t),
        anchor_hz: lerp_f64(from.anchor_hz, to.anchor_hz, t),
        confidence: lerp_f32(from.confidence, to.confidence, t),
        strength_db: lerp_f32(from.strength_db, to.strength_db, t),
    }
}

/// One track's render-owned visual state across frames.
#[derive(Debug, Clone)]
struct TrackVisual {
    from: Geom,
    to: Geom,
    from_t: f64,
    to_t: f64,
    label: String,
    detail: Vec<String>,
    appeared_at: f64,
    removed_at: Option<f64>,
}

impl TrackVisual {
    fn current_geom(&self, now: f64) -> Geom {
        let span = (self.to_t - self.from_t).max(1e-9);
        let t = ((now - self.from_t) / span).clamp(0.0, 1.0);
        interpolate(self.from, self.to, t)
    }

    fn fade_alpha(&self, now: f64) -> f32 {
        let in_alpha = ((now - self.appeared_at) / FADE_S).clamp(0.0, 1.0) as f32;
        match self.removed_at {
            // D-125 item 5: fully visible through the hold, then a linear
            // fade over the next `LABEL_FADE_S`. `update()` resets
            // `removed_at` to `None` the instant the track reappears in a
            // snapshot, at any point before this reaches 0 — that reset is
            // the fade cancellation the ruling asks for; nothing further is
            // needed here.
            Some(r) => {
                let since_removed = now - r;
                let out_alpha = if since_removed <= LABEL_HOLD_S {
                    1.0
                } else {
                    (1.0 - (since_removed - LABEL_HOLD_S) / LABEL_FADE_S).clamp(0.0, 1.0) as f32
                };
                in_alpha.min(out_alpha)
            }
            None => in_alpha,
        }
    }
}

/// A track's fully resolved visual state for the current frame, ready to
/// place and draw.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleTrack {
    /// The track.
    pub track: TrackId,
    /// Interpolated occupied-frequency extent.
    pub span: FreqSpanHz,
    /// Interpolated anchor frequency.
    pub anchor_hz: f64,
    /// Interpolated confidence.
    pub confidence: f32,
    /// Interpolated strength (SNR, dB).
    pub strength_db: f32,
    /// The latest label text — snaps immediately on arrival, never
    /// interpolated (§2.5: "text has no meaningful midpoint").
    pub label: String,
    /// The latest detail lines, likewise snapped.
    pub detail: Vec<String>,
    /// Fade in/out opacity, `0.0..=1.0`.
    pub alpha: f32,
    /// Whether declutter membership currently shows this track's label
    /// (§2.4) — `false` means shade-band-only, dimmer, no text.
    pub labelled: bool,
}

/// The render-owned overlay state: reads the latest [`AnnotationFeed`]
/// snapshot each frame and derives everything [`draw`] needs. Not `Copy` —
/// owned by the caller (the app) across frames, alongside its
/// [`crate::layout::ViewState`].
#[derive(Debug, Default)]
pub struct AnnotationOverlayState {
    tracks: HashMap<TrackId, TrackVisual>,
    membership: DeclutterMembership,
    last_arc: Option<Arc<Vec<Annotation>>>,
    last_arrival_t: Option<f64>,
    detail: DetailPanelState,
}

impl AnnotationOverlayState {
    /// A fresh overlay state — nothing tracked yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Poll `feed` and, if a **new** set was published since the last call
    /// (detected by `Arc` pointer identity — cheap, and exactly matches "a
    /// new analysis cycle happened" since the analysis worker publishes a
    /// fresh `Arc` each cycle while render polls every frame), update every
    /// track's interpolation targets, fade timers and declutter membership.
    /// A poll that sees the same `Arc` is a no-op: geometry keeps
    /// interpolating via `now` at draw time regardless.
    pub fn update(&mut self, feed: &AnnotationFeed, cap: usize, now: f64) {
        let snapshot = feed.snapshot();
        let is_new = self
            .last_arc
            .as_ref()
            .is_none_or(|prev| !Arc::ptr_eq(prev, &snapshot));
        if !is_new {
            return;
        }
        let interval = self.last_arrival_t.map_or(0.0, |t| (now - t).max(1e-6));
        self.last_arrival_t = Some(now);

        let present: std::collections::HashSet<TrackId> =
            snapshot.iter().map(|a| a.track).collect();

        for ann in snapshot.iter() {
            let geom = Geom::from(ann);
            match self.tracks.get_mut(&ann.track) {
                Some(tv) => {
                    let mid = tv.current_geom(now);
                    tv.from = mid;
                    tv.to = geom;
                    tv.from_t = now;
                    tv.to_t = now + interval;
                    tv.label = ann.label.clone();
                    tv.detail = ann.detail.clone();
                    tv.removed_at = None;
                }
                None => {
                    self.tracks.insert(
                        ann.track,
                        TrackVisual {
                            from: geom,
                            to: geom,
                            from_t: now,
                            to_t: now,
                            label: ann.label.clone(),
                            detail: ann.detail.clone(),
                            appeared_at: now,
                            removed_at: None,
                        },
                    );
                }
            }
        }
        for (id, tv) in self.tracks.iter_mut() {
            if !present.contains(id) && tv.removed_at.is_none() {
                tv.removed_at = Some(now);
            }
        }
        // Purge tracks whose hold-then-fade (D-125 item 5) has finished;
        // harmless to be a little late (the next new snapshot will always
        // catch up). Must not purge before `LABEL_HOLD_S + LABEL_FADE_S`,
        // or a track reappearing near the end of its hold would find its
        // entry already gone and restart from a fresh fade-in instead of
        // cancelling the fade.
        self.tracks.retain(|_, tv| {
            tv.removed_at
                .is_none_or(|r| now - r < LABEL_HOLD_S + LABEL_FADE_S)
        });

        let ranked = declutter_rank(&snapshot);
        let ranked_ids: Vec<TrackId> = ranked.iter().map(|&i| snapshot[i].track).collect();
        self.membership.update(&ranked_ids, cap);

        self.last_arc = Some(snapshot);
    }

    /// This frame's fully resolved visible tracks (any fade alpha > 0), in
    /// track-id order (D-110) — `self.tracks` is a `HashMap`, whose iteration
    /// order is randomised per process; drawing overlapping, alpha-blended
    /// tracks in that order made two identical captures composite a
    /// handful of pixels a level or two apart, which a byte-identical
    /// determinism check (unlike any perceptual one) does not absorb. Sorted
    /// here rather than by switching the field to an ordered map, so the
    /// hot path (`update`'s per-frame lookups) keeps `HashMap`'s O(1) access.
    pub fn visible(&self, now: f64) -> Vec<VisibleTrack> {
        let mut ids: Vec<&TrackId> = self.tracks.keys().collect();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| {
                let tv = &self.tracks[id];
                let alpha = tv.fade_alpha(now);
                if alpha <= 0.0 {
                    return None;
                }
                let g = tv.current_geom(now);
                Some(VisibleTrack {
                    track: *id,
                    span: FreqSpanHz {
                        low_hz: g.low_hz,
                        high_hz: g.high_hz,
                    },
                    anchor_hz: g.anchor_hz,
                    confidence: g.confidence,
                    strength_db: g.strength_db,
                    label: tv.label.clone(),
                    detail: tv.detail.clone(),
                    alpha,
                    labelled: self.membership.is_labelled(*id),
                })
            })
            .collect()
    }

    /// Live (currently present, not fading out) track count — the "N
    /// TRACKS" status-bar figure (situation 1/3).
    pub fn live_track_count(&self) -> usize {
        self.tracks
            .values()
            .filter(|tv| tv.removed_at.is_none())
            .count()
    }

    /// Clear every tracked annotation immediately (D-056, AC-9): a retune
    /// invalidates frequency-dependent state exactly like the persistence
    /// histogram, max-hold trace and waterfall ring. Call this coincident
    /// with those resets; the caller is also responsible for clearing (or
    /// letting the analysis worker re-clear) the underlying
    /// [`AnnotationFeed`] so stale data is not re-adopted as "new" on the
    /// next poll.
    pub fn clear(&mut self) {
        self.tracks.clear();
        self.membership = DeclutterMembership::new();
        self.last_arc = None;
        self.last_arrival_t = None;
        self.detail = DetailPanelState::default();
    }

    /// The detail-panel state (§2.6), for [`draw`]'s click/hover handling.
    pub fn detail_mut(&mut self) -> &mut DetailPanelState {
        &mut self.detail
    }

    /// The detail-panel state, read-only.
    pub fn detail(&self) -> &DetailPanelState {
        &self.detail
    }
}

// ---------------------------------------------------------------------
// §2.3/§3.5/§3.6: clip-to-edge and minimum-width geometry
// ---------------------------------------------------------------------

/// A band clipped to the visible `0..=1` view-fraction window, or `None` if
/// it does not intersect the window at all (AC-7: absent from the draw
/// list, not merely off-canvas).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClippedBand {
    /// Clipped low edge, view fraction `0..=1`.
    pub t_lo: f64,
    /// Clipped high edge, view fraction `0..=1`.
    pub t_hi: f64,
    /// The true low edge was outside the window (draw an edge-arrow glyph).
    pub clipped_lo: bool,
    /// The true high edge was outside the window.
    pub clipped_hi: bool,
    /// The label anchor's view fraction, clamped into `0..=1` if the true
    /// anchor was off-screen (AC-5: anchor at the clipped edge, never off
    /// canvas).
    pub label_t: f64,
    /// The anchor itself was off-screen.
    pub anchor_clipped: bool,
}

/// Clip a band's edges (view fractions, need not be ordered by magnitude
/// relative to `0..1`) to the visible window, or `None` if it does not
/// intersect at all.
pub fn clip_band(t_lo: f64, t_hi: f64, anchor_t: f64) -> Option<ClippedBand> {
    if t_hi < 0.0 || t_lo > 1.0 {
        return None;
    }
    Some(ClippedBand {
        t_lo: t_lo.max(0.0),
        t_hi: t_hi.min(1.0),
        clipped_lo: t_lo < 0.0,
        clipped_hi: t_hi > 1.0,
        label_t: anchor_t.clamp(0.0, 1.0),
        anchor_clipped: !(0.0..=1.0).contains(&anchor_t),
    })
}

/// Widen `(lo, hi)` (egui points) about its center to at least
/// [`MIN_BAND_DEVICE_PX`] **device** pixels (AC-6) — never narrower, never
/// stretched when already wider.
pub fn clamp_min_width_pt(lo: f32, hi: f32, pixels_per_point: f32) -> (f32, f32) {
    let min_pt = MIN_BAND_DEVICE_PX / pixels_per_point;
    let width = hi - lo;
    if width >= min_pt {
        return (lo, hi);
    }
    let center = (lo + hi) / 2.0;
    (center - min_pt / 2.0, center + min_pt / 2.0)
}

// ---------------------------------------------------------------------
// §2.6: the detail panel — open/close state machine
// ---------------------------------------------------------------------

/// How the detail panel is currently showing a track, if at all (§2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetailOpen {
    /// Opened by a non-dragged click — stays open until explicitly closed.
    Pinned(TrackId),
    /// Opened by a hover dwell — closes automatically on pointer-leave.
    Preview(TrackId),
}

/// Click/hover state driving the §2.6 detail panel: opens on a non-dragged
/// click or a [`HOVER_DWELL_S`] hover dwell; a hover-only open auto-closes on
/// pointer-leave, a click-open does not.
#[derive(Debug, Clone, Default)]
pub struct DetailPanelState {
    open: Option<DetailOpen>,
    /// Track currently hovered and the time the hover started, for dwell
    /// timing.
    hover_since: Option<(TrackId, f64)>,
}

impl DetailPanelState {
    /// A non-dragged click on `track`'s band/label/anchor: pins the panel
    /// open.
    pub fn click(&mut self, track: TrackId) {
        self.open = Some(DetailOpen::Pinned(track));
    }

    /// Explicit close (e.g. a close button, or clicking elsewhere) — clears
    /// a pin; a no-op for a hover preview, which closes itself on
    /// pointer-leave instead.
    pub fn close(&mut self) {
        if matches!(self.open, Some(DetailOpen::Pinned(_))) {
            self.open = None;
        }
    }

    /// The pointer is hovering `track` this frame (or `None` if hovering
    /// nothing annotated). Starts/continues the dwell timer and opens a
    /// preview once [`HOVER_DWELL_S`] elapses; leaving closes any open
    /// preview (never a pin).
    pub fn hover(&mut self, track: Option<TrackId>, now: f64) {
        match track {
            Some(id) => match self.hover_since {
                Some((h, since)) if h == id => {
                    if now - since >= HOVER_DWELL_S && self.open.is_none() {
                        self.open = Some(DetailOpen::Preview(id));
                    }
                }
                _ => self.hover_since = Some((id, now)),
            },
            None => {
                self.hover_since = None;
                if matches!(self.open, Some(DetailOpen::Preview(_))) {
                    self.open = None;
                }
            }
        }
    }

    /// The track the panel is currently showing, if any.
    pub fn open_track(&self) -> Option<TrackId> {
        match self.open {
            Some(DetailOpen::Pinned(id) | DetailOpen::Preview(id)) => Some(id),
            None => None,
        }
    }

    /// Whether the currently open panel (if any) is a pin, as opposed to a
    /// hover preview.
    pub fn is_pinned(&self) -> bool {
        matches!(self.open, Some(DetailOpen::Pinned(_)))
    }
}

// ---------------------------------------------------------------------
// Drawing (§2.1 shade band, §2.2/§2.3 label, §2.6 leader) — the one part
// of this module that needs an egui context.
// ---------------------------------------------------------------------

/// Draw the annotation overlay: shade bands (§2.1), fixed-lane labels with
/// leaders (§2.3), and unlabelled-but-marked anchor ticks (§2.4) — alongside
/// `chrome`'s existing chrome, using the same [`Layout`]/frequency mapping
/// axis labels use. Returns the track the pointer is over this frame, if
/// any (for the caller's hover/click handling).
#[allow(clippy::too_many_arguments)] // one egui draw call; every argument is a genuinely distinct input, not a group to bundle
pub fn draw(
    painter: &Painter,
    layout: &Layout,
    params: &DisplayParams,
    theme: &Theme,
    ppp: f32,
    level: AnnotationLevel,
    label_style: LabelStyle,
    visible: &[VisibleTrack],
) {
    let grid = layout.grid;
    if grid.width() <= 0.0 || visible.is_empty() {
        return;
    }

    // Geometry pass: map every visible track to clipped screen intervals,
    // dropping any whose span does not intersect the window at all (AC-7).
    struct Placed<'a> {
        v: &'a VisibleTrack,
        band: ClippedBand,
    }
    let mut placed: Vec<Placed> = Vec::new();
    for v in visible {
        let t_lo = freq_to_view_fraction(v.span.low_hz, params);
        let t_hi = freq_to_view_fraction(v.span.high_hz, params);
        let anchor_t = freq_to_view_fraction(v.anchor_hz, params);
        if let Some(band) = clip_band(t_lo, t_hi, anchor_t) {
            placed.push(Placed { v, band });
        }
    }
    if placed.is_empty() {
        return;
    }

    // Label attempt (Full mode only, declutter-membership-gated) — computed
    // before any drawing at all, fix pass 1 (the review's finding): with
    // every track's actual outcome (real text, or degraded all the way to
    // none) known up front, the two drawing passes below can be strictly
    // ordered — every non-text mark first, every label second — instead of
    // interleaving per track.
    let (lane_y, lane_baseline_y) = match label_style {
        // ≈16pt lane starting 4pt below the grid top.
        LabelStyle::SingleLine => (grid.min.y + 4.0 + 8.0, grid.min.y + 4.0 + 16.0),
        LabelStyle::Stacked => {
            let top = grid.min.y + 4.0;
            (top, top + STACKED_LANE_HEIGHT_PT)
        }
    };
    let font = FontId::monospace(11.0);
    let by_track: HashMap<TrackId, PlacedLabel> = if matches!(level, AnnotationLevel::Full) {
        let candidates: Vec<LabelCandidate> = placed
            .iter()
            .filter(|p| p.v.labelled)
            .map(|p| LabelCandidate {
                track: p.v.track,
                anchor_x: x_for_fraction(p.band.label_t, grid),
                full_label: p.v.label.clone(),
            })
            .collect();
        // Re-rank the candidates being placed (declutter's own total order,
        // §2.3) so ties/degradation are resolved best-first regardless of
        // `visible()`'s (unordered) iteration order.
        let rank_input: Vec<Annotation> = placed
            .iter()
            .filter(|p| p.v.labelled)
            .map(|p| Annotation {
                track: p.v.track,
                span: p.v.span,
                anchor_hz: p.v.anchor_hz,
                label: p.v.label.clone(),
                detail: p.v.detail.clone(),
                confidence: p.v.confidence,
                strength_db: p.v.strength_db,
            })
            .collect();
        let order = declutter_rank(&rank_input);
        let ranked_candidates: Vec<LabelCandidate> =
            order.iter().map(|&i| candidates[i].clone()).collect();
        // Collision width (D-102 item 1): the stacked style's real footprint
        // is its *widest line*, not the flat grammar string's own width —
        // the same abbreviation ladder (`degrade_once`, on the flat string)
        // still decides what content survives; only how that content's
        // width is measured for collision purposes changes with the style.
        let measure = |s: &str| {
            painter
                .layout_no_wrap(s.to_owned(), font.clone(), theme.text)
                .size()
                .x
        };
        let width_of = |s: &str| match label_style {
            LabelStyle::SingleLine => measure(s),
            LabelStyle::Stacked => stack_label(s)
                .iter()
                .map(|line| measure(line))
                .fold(0.0f32, f32::max),
        };
        resolve_collisions(&ranked_candidates, &width_of)
            .into_iter()
            .map(|p| (p.track, p))
            .collect()
    } else {
        HashMap::new()
    };
    let text_of = |p: &Placed| -> Option<&str> {
        p.v.labelled
            .then(|| by_track.get(&p.v.track))
            .flatten()
            .and_then(|pl| pl.text.as_deref())
    };

    // Pass 1 (fix pass 1): every band, edge, arrow, leader and anchor tick,
    // for every track — nothing text-shaped draws yet. This is what makes
    // pass 2 safe: whichever order this loop happens to visit tracks in
    // (declutter membership, not iteration order, decided *what* draws;
    // this loop only decides *when*), no mark from any track can still be
    // painted after any label once pass 2 begins.
    for p in &placed {
        let (lo, hi) = clamp_min_width_pt(
            x_for_fraction(p.band.t_lo, grid),
            x_for_fraction(p.band.t_hi, grid),
            ppp,
        );
        let alpha = |a: u8| (f32::from(a) * p.v.alpha) as u8;
        let dim = if p.v.labelled || matches!(level, AnnotationLevel::ShadeOnly) {
            1.0
        } else {
            0.5
        };
        let fill =
            Color32::from_rgba_unmultiplied(255, 255, 255, alpha((HALO_FILL_ALPHA * dim) as u8));
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(lo, grid.min.y), Pos2::new(hi, grid.max.y)),
            0.0,
            fill,
        );
        let casing =
            Color32::from_rgba_unmultiplied(0, 0, 0, alpha((HALO_CASING_ALPHA * dim) as u8));
        let core =
            Color32::from_rgba_unmultiplied(255, 255, 255, alpha((HALO_CORE_ALPHA * dim) as u8));
        for (x, clipped, points_left) in [
            (lo, p.band.clipped_lo, true),
            (hi, p.band.clipped_hi, false),
        ] {
            let x = snap(x, ppp);
            painter.line_segment(
                [Pos2::new(x, grid.min.y), Pos2::new(x, grid.max.y)],
                Stroke::new(2.0 / ppp, casing),
            );
            painter.line_segment(
                [Pos2::new(x, grid.min.y), Pos2::new(x, grid.max.y)],
                Stroke::new(1.0 / ppp, core),
            );
            if clipped {
                draw_edge_arrow(painter, x, grid, points_left, core);
            }
        }

        let leader_x = snap(x_for_fraction(p.band.label_t, grid), ppp);
        match text_of(p) {
            Some(_) => {
                painter.line_segment(
                    [Pos2::new(leader_x, lane_y), Pos2::new(leader_x, grid.min.y)],
                    Stroke::new(1.0 / ppp, theme.text_dim.gamma_multiply(p.v.alpha)),
                );
            }
            // §2.4 unlabelled-but-marked (beyond the cap), §2.3's last
            // abbreviation rung (degraded all the way to nothing), and
            // every track in ShadeOnly mode: no text, but the anchor is
            // still reachable — a small filled anchor-tick triangle at the
            // lane's baseline (AC-3, AC-4).
            None => {
                let tick_color = theme.text_dim.gamma_multiply(p.v.alpha);
                draw_anchor_tick(painter, leader_x, lane_baseline_y, tick_color);
            }
        }
    }

    // Pass 2 (fix pass 1): every label — backing plate then text — drawn
    // strictly after every mark pass 1 could possibly have placed near it,
    // for any track. A tick or an edge can no longer land on top of a
    // label's plate or text, regardless of which track pass 1 happened to
    // visit last.
    for p in &placed {
        let Some(text) = text_of(p) else { continue };
        let leader_x = snap(x_for_fraction(p.band.label_t, grid), ppp);
        let alpha_u8 = |a: u8| (f32::from(a) * p.v.alpha) as u8;
        let lines: Vec<String> = match label_style {
            LabelStyle::SingleLine => vec![text.to_string()],
            LabelStyle::Stacked => stack_label(text),
        };
        draw_label(painter, &font, theme, leader_x, lane_y, &lines, &alpha_u8);
    }
}

/// Edge-arrow glyph (AC-5): a small filled triangle at the grid's vertical
/// center, pointing off-canvas on whichever side a band's true extent was
/// clipped to the visible window.
fn draw_edge_arrow(painter: &Painter, x: f32, grid: Rect, points_left: bool, color: Color32) {
    let y = grid.center().y;
    let tip_x = if points_left {
        x - EDGE_ARROW_SIZE
    } else {
        x + EDGE_ARROW_SIZE
    };
    let points = vec![
        Pos2::new(x, y - EDGE_ARROW_SIZE),
        Pos2::new(x, y + EDGE_ARROW_SIZE),
        Pos2::new(tip_x, y),
    ];
    painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
}

/// Anchor-tick glyph (§2.4/AC-3/AC-4): a small filled triangle at the fixed
/// label lane's baseline, marking a track's anchor when no label text is
/// shown there (beyond the declutter cap, degraded to nothing by a
/// collision, or ShadeOnly mode) — the track stays reachable (detail on
/// click/hover, §2.6) even with no text.
fn draw_anchor_tick(painter: &Painter, x: f32, baseline_y: f32, color: Color32) {
    let points = vec![
        Pos2::new(x - ANCHOR_TICK_HALF_WIDTH, baseline_y),
        Pos2::new(x + ANCHOR_TICK_HALF_WIDTH, baseline_y),
        Pos2::new(x, baseline_y + ANCHOR_TICK_HEIGHT),
    ];
    painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
}

/// Edge-arrow triangle half-height/reach, points.
const EDGE_ARROW_SIZE: f32 = 5.0;
/// Anchor-tick triangle half-width, points.
const ANCHOR_TICK_HALF_WIDTH: f32 = 3.0;
/// Anchor-tick triangle height, points.
const ANCHOR_TICK_HEIGHT: f32 = 5.0;

/// Build one line's `LayoutJob`, styling the trailing `~<signature>` run (if
/// present) in the dimmer/italic idiom (§2.2, AC-16): a genuine italic slant
/// via egui's own `TextFormat::italics` (a synthetic shear applied at the
/// glyph mesh, so it works with the one face this chrome installs — no
/// separate italic font file needed) — a mechanical split on the `~` marker,
/// not a re-parse of the measurement fields it stands in for. Shared by both
/// [`LabelStyle`]s: a stacked label's third line carries the same optional
/// `~<signature>` run a single-line label would, so it needs the same split.
fn build_line_job(
    font: &FontId,
    theme: &Theme,
    text: &str,
    alpha: &dyn Fn(u8) -> u8,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let measured_color =
        Color32::from_rgba_unmultiplied(theme.text.r(), theme.text.g(), theme.text.b(), alpha(255));
    match text.rfind(" · ~") {
        // The `~` marker appears only in the trailing signature run (§2.2),
        // so a single split point (the last `~`) separates the measured
        // content from it — never a substring match inside an ordinary
        // token.
        Some(pos) => {
            let (measured, sig) = text.split_at(pos);
            job.append(
                measured,
                0.0,
                egui::text::TextFormat {
                    font_id: font.clone(),
                    color: measured_color,
                    ..Default::default()
                },
            );
            let sig_color = Color32::from_rgba_unmultiplied(
                theme.text_dim.r(),
                theme.text_dim.g(),
                theme.text_dim.b(),
                alpha(255),
            );
            job.append(
                sig,
                0.0,
                egui::text::TextFormat {
                    font_id: font.clone(),
                    color: sig_color,
                    italics: true,
                    ..Default::default()
                },
            );
        }
        None => {
            job.append(
                text,
                0.0,
                egui::text::TextFormat {
                    font_id: font.clone(),
                    color: measured_color,
                    ..Default::default()
                },
            );
        }
    }
    job
}

/// Draw one label, one or more lines (D-102 item 1: the single-line style
/// passes one line; the stacked style passes up to three, top to bottom,
/// each centred on the same `x` anchor).
fn draw_label(
    painter: &Painter,
    font: &FontId,
    theme: &Theme,
    x: f32,
    y: f32,
    lines: &[String],
    alpha: &dyn Fn(u8) -> u8,
) {
    let galleys: Vec<_> = lines
        .iter()
        .map(|line| painter.layout_job(build_line_job(font, theme, line, alpha)))
        .collect();
    let widest = galleys.iter().map(|g| g.size().x).fold(0.0f32, f32::max);
    let total_height = galleys.iter().map(|g| g.size().y).sum::<f32>()
        + STACKED_LINE_GAP * galleys.len().saturating_sub(1) as f32;
    let block_rect =
        Align2::CENTER_TOP.anchor_size(Pos2::new(x, y), egui::Vec2::new(widest, total_height));

    // Backing plate (D-102 item 7): without it, the live trace and the §2.1
    // shade-band edges draw straight through the label text — "42.24K" reads
    // as "4|.24K" wherever a band edge crosses it. Painted **after** the
    // bands/edges above (so it hides them) and **before** the text below (so
    // the text sits on top of it), just larger than the glyphs (the whole
    // stacked block, when there is more than one line), theme-agnostically
    // dark and only partly opaque so it reads as "text got a little quieter
    // background" rather than a chrome box drawn on the display. Reuses the
    // §2.1 halo's own near-black/casing-alpha pairing rather than inventing a
    // fourth colour/alpha for the same "stay legible over any colormap" job.
    let backing_rect =
        block_rect.expand2(egui::Vec2::new(LABEL_BACKING_PAD_X, LABEL_BACKING_PAD_Y));
    let backing = Color32::from_rgba_unmultiplied(0, 0, 0, alpha(LABEL_BACKING_ALPHA as u8));
    painter.rect_filled(backing_rect, LABEL_BACKING_ROUNDING, backing);

    let mut cursor_y = y;
    for galley in galleys {
        let rect = Align2::CENTER_TOP.anchor_size(Pos2::new(x, cursor_y), galley.size());
        cursor_y += galley.size().y + STACKED_LINE_GAP;
        painter.galley(rect.min, galley, theme.text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;
    use crate::zoom::ZoomSpan;

    /// Render `draw` for real (a genuine egui pass, not a copy of its
    /// logic) and flatten the resulting shapes, so tests can observe what
    /// actually gets painted rather than re-deriving it. The single-line
    /// style, for every test that isn't specifically exercising the stacked
    /// one.
    fn render(
        level: AnnotationLevel,
        visible: &[VisibleTrack],
        params: &DisplayParams,
        grid: Rect,
    ) -> Vec<egui::Shape> {
        render_styled(level, LabelStyle::SingleLine, visible, params, grid)
    }

    /// [`render`], with the label style also under test control (D-102 item
    /// 1).
    fn render_styled(
        level: AnnotationLevel,
        label_style: LabelStyle,
        visible: &[VisibleTrack],
        params: &DisplayParams,
        grid: Rect,
    ) -> Vec<egui::Shape> {
        let ctx = egui::Context::default();
        let theme = Theme::default();
        // Real font metrics matter here (collision resolution measures
        // actual text width) — without installing the chrome's font, this
        // context has none (default_fonts is off, LC-2), and every string
        // would measure zero-width.
        theme.install(&ctx);
        let layout = Layout { grid };
        let input = egui::RawInput {
            screen_rect: Some(grid),
            ..Default::default()
        };
        let output = ctx.run_ui(input, |ui| {
            draw(
                ui.painter(),
                &layout,
                params,
                &theme,
                1.0,
                level,
                label_style,
                visible,
            );
        });
        let mut shapes = Vec::new();
        fn flatten(shape: &egui::Shape, out: &mut Vec<egui::Shape>) {
            match shape {
                egui::Shape::Vec(nested) => {
                    for s in nested {
                        flatten(s, out);
                    }
                }
                other => out.push(other.clone()),
            }
        }
        for clipped in &output.shapes {
            flatten(&clipped.shape, &mut shapes);
        }
        output.drop_without_applying_deltas();
        shapes
    }

    fn visible_track(id: u64, anchor_hz: f64, label: &str, labelled: bool) -> VisibleTrack {
        VisibleTrack {
            track: TrackId(id),
            span: FreqSpanHz {
                low_hz: anchor_hz - 5_000.0,
                high_hz: anchor_hz + 5_000.0,
            },
            anchor_hz,
            confidence: 0.9,
            strength_db: 20.0,
            label: label.to_string(),
            detail: vec![],
            alpha: 1.0,
            labelled,
        }
    }

    fn rated_params(zoom: ZoomSpan) -> DisplayParams {
        DisplayParams {
            sample_rate: Some(1_000_000.0),
            center: Some(0.0),
            zoom,
            ..Default::default()
        }
    }

    fn ann(id: u64, strength: f32, confidence: f32, anchor_hz: f64, label: &str) -> Annotation {
        Annotation {
            track: TrackId(id),
            span: FreqSpanHz {
                low_hz: anchor_hz - 5_000.0,
                high_hz: anchor_hz + 5_000.0,
            },
            anchor_hz,
            label: label.to_string(),
            detail: vec!["CENTER ...".to_string()],
            confidence,
            strength_db: strength,
        }
    }

    // --- AC-19 / §2.3/§2.4: declutter total order -------------------------

    #[test]
    fn ac19_declutter_rank_is_a_strict_total_order_independent_of_input_order() {
        let a = vec![
            ann(3, 20.0, 0.5, 100.0, "c"),
            ann(1, 30.0, 0.5, 0.0, "a"),
            ann(2, 30.0, 0.5, -100.0, "b"), // tie on strength/confidence with id 1
        ];
        let order = declutter_rank(&a);
        let ids: Vec<u64> = order.iter().map(|&i| a[i].track.0).collect();
        // Strength desc: {1,2} (30dB) before 3 (20dB); tie broken by
        // anchor_hz asc: 2 (-100) before 1 (0).
        assert_eq!(ids, vec![2, 1, 3]);

        // Shuffled input, same result.
        let mut shuffled = a.clone();
        shuffled.reverse();
        let order2 = declutter_rank(&shuffled);
        let ids2: Vec<u64> = order2.iter().map(|&i| shuffled[i].track.0).collect();
        assert_eq!(ids, ids2);
    }

    // --- AC-3: cap + tie-broken membership ---------------------------------

    #[test]
    fn ac3_top_n_by_rank_get_membership_tie_broken_by_frequency() {
        let a: Vec<Annotation> = (0..8)
            .map(|i| {
                ann(
                    i,
                    if i < 2 { 30.0 } else { 10.0 },
                    0.5,
                    i as f64 * 1000.0,
                    "x",
                )
            })
            .collect();
        let order = declutter_rank(&a);
        let ranked_ids: Vec<TrackId> = order.iter().map(|&i| a[i].track).collect();
        let mut m = DeclutterMembership::new();
        m.update(&ranked_ids, 5);
        let labelled: usize = a.iter().filter(|x| m.is_labelled(x.track)).count();
        assert_eq!(labelled, 5);
        // The two strongest (30dB) are always in; among the eight 10dB-tier
        // tracks minus none... here ids 0,1 are 30dB so labelled regardless;
        // remaining 3 slots go to the lowest anchor_hz among the rest.
        assert!(m.is_labelled(TrackId(0)) && m.is_labelled(TrackId(1)));
    }

    #[test]
    fn membership_hysteresis_needs_two_consecutive_snapshots_to_drop() {
        let mut m = DeclutterMembership::new();
        let ids: Vec<TrackId> = (0..5).map(TrackId).collect();
        m.update(&ids, 3);
        assert!(m.is_labelled(TrackId(2)));
        // TrackId(2) drops to rank 4 (two outside the cap of 3) once.
        let bumped = vec![TrackId(10), TrackId(11), TrackId(0), TrackId(1), TrackId(2)];
        m.update(&bumped, 3);
        assert!(
            m.is_labelled(TrackId(2)),
            "one bad snapshot must not drop it"
        );
        m.update(&bumped, 3);
        assert!(
            !m.is_labelled(TrackId(2)),
            "two consecutive bad snapshots must drop it"
        );
    }

    #[test]
    fn one_position_outside_the_cap_never_flickers() {
        let mut m = DeclutterMembership::new();
        let ids: Vec<TrackId> = (0..4).map(TrackId).collect(); // TrackId(3) at rank 3, cap 3 => exactly one outside
        m.update(&ids, 3);
        assert!(
            !m.is_labelled(TrackId(3)),
            "never labelled in the first place stays unlabelled"
        );
        // But a track that WAS labelled and dips to exactly one-outside
        // keeps its label indefinitely (grace, not a 2-strike countdown).
        let mut m2 = DeclutterMembership::new();
        m2.update(&[TrackId(0), TrackId(1), TrackId(2)], 3);
        assert!(m2.is_labelled(TrackId(2)));
        let one_outside = vec![TrackId(10), TrackId(11), TrackId(12), TrackId(2)];
        for _ in 0..10 {
            m2.update(&one_outside, 3);
            assert!(
                m2.is_labelled(TrackId(2)),
                "one-outside must never flicker off"
            );
        }
    }

    // --- AC-4/AC-19: collision resolution ----------------------------------

    #[test]
    fn ac4_lower_ranked_degrades_before_the_higher_ranked_is_ever_touched() {
        let ranked = vec![
            LabelCandidate {
                track: TrackId(1),
                anchor_x: 100.0,
                full_label: "FSK4 · 12.5K · 22dB · HOP · ~POCSAG".to_string(),
            },
            LabelCandidate {
                track: TrackId(2),
                anchor_x: 110.0, // close enough to collide at full width
                full_label: "OOK · 3K · 9dB".to_string(),
            },
        ];
        // Every string is "wide" (50pt) so the two candidates collide at
        // full size.
        let width_of = |_: &str| 50.0;
        let placed = resolve_collisions(&ranked, &width_of);
        assert_eq!(
            placed[0].text.as_deref(),
            Some("FSK4 · 12.5K · 22dB · HOP · ~POCSAG")
        );
        // The lower-ranked one must have degraded (shorter than its full
        // label, or empty).
        assert_ne!(placed[1].text.as_deref(), Some("OOK · 3K · 9dB"));
    }

    /// AC-19's full pipeline (declutter_rank, then resolve_collisions),
    /// driven from the *unranked* `Annotation` slice a `push_annotations`
    /// call would actually arrive as — genuinely permuted several different
    /// ways (not a reverse-then-reverse-back no-op), never just one already-
    /// sorted `LabelCandidate` order fed straight in.
    fn declutter_and_place(
        annotations: &[Annotation],
        width_of: &dyn Fn(&str) -> f32,
    ) -> Vec<PlacedLabel> {
        let order = declutter_rank(annotations);
        let ranked: Vec<LabelCandidate> = order
            .iter()
            .map(|&i| LabelCandidate {
                track: annotations[i].track,
                anchor_x: annotations[i].anchor_hz as f32,
                full_label: annotations[i].label.clone(),
            })
            .collect();
        resolve_collisions(&ranked, width_of)
    }

    #[test]
    fn ac19_n_way_collision_places_all_with_no_overlap_deterministically() {
        let annotations = vec![
            ann(1, 30.0, 0.9, 100.0, "SIGNAL · 1K · 30dB · HOP · ~A"),
            ann(2, 25.0, 0.9, 102.0, "SIGNAL · 1K · 25dB · HOP · ~B"),
            ann(3, 20.0, 0.9, 104.0, "SIGNAL · 1K · 20dB · HOP · ~C"),
            ann(4, 15.0, 0.9, 106.0, "SIGNAL · 1K · 15dB · HOP · ~D"),
        ];
        let width_of = |s: &str| s.len() as f32 * 6.0; // synthetic monospace metric
        let placed = declutter_and_place(&annotations, &width_of);
        // Only entries that still have text claim a real box; a degraded-to-
        // nothing entry draws no text (only the anchor tick, §2.3), so its
        // recorded zero-width point is not a box that can "overlap" anything.
        let boxed: Vec<&PlacedLabel> = placed.iter().filter(|p| p.text.is_some()).collect();
        for i in 0..boxed.len() {
            for j in (i + 1)..boxed.len() {
                assert!(
                    !intervals_overlap(boxed[i].interval, boxed[j].interval),
                    "{:?} overlaps {:?}",
                    boxed[i],
                    boxed[j]
                );
            }
        }

        // Several genuinely different (non-identity) permutations of the
        // *unranked* input — reverse, a rotation, and an interior swap —
        // each fed through the same full pipeline, must reproduce the
        // exact same placement byte-for-byte: the total order in
        // `declutter_rank` (§2.3), not arrival order, decides everything.
        let mut reversed = annotations.clone();
        reversed.reverse();
        assert_ne!(
            reversed.iter().map(|a| a.track).collect::<Vec<_>>(),
            annotations.iter().map(|a| a.track).collect::<Vec<_>>(),
            "the permutation must actually differ from the original"
        );
        assert_eq!(declutter_and_place(&reversed, &width_of), placed);

        let rotated: Vec<Annotation> = annotations[2..]
            .iter()
            .chain(&annotations[..2])
            .cloned()
            .collect();
        assert_eq!(declutter_and_place(&rotated, &width_of), placed);

        let mut swapped = annotations.clone();
        swapped.swap(0, 3);
        swapped.swap(1, 2);
        assert_eq!(declutter_and_place(&swapped, &width_of), placed);
    }

    #[test]
    fn degrade_ladder_strips_segments_right_to_left_then_empties() {
        let full = "FSK4 · 12.5K · 22dB · HOP · ~POCSAG";
        let step1 = degrade_once(full).unwrap();
        assert_eq!(step1, "FSK4 · 12.5K · 22dB · HOP");
        let step2 = degrade_once(&step1).unwrap();
        assert_eq!(step2, "FSK4 · 12.5K · 22dB");
        let step3 = degrade_once(&step2).unwrap();
        assert_eq!(step3, "FSK4 · 12.5K");
        let step4 = degrade_once(&step3).unwrap();
        assert_eq!(step4, "FSK4");
        assert_eq!(degrade_once(&step4), None);
    }

    // --- D-102 item 1: the stacked label style ------------------------------

    /// D-105: the owner chose stacked labels as the production default,
    /// superseding D-102 item 1's "single line stays the default". Single
    /// line stays in the code, reachable only by a caller setting
    /// `LabelStyle::SingleLine` directly (the look-review command).
    #[test]
    fn stacked_is_the_default_label_style() {
        assert_eq!(LabelStyle::default(), LabelStyle::Stacked);
    }

    #[test]
    fn stack_label_groups_fields_into_up_to_three_lines() {
        assert_eq!(stack_label("SIGNAL"), vec!["SIGNAL".to_string()]);
        assert_eq!(
            stack_label("SIGNAL · 1K"),
            vec!["SIGNAL".to_string(), "1K".to_string()]
        );
        assert_eq!(
            stack_label("SIGNAL · 1K · 20dB"),
            vec!["SIGNAL".to_string(), "1K".to_string(), "20dB".to_string()]
        );
        // A flag and a signature both land on the third line with power,
        // never growing past three lines.
        assert_eq!(
            stack_label("FSK4 · 12.5K · 22dB · HOP"),
            vec![
                "FSK4".to_string(),
                "12.5K".to_string(),
                "22dB · HOP".to_string()
            ]
        );
        assert_eq!(
            stack_label("FSK4 · 12.5K · 22dB · HOP · ~POCSAG"),
            vec![
                "FSK4".to_string(),
                "12.5K".to_string(),
                "22dB · HOP · ~POCSAG".to_string()
            ]
        );
    }

    #[test]
    fn stacked_style_draws_one_text_shape_per_line_centered_on_the_same_anchor() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(
            1,
            0.0,
            "FSK4 · 12.5K · 22dB · HOP · ~POCSAG",
            true,
        )];
        let shapes = render_styled(
            AnnotationLevel::Full,
            LabelStyle::Stacked,
            &visible,
            &params,
            grid,
        );
        let texts: Vec<&egui::epaint::TextShape> = shapes
            .iter()
            .filter_map(|s| match s {
                egui::Shape::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts.len(),
            3,
            "identity / bandwidth / power+flag+signature = 3 lines"
        );
        // Every line is centred on the same x anchor.
        let centers: Vec<f32> = texts
            .iter()
            .map(|t| t.pos.x + t.galley.size().x / 2.0)
            .collect();
        for c in &centers {
            assert!(
                (c - centers[0]).abs() < 1.0,
                "{centers:?} not all centered on the same anchor"
            );
        }
        // Stacked top to bottom, each line strictly below the last.
        let mut ys: Vec<f32> = texts.iter().map(|t| t.pos.y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            ys[0] < ys[1] && ys[1] < ys[2],
            "{ys:?} not stacked top-down"
        );
    }

    #[test]
    fn stacked_collision_uses_the_widest_line_not_the_flat_strings_total_width() {
        // A multi-segment label's *flat* string (all segments concatenated
        // with " · ") is always at least as wide as its widest *stacked*
        // line (a strict subset of the same characters) — so measuring
        // collision boxes with the flat width, as the single-line style
        // does, overstates a stacked label's real footprint and can trigger
        // a collision the stacked rendering would never actually have.
        let ranked = vec![
            LabelCandidate {
                track: TrackId(1),
                anchor_x: 0.0,
                full_label: "SIGNAL · 20dB".to_string(),
            },
            LabelCandidate {
                track: TrackId(2),
                anchor_x: 25.0,
                full_label: "X".to_string(),
            },
        ];
        let flat_width_of = |s: &str| s.len() as f32 * 6.0;
        let stacked_width_of = |s: &str| {
            stack_label(s)
                .iter()
                .map(|line| line.len() as f32 * 6.0)
                .fold(0.0f32, f32::max)
        };

        // The flat measurement ("SIGNAL · 20dB", 13 chars) reaches track 2's
        // anchor and forces it to degrade — all the way to nothing, since a
        // single-segment label ("X") has nowhere left to abbreviate to.
        let flat_placed = resolve_collisions(&ranked, &flat_width_of);
        assert_eq!(flat_placed[1].text, None);

        // The stacked measurement (widest line: "SIGNAL", 6 chars) does not
        // reach it — track 2 keeps its full text.
        let stacked_placed = resolve_collisions(&ranked, &stacked_width_of);
        assert_eq!(stacked_placed[1].text.as_deref(), Some("X"));
    }

    // --- AC-5: clip to edge / anchor fallback ------------------------------

    #[test]
    fn ac5_band_clips_to_the_grid_edge_and_absent_when_disjoint() {
        // Straddles the left edge.
        let band = clip_band(-0.2, 0.3, -0.2).unwrap();
        assert_eq!(band.t_lo, 0.0);
        assert_eq!(band.t_hi, 0.3);
        assert!(band.clipped_lo && !band.clipped_hi);
        assert_eq!(band.label_t, 0.0);
        assert!(band.anchor_clipped);

        // Fully outside: absent from the draw list entirely.
        assert!(clip_band(1.2, 1.5, 1.3).is_none());
        assert!(clip_band(-0.5, -0.1, -0.3).is_none());

        // Fully inside: untouched.
        let band = clip_band(0.4, 0.6, 0.5).unwrap();
        assert!(!band.clipped_lo && !band.clipped_hi && !band.anchor_clipped);
    }

    // --- AC-6: minimum band width -------------------------------------------

    #[test]
    fn ac6_band_never_renders_narrower_than_three_device_pixels() {
        let (lo, hi) = clamp_min_width_pt(100.0, 100.3, 2.0); // 0.6pt = 1.2 device px at ppp=2
        assert!((hi - lo) * 2.0 >= MIN_BAND_DEVICE_PX - 1e-6);
        // A band already wide enough is untouched (never stretched).
        let (lo2, hi2) = clamp_min_width_pt(100.0, 130.0, 1.0);
        assert_eq!((lo2, hi2), (100.0, 130.0));
    }

    // --- AC-7: frequency-to-pixel mapping matches axis::freq_at's inverse -

    #[test]
    fn ac7_freq_to_view_fraction_inverts_axis_freq_at() {
        for (rate, center) in [(2_400_000.0, None), (2_400_000.0, Some(2_450_000_000.0))] {
            let params = DisplayParams {
                sample_rate: Some(rate),
                center,
                zoom: ZoomSpan::new(0.2, 0.8),
                ..Default::default()
            };
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let crate::axis::FreqValue::Hz(hz) = crate::axis::freq_at(&params, t) else {
                    panic!("expected Hz");
                };
                let back = freq_to_view_fraction(hz, &params);
                assert!((back - t).abs() < 1e-9, "t={t} round-trips to {back}");
            }
        }
        // Rate-less: the normalized value is the domain freq_to_view_fraction
        // expects directly (D-098 §7.3 convention).
        let params = DisplayParams {
            sample_rate: None,
            center: None,
            zoom: ZoomSpan::FULL,
            ..Default::default()
        };
        for t in [0.0, 0.3, 0.7, 1.0] {
            let crate::axis::FreqValue::Normalized(v) = crate::axis::freq_at(&params, t) else {
                panic!("expected normalized");
            };
            let back = freq_to_view_fraction(v, &params);
            assert!((back - t).abs() < 1e-9);
        }
    }

    #[test]
    fn ac7_a_track_whose_span_never_intersects_the_view_is_absent() {
        let params = DisplayParams {
            sample_rate: Some(1_000_000.0),
            center: Some(0.0),
            zoom: ZoomSpan::new(0.4, 0.6), // visible: -100k..+100k
            ..Default::default()
        };
        let t_lo = freq_to_view_fraction(-900_000.0, &params);
        let t_hi = freq_to_view_fraction(-800_000.0, &params);
        assert!(clip_band(t_lo, t_hi, (t_lo + t_hi) / 2.0).is_none());
    }

    // --- AC-12: linear interpolation over the measured interval -----------

    #[test]
    fn ac12_geometry_interpolates_linearly_text_snaps_immediately() {
        let feed = AnnotationFeed::new();
        let mut state = AnnotationOverlayState::new();
        feed.push_annotations(&[ann(1, 20.0, 0.8, 100_000.0, "OOK · 1K · 20dB")]);
        state.update(&feed, 8, 0.0);
        feed.push_annotations(&[ann(1, 30.0, 0.9, 200_000.0, "OOK · 1K · 30dB")]);
        state.update(&feed, 8, 1.0); // arrives 1s after the first

        // Halfway through the (measured) 1s interval, geometry is halfway.
        let mid = state.visible(1.5);
        let t = mid.iter().find(|v| v.track == TrackId(1)).unwrap();
        assert!((t.anchor_hz - 150_000.0).abs() < 1.0, "{}", t.anchor_hz);
        assert!((t.strength_db - 25.0).abs() < 0.01);
        // Text snaps immediately on arrival — no midpoint string exists.
        assert_eq!(t.label, "OOK · 1K · 30dB");

        // Fully caught up (well past the interval): exactly the new value.
        let done = state.visible(2.0);
        let t = done.iter().find(|v| v.track == TrackId(1)).unwrap();
        assert!((t.anchor_hz - 200_000.0).abs() < 1e-6);
    }

    #[test]
    fn a_new_snapshot_with_the_same_arc_is_a_cheap_no_op() {
        let feed = AnnotationFeed::new();
        let mut state = AnnotationOverlayState::new();
        feed.push_annotations(&[ann(1, 20.0, 0.8, 0.0, "x")]);
        state.update(&feed, 8, 0.0);
        // Same content, polled again without a new push: no transition.
        state.update(&feed, 8, 0.5);
        state.update(&feed, 8, 10.0);
        let v = state.visible(10.0);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].anchor_hz, 0.0);
    }

    // --- AC-9/D-056: retune clears everything immediately -------------------

    #[test]
    fn ac9_clear_removes_every_track_and_resets_membership() {
        let feed = AnnotationFeed::new();
        let mut state = AnnotationOverlayState::new();
        feed.push_annotations(&[ann(1, 20.0, 0.8, 0.0, "x")]);
        state.update(&feed, 8, 0.0);
        assert_eq!(state.visible(FADE_S * 2.0).len(), 1);
        state.clear();
        assert!(state.visible(FADE_S * 2.0).is_empty());
        assert_eq!(state.live_track_count(), 0);
    }

    // --- Fade in/out ---------------------------------------------------------

    #[test]
    fn fade_in_ramps_then_disappearance_holds_then_fades_out_before_removal() {
        let feed = AnnotationFeed::new();
        let mut state = AnnotationOverlayState::new();
        feed.push_annotations(&[ann(1, 20.0, 0.8, 0.0, "x")]);
        state.update(&feed, 8, 0.0);
        // Alpha is exactly 0 at the birth instant, so the track is not yet
        // in the visible set at all (nothing to draw at zero opacity).
        assert!(state.visible(0.0).is_empty());
        let mid_fade = state.visible(FADE_S / 2.0);
        assert!(mid_fade[0].alpha > 0.0 && mid_fade[0].alpha < 1.0);
        let full = state.visible(FADE_S * 2.0);
        assert_eq!(full[0].alpha, 1.0);

        // Now remove it: D-125 item 5 — fully visible for LABEL_HOLD_S,
        // then a slow linear fade over LABEL_FADE_S, gone at the 10s total.
        let removed_t = FADE_S * 2.0;
        feed.push_annotations(&[]);
        state.update(&feed, 8, removed_t);
        let just_removed = state.visible(removed_t);
        assert_eq!(just_removed[0].alpha, 1.0);
        let still_held = state.visible(removed_t + LABEL_HOLD_S - 0.01);
        assert_eq!(
            still_held[0].alpha, 1.0,
            "label must stay fully visible through the whole hold"
        );
        let just_past_hold = state.visible(removed_t + LABEL_HOLD_S + 0.01);
        assert!(
            just_past_hold[0].alpha > 0.0 && just_past_hold[0].alpha < 1.0,
            "fade must have started just past the hold: alpha {}",
            just_past_hold[0].alpha
        );
        let mid_fade_out = state.visible(removed_t + LABEL_HOLD_S + LABEL_FADE_S / 2.0);
        assert!(
            (mid_fade_out[0].alpha - 0.5).abs() < 0.01,
            "linear fade should be about half-alpha halfway through: {}",
            mid_fade_out[0].alpha
        );
        let just_before_gone = state.visible(removed_t + LABEL_HOLD_S + LABEL_FADE_S - 0.01);
        assert!(!just_before_gone.is_empty() && just_before_gone[0].alpha > 0.0);
        let gone = state.visible(removed_t + LABEL_HOLD_S + LABEL_FADE_S + 0.01);
        assert!(gone.is_empty(), "must be fully gone at the 10s total");
    }

    #[test]
    fn a_track_returning_within_the_hold_cancels_the_fade() {
        // D-125 item 5, the ruling's last clause: "a track that returns
        // within the hold cancels the fade."
        let feed = AnnotationFeed::new();
        let mut state = AnnotationOverlayState::new();
        feed.push_annotations(&[ann(1, 20.0, 0.8, 0.0, "x")]);
        state.update(&feed, 8, 0.0);
        state.update(&feed, 8, FADE_S * 2.0); // fully faded in, no-op poll

        // Leaves the feed, well into its hold...
        feed.push_annotations(&[]);
        let removed_t = FADE_S * 2.0;
        state.update(&feed, 8, removed_t);
        let mid_hold = removed_t + LABEL_HOLD_S / 2.0;
        assert_eq!(state.visible(mid_hold)[0].alpha, 1.0);

        // ...then returns before the hold ends: alpha snaps back to fully
        // visible with no fade in progress, at any time after.
        feed.push_annotations(&[ann(1, 20.0, 0.8, 0.0, "x")]);
        state.update(&feed, 8, mid_hold);
        let long_after = mid_hold + LABEL_HOLD_S + LABEL_FADE_S;
        let v = state.visible(long_after);
        assert_eq!(v.len(), 1, "the track must still be tracked, not purged");
        assert_eq!(v[0].alpha, 1.0, "no fade should be in progress");
    }

    // --- AC-16: the ~signature run renders genuinely italic ---------------

    #[test]
    fn ac16_signature_suffix_renders_italic_and_dim_measured_text_does_not() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(1, 0.0, "SIGNAL · 1K · 20dB · ~POCSAG", true)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        let mut saw_italic = false;
        let mut saw_non_italic = false;
        for shape in &shapes {
            if let egui::Shape::Text(t) = shape {
                for section in &t.galley.job.sections {
                    if section.format.italics {
                        saw_italic = true;
                    } else {
                        saw_non_italic = true;
                    }
                }
            }
        }
        assert!(
            saw_italic,
            "the ~signature run must render with a real italic slant"
        );
        assert!(
            saw_non_italic,
            "the measured headline content must NOT be italicized"
        );
    }

    #[test]
    fn ac16_label_with_no_signature_has_no_italic_run() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(1, 0.0, "SIGNAL · 1K · 20dB", true)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        let any_italic = shapes.iter().any(|s| {
            matches!(s, egui::Shape::Text(t) if t.galley.job.sections.iter().any(|sec| sec.format.italics))
        });
        assert!(
            !any_italic,
            "no signature suffix means nothing to italicize"
        );
    }

    // --- D-102 item 7: the label backing plate ------------------------------

    #[test]
    fn label_backing_plate_draws_directly_behind_its_own_text_and_smaller_than_the_band() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(1, 0.0, "SIGNAL · 1K · 20dB", true)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);

        let text_idx = shapes
            .iter()
            .position(|s| matches!(s, egui::Shape::Text(_)))
            .expect("a labelled track must draw its text");
        // The backing must be the shape immediately preceding the text —
        // drawn after everything it must hide (the band/leader, both already
        // painted earlier in the pass) and immediately before the text it
        // exists to keep legible. Reverting the backing draw removes this
        // Rect and the very next assertion goes red deterministically.
        let backing = match &shapes[text_idx - 1] {
            egui::Shape::Rect(r) => r,
            other => {
                panic!("expected the backing plate directly before the label text, got {other:?}")
            }
        };
        // It must be a small plate hugging the text, not the full-height
        // shade band (also a `Shape::Rect`, drawn earlier in the pass).
        assert!(
            backing.rect.height() < grid.height() / 2.0,
            "backing plate {:?} looks like the full-height band, not a text plate",
            backing.rect
        );
        assert!(backing.rect.height() > 0.0 && backing.rect.width() > 0.0);
        // It is dark and only partly opaque (§2.1 halo casing alpha), never
        // fully opaque chrome.
        assert_eq!(backing.fill.r(), 0);
        assert_eq!(backing.fill.g(), 0);
        assert_eq!(backing.fill.b(), 0);
        assert!(backing.fill.a() > 0 && backing.fill.a() < 255);
    }

    #[test]
    fn no_backing_plate_when_no_text_is_drawn() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        // Beyond the cap: no label text, so nothing to back.
        let visible = vec![visible_track(1, 0.0, "SIGNAL · 1K · 20dB", false)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        assert!(!shapes.iter().any(|s| matches!(s, egui::Shape::Text(_))));
        // Only the band's own full-height Rect remains — no small plate.
        let rects: Vec<&egui::epaint::RectShape> = shapes
            .iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r) => Some(r),
                _ => None,
            })
            .collect();
        assert!(
            rects.iter().all(|r| r.rect.height() >= grid.height() / 2.0),
            "no small backing plate should draw when there is no label text"
        );
    }

    /// Fix pass 1 (review finding): a per-track band/edge/tick/leader must
    /// never land on top of *another* track's already-drawn label — the
    /// original `ac16`-style test only checked a label's plate against its
    /// own text, which cannot catch a **different** track's mark painted
    /// later. Whole-shape-list version: once a label's plate has drawn,
    /// nothing non-text painted afterward may touch its rect, for any
    /// track. `B` is listed *after* `A` deliberately — the single
    /// interleaved-per-track loop this fix pass replaced would draw `B`'s
    /// anchor tick strictly after `A`'s already-drawn label in exactly this
    /// input order, so reverting the two-pass split makes this go red
    /// deterministically (the fix pass 1 negative control). Run for both
    /// styles (D-105: the production default is now [`LabelStyle::Stacked`],
    /// so this must hold for it, not only the look-review-only single line).
    fn assert_no_non_text_shape_paints_after_a_labels_plate(label_style: LabelStyle) {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let a = visible_track(1, 0.0, "SIGNAL · 12.5K · 22dB · HOP · ~POCSAG", true);
        // B's anchor sits well inside A's (wide) label's rendered box, and
        // B is unlabelled, so it draws only an anchor tick — the exact
        // shape the review found sitting on top of a neighboring label.
        let b = visible_track(2, 50_000.0, "SIGNAL · 1K · 10dB", false);
        let visible = [a, b];
        let shapes = render_styled(AnnotationLevel::Full, label_style, &visible, &params, grid);

        // Locate every label's plate: a Rect immediately followed by a
        // Text (the same convention `label_backing_plate_...` already
        // established for one label; here applied across the whole list).
        // For the stacked style, several Text shapes (one per line) follow
        // the same plate — the loop below skips every Text shape after the
        // plate for exactly that reason.
        let mut plates: Vec<(usize, Rect)> = Vec::new();
        for i in 0..shapes.len().saturating_sub(1) {
            if let (egui::Shape::Rect(r), egui::Shape::Text(_)) = (&shapes[i], &shapes[i + 1]) {
                plates.push((i, r.rect));
            }
        }
        assert_eq!(plates.len(), 1, "exactly one label draws in this scene");
        let (plate_idx, plate_rect) = plates[0];

        fn touches(shape: &egui::Shape, rect: Rect) -> bool {
            match shape {
                egui::Shape::LineSegment { points, .. } => points.iter().any(|p| rect.contains(*p)),
                egui::Shape::Path(p) => p.points.iter().any(|pt| rect.contains(*pt)),
                egui::Shape::Rect(r) => rect.intersects(r.rect),
                _ => false,
            }
        }

        for (i, s) in shapes.iter().enumerate() {
            if i <= plate_idx || matches!(s, egui::Shape::Text(_)) {
                // Shapes at/before the plate, and the plate's own text
                // (every line, for the stacked style) immediately after
                // it, are expected.
                continue;
            }
            assert!(
                !touches(s, plate_rect),
                "shape #{i} ({s:?}) painted after label plate #{plate_idx} \
                 ({plate_rect:?}) touches its rect — something drew on top \
                 of the label ({label_style:?} style)"
            );
        }
    }

    #[test]
    fn no_non_text_shape_ever_paints_after_a_labels_plate_inside_its_rect_single_line() {
        assert_no_non_text_shape_paints_after_a_labels_plate(LabelStyle::SingleLine);
    }

    #[test]
    fn no_non_text_shape_ever_paints_after_a_labels_plate_inside_its_rect_stacked() {
        assert_no_non_text_shape_paints_after_a_labels_plate(LabelStyle::Stacked);
    }

    // --- AC-3/AC-4: anchor ticks for unlabelled/degraded tracks -----------

    #[test]
    fn ac3_unlabelled_beyond_cap_draws_no_text_but_does_draw_its_anchor_tick() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(1, 200_000.0, "SIGNAL · 1K · 20dB", false)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        assert!(
            !shapes.iter().any(|s| matches!(s, egui::Shape::Text(_))),
            "an unlabelled (beyond-cap) track must draw no label text"
        );
        assert!(
            shapes
                .iter()
                .any(|s| matches!(s, egui::Shape::Path(p) if p.points.len() == 3)),
            "an unlabelled track must still draw its anchor-tick triangle"
        );
    }

    #[test]
    fn ac4_a_label_degraded_all_the_way_to_nothing_still_gets_an_anchor_tick() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        // Identical anchors: the weaker (lower-ranked) label collides with
        // the stronger one's claimed interval at every abbreviation step,
        // including the empty one, since the stronger label never moves.
        let strong = visible_track(1, 0.0, "SIGNAL · 1K · 30dB · HOP · ~STRONG", true);
        let mut weak = visible_track(2, 0.0, "SIGNAL · 1K · 10dB · HOP · ~WEAK", true);
        weak.strength_db = 5.0;
        let visible = [strong, weak];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        assert_eq!(
            shapes
                .iter()
                .filter(|s| matches!(s, egui::Shape::Text(_)))
                .count(),
            1,
            "only the higher-ranked label keeps its text"
        );
        assert_eq!(
            shapes
                .iter()
                .filter(|s| matches!(s, egui::Shape::Path(p) if p.points.len() == 3))
                .count(),
            1,
            "the fully-degraded track must still show its own anchor tick"
        );
    }

    #[test]
    fn shade_only_mode_draws_ticks_for_every_track_and_no_text_at_all() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![
            visible_track(1, -200_000.0, "SIGNAL · 1K · 30dB", true),
            visible_track(2, 200_000.0, "SIGNAL · 1K · 10dB", false),
        ];
        let shapes = render(AnnotationLevel::ShadeOnly, &visible, &params, grid);
        assert!(
            !shapes.iter().any(|s| matches!(s, egui::Shape::Text(_))),
            "ShadeOnly must never draw label text, labelled or not"
        );
        assert_eq!(
            shapes
                .iter()
                .filter(|s| matches!(s, egui::Shape::Path(p) if p.points.len() == 3))
                .count(),
            2,
            "every track gets an anchor tick in ShadeOnly mode"
        );
    }

    // --- AC-5: edge-arrow glyph on a clipped band --------------------------

    #[test]
    fn ac5_edge_arrow_draws_on_the_clipped_side_and_not_the_open_one() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        // Zoom to the middle fifth of a 1 MHz span: visible -100k..+100k Hz.
        let params = rated_params(ZoomSpan::new(0.4, 0.6));
        let mut v = visible_track(1, -50_000.0, "SIGNAL · 1K · 20dB", true);
        // True span extends far past the left edge of the visible window;
        // the right edge (-50k) sits inside it.
        v.span = FreqSpanHz {
            low_hz: -900_000.0,
            high_hz: -50_000.0,
        };
        let shapes = render(AnnotationLevel::Full, &[v], &params, grid);
        let arrow_y = grid.center().y;
        let triangles: Vec<&egui::epaint::PathShape> = shapes
            .iter()
            .filter_map(|s| match s {
                egui::Shape::Path(p) if p.points.len() == 3 => Some(p),
                _ => None,
            })
            .collect();
        let arrow_triangles: Vec<&&egui::epaint::PathShape> = triangles
            .iter()
            .filter(|p| p.points.iter().any(|pt| (pt.y - arrow_y).abs() < 1.0))
            .collect();
        assert_eq!(
            arrow_triangles.len(),
            1,
            "expected exactly one edge-arrow triangle (the clipped left edge only)"
        );
        // It must point left (its tip x is less than the two base points' x).
        let tri = arrow_triangles[0];
        let tip = tri
            .points
            .iter()
            .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
            .unwrap();
        let base_x = tri
            .points
            .iter()
            .filter(|p| *p != tip)
            .map(|p| p.x)
            .fold(0.0, f32::max);
        assert!(
            tip.x < base_x,
            "the edge arrow at a left-clipped band must point left"
        );
    }

    #[test]
    fn no_edge_arrow_when_the_band_is_fully_visible() {
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        let params = rated_params(ZoomSpan::FULL);
        let visible = vec![visible_track(1, 0.0, "SIGNAL · 1K · 20dB", true)];
        let shapes = render(AnnotationLevel::Full, &visible, &params, grid);
        let arrow_y = grid.center().y;
        let any_arrow = shapes.iter().any(|s| match s {
            egui::Shape::Path(p) => {
                p.points.len() == 3 && p.points.iter().any(|pt| (pt.y - arrow_y).abs() < 1.0)
            }
            _ => false,
        });
        assert!(!any_arrow, "a fully visible band must draw no edge arrow");
    }

    // --- §2.6 hit-testing: never narrower than the minimum hit region -----

    #[test]
    fn hit_test_widens_a_sub_bin_wide_band_to_the_minimum_hit_region() {
        let params = DisplayParams {
            sample_rate: Some(1_000_000.0),
            center: Some(0.0),
            zoom: ZoomSpan::FULL,
            ..Default::default()
        };
        let grid = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 500.0));
        // A track a fraction of a Hz wide at DC: t=0.5, pixel x=500.
        let visible = vec![VisibleTrack {
            track: TrackId(1),
            span: FreqSpanHz {
                low_hz: -0.01,
                high_hz: 0.01,
            },
            anchor_hz: 0.0,
            confidence: 0.9,
            strength_db: 20.0,
            label: "x".to_string(),
            detail: vec![],
            alpha: 1.0,
            labelled: true,
        }];
        assert_eq!(hit_test(&visible, &params, grid, 500.0), Some(TrackId(1)));
        // Still hits a few points away, inside the widened region.
        assert_eq!(hit_test(&visible, &params, grid, 504.0), Some(TrackId(1)));
        assert_eq!(hit_test(&visible, &params, grid, 550.0), None);
    }

    // --- §2.6 detail panel state machine (AC-17) ---------------------------

    #[test]
    fn ac17_click_pins_hover_previews_and_leaving_closes_preview_not_pin() {
        let mut d = DetailPanelState::default();
        assert_eq!(d.open_track(), None);

        // Hover dwell opens a preview.
        d.hover(Some(TrackId(1)), 0.0);
        assert_eq!(d.open_track(), None, "not yet dwelled");
        d.hover(Some(TrackId(1)), HOVER_DWELL_S + 0.01);
        assert_eq!(d.open_track(), Some(TrackId(1)));
        assert!(!d.is_pinned());

        // Leaving closes the preview automatically.
        d.hover(None, HOVER_DWELL_S + 0.1);
        assert_eq!(d.open_track(), None);

        // A click pins it, and hovering elsewhere/leaving does not close it.
        d.click(TrackId(2));
        assert_eq!(d.open_track(), Some(TrackId(2)));
        assert!(d.is_pinned());
        d.hover(None, 100.0);
        assert_eq!(
            d.open_track(),
            Some(TrackId(2)),
            "a pin survives pointer-leave"
        );
        d.close();
        assert_eq!(d.open_track(), None);
    }
}
