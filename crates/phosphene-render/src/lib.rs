// SPDX-License-Identifier: MIT

//! # phosphene-render — data surface + chrome (product spec §8.1)
//!
//! ## Architecture
//!
//! This crate owns everything that puts pixels on screen and nothing else: no
//! DSP math, no source I/O (the lane contract of `plan.md`). It renders a
//! calibrated spectrum handed to it as a slice of dBFS/bin values.
//!
//! Two layers compose each frame, in one render pass, into any
//! `wgpu::TextureView`:
//!
//! * the **data surface** — the persistence histogram (FR-D1, §7.3 render
//!   half) and waterfall (FR-D4, §7.6) drawn as textured quads through the
//!   [`colormap`] LUT, plus grid and live trace built as colored triangles
//!   on the CPU ([`scene`]) and drawn by a single wgpu pipeline
//!   ([`gpu::SceneRenderer`]); the [`waterfall`] module holds the §7.6 row
//!   cadence/aggregation arithmetic, GPU-free;
//! * the **chrome** — wordmark, status readouts, axis labels, cursor
//!   readout, zoom gestures, keyboard map and help overlay — drawn by egui
//!   ([`chrome`]), styled to the D-010 identity ([`theme`]).
//!
//! Axis labels and grid are computed from one [`layout::Layout`], which is
//! what guarantees their exact alignment (clarification C6). The windowed and
//! headless paths share [`gpu::FrameComposer`], so the CI golden PNG exercises
//! the same code a user sees.
//!
//! Three GPU-free modules carry the M1-C display truthfulness contract:
//! [`axis`] holds every number the instrument states (tick labels and the
//! FR-D6 cursor readout share one mapping, so they agree by construction —
//! and a rate-less source renders normalized frequency with no Hz unit
//! anywhere, D-028 §2); [`zoom`] holds the FR-D6 display-side zoom window
//! (D-011: it reinterprets the existing FFT, never the device); [`keymap`]
//! holds the FR-D7 binding table, from which the `?` help overlay is
//! generated so no binding can exist that the overlay does not list. The
//! caller owns a [`layout::ViewState`], hands it mutably to [`chrome::draw`]
//! and immutably to the composer, and forwards the returned
//! [`keymap::ChromeRequests`] to [`gpu::FrameComposer::apply_requests`].
//!
//! One frozen seam looks outward instead of at pixels: [`annotation`] carries
//! the next-gen overlay feed (`push_annotations(&[Annotation])`, D-017) — a
//! no-op in v1 until the analysis track's late-integration lane fills it in.

#![deny(missing_docs)]

pub mod annotation;
pub mod axis;
pub mod chrome;
pub mod colormap;
pub mod gpu;
pub mod keymap;
pub mod layout;
pub mod overlay;
pub mod scene;
mod surface;
pub mod theme;
pub mod waterfall;
pub mod zoom;

pub use annotation::{AnalysisLoad, Annotation, AnnotationFeed, FreqSpanHz, TrackId};
pub use chrome::{ChromeResponse, HudStats};
pub use colormap::Colormap;
pub use gpu::{EguiFrame, FrameComposer, OffscreenTarget, RtsaInput, OFFSCREEN_FORMAT};
pub use keymap::{Action, Binding, ChromeRequests, BINDINGS, TAU_STEP, WF_SPAN_STEP};
pub use layout::{DisplayParams, Layout, SplitRegions, ViewState, DEFAULT_DECLUTTER_CAP};
pub use overlay::{AnnotationLevel, AnnotationOverlayState, LabelStyle};
pub use surface::{
    DEFAULT_GAMMA, DEFAULT_WF_GAMMA, GAMMA_MAX, GAMMA_MIN, INTENSITY_FLOOR_EPSILON, RING_ROWS,
};
pub use theme::Theme;
pub use waterfall::{
    derive_cadence, RowAggregation, RowAggregator, RowCadence, RowRing, RowTimeBase, WaterfallMode,
    FAST_INTERVAL_DEFAULT_S, ROW_FLOOR_DBFS,
};
pub use zoom::ZoomSpan;
