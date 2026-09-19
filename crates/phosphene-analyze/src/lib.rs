// SPDX-License-Identifier: MIT

//! # phosphene-analyze — signal analysis & annotation (nextgen spec 002, D-017)
//!
//! ## Architecture
//!
//! This crate is the Signal Inspector: it finds every signal in the band the
//! v1 display is showing, measures it, and (in later phases) classifies it —
//! publishing the result as always-on in-place annotations that
//! `phosphene-render` draws (nextgen spec §1, §5). It consumes v1 through two
//! narrow read-only taps ([`tap::SpectrumTap`] / [`tap::AnalysisTap`], spec
//! §2.2) and produces one output, the [`annotate::AnnotationFeed`]; it shares
//! **no files** with any v1 lane, which is what makes the two tracks
//! parallel-safe (spec §2.3, D-017).
//!
//! Everything here is gated behind the **`analyze`** cargo feature from the
//! first commit (D-017): with the flag off this crate is an empty library
//! with no dependencies, and v1 artifacts are unaffected.
//!
//! Modules (spec §8.1) and the types they exchange — the contract the
//! parallel A0 feature lanes build against:
//!
//! * [`tap`] — the [`tap::SpectrumTap`] / [`tap::AnalysisTap`] traits and
//!   their supporting types, re-exported from their canonical home in
//!   `phosphene_core::tap` (the D-017 swap has landed).
//! * [`detect`] — noise floor, CFAR, blob extraction (spec §7.1–§7.3).
//!   Skeleton: owns [`detect::Detection`], the per-blob output.
//! * [`track`] — track lifecycle and hopper clustering (spec §7.4).
//!   Skeleton: owns [`track::Track`], the central data model (spec §8.3).
//! * [`measure`] — the FR-AM measurements (spec §7.5). Skeleton: owns
//!   [`measure::MeasurementSet`] and the [`measure::Measured`] value+confidence
//!   wrapper (NFR-A3: every asserted value carries a confidence).
//! * [`classify`] — [`classify::classify`]: offline modulation
//!   classification from one track's raw-IQ snapshot (spec §7.6–§7.9, phase
//!   A1 scope — OOK/ASK, FSK, or Unknown). No live wiring (CL-3).
//! * [`annotate`] — [`annotate::Annotation`] and the feed shape v1's render
//!   consumes (`push_annotations`, D-017).
//! * [`harness`] — the file/siggen-backed offline test harness (spec §2.3):
//!   feeds recorded or generated IQ through `phosphene-core`'s analyzer and
//!   serves it back through both tap traits, so every lane in this crate
//!   develops against known ground truth without live v1.
//!
//! ## Clean room (LC-1)
//!
//! All algorithms implemented in this crate derive from nextgen spec §7 and
//! the public literature it cites — never from GPL codebases (fosphor, URH,
//! inspectrum, gr-inspector, SigDigger, gr-cyclo are all named and off
//! limits).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "analyze")]
pub mod annotate;
#[cfg(feature = "analyze")]
pub mod classify;
#[cfg(feature = "analyze")]
pub mod detect;
#[cfg(feature = "analyze")]
pub mod harness;
#[cfg(feature = "analyze")]
pub mod measure;
#[cfg(feature = "analyze")]
pub mod tap;
#[cfg(feature = "analyze")]
pub mod track;

#[cfg(feature = "analyze")]
pub use annotate::{build_annotation, Annotation, AnnotationFeed, FreqSpanHz};
#[cfg(feature = "analyze")]
pub use classify::classify;
#[cfg(feature = "analyze")]
pub use detect::Detection;
#[cfg(feature = "analyze")]
pub use measure::{
    BehaviorFlags, CenterFrequency, CenterMethod, HopDetail, Measured, MeasurementSet,
    ObwConvention, OccupiedBandwidth, Temporal,
};
#[cfg(feature = "analyze")]
pub use tap::{
    AnalysisTap, BandMeta, CaptureMeta, FrameView, IqBuf, SpectrumTap, TapCaps, TapError, TimeSpan,
};
#[cfg(feature = "analyze")]
pub use track::{Classification, ModulationClass, SignatureCandidate, Track, TrackId, TrackState};
