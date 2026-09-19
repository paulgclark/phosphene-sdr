// SPDX-License-Identifier: MIT

//! # phosphene-sources — the `SampleSource` seam + source backends
//!
//! ## Architecture
//!
//! This crate owns the boundary between *where samples come from* and
//! *everything else*: the [`SampleSource`] trait (product spec §8.2) that every
//! backend implements, and the backends themselves, each behind a cargo
//! feature (spec §8.1). It does **no DSP math and no rendering** — sources
//! deliver calibrated-format `cf32` complex baseband plus metadata
//! ([`SourceMeta`]), and the compute pipeline in `phosphene-core` takes it
//! from there.
//!
//! Modules:
//!
//! * [`source`] — the [`SampleSource`] / [`SampleSink`] traits and their
//!   supporting types ([`SourceDesc`], [`SourceMeta`], [`ControlCaps`],
//!   [`Control`], [`SourceError`]).
//! * [`siggen`] (feature `siggen`, default-on) — the deterministic signal
//!   generator: tones, a calibrated noise floor, chirps, and a bursty
//!   frequency hopper. It is both the demo mode (`--sdr siggen`) and the
//!   known-truth test fixture every later lane tests against.
//! * [`format`] (always on) — the five raw IQ formats of FR-S1 (`cf32`,
//!   `cs32`, `cs16`, `cs8`, `cu8`) and their conversion to/from
//!   full-scale-normalised `cf32` under the one D-006 calibration; shared by
//!   every raw-IQ backend (file now, rtl_tcp later).
//! * [`metadata`] (always on) — what the capture says about itself (D-055):
//!   the SigMF `.sigmf-meta` sidecar, the owner's `cap_<freq>_<rate>sps`
//!   filename convention, and the `flag > sidecar > filename > nothing`
//!   precedence chain between them. Every resolved value carries its own
//!   [`Provenance`], because a rate a sidecar *declared* and a rate a
//!   filename *hinted* are not the same claim.
//! * [`file`] (feature `file`, default-on) — the file / FIFO / stdin replay
//!   source (FR-S1, lane M1-E): throttled real-time replay by default,
//!   `--loop`, fast-forward, and clarification C5's unthrottled
//!   normalised-axis behaviour when no rate is given.
//! * [`soapy`] (feature `soapy`, **default-off**) — the SoapySDR hardware
//!   backend (FR-S5, lane S1-A), spec §4.5 **Pattern C**: links BSL-1.0
//!   libSoapySDR only, which loads the user's installed vendor modules
//!   (SoapyUHD & co., GPL included) at runtime — nothing GPL is ever linked
//!   or shipped (LC-2; see the module docs for the audit trail). The
//!   module's pure-Rust half (config, device-args parsing, selection, error
//!   text) compiles unconditionally so every build can represent and explain
//!   `--source soapy`.
//!
//! ## Licensing pattern discipline (LC-2, spec §4.5)
//!
//! Every hardware backend added to this crate must carry its integration
//! pattern tag — A (direct link, permissive libs only), B (subprocess
//! capture), or C (SoapySDR runtime plugin host). Per D-015 the first real
//! backends will be **out-of-process** (UHD via Soapy module or subprocess —
//! GPL-3, never linked), which is why the [`SampleSource`] trait is designed
//! around blocking delivery into a sink rather than any in-process
//! library-callback shape. The signal generator is pure Rust with no external
//! code at all (trivially Pattern A).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "file")]
pub mod file;
pub mod format;
pub mod metadata;
pub mod soapy;
pub mod source;

#[cfg(feature = "siggen")]
pub mod siggen;

pub use format::IqFormat;
pub use metadata::{CaptureMetadata, Provenance, Sourced};
pub use phosphene_core::Complex;
pub use source::{
    Control, ControlCaps, SampleSink, SampleSource, SinkFlow, SourceDesc, SourceError, SourceMeta,
};

#[cfg(feature = "file")]
pub use file::{FileSource, FileSourceConfig, FileSourceStats, Input, ReplayPace};
#[cfg(feature = "siggen")]
pub use siggen::{
    ChirpConfig, HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig,
};
pub use soapy::SoapySourceConfig;
#[cfg(feature = "soapy")]
pub use soapy::{SoapySource, SoapySourceStats};
