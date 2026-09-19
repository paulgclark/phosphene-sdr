// SPDX-License-Identifier: MIT

//! The SoapySDR source backend (FR-S5, lane S1-A) — spec §4.5 integration
//! **Pattern C**: the runtime plugin host.
//!
//! ## Licensing (LC-2) — the audit trail `cargo-deny` cannot produce
//!
//! This backend's only link-time dependency is **libSoapySDR** — Boost
//! Software License 1.0, on the LC-2 allow-list — reached through the
//! `soapysdr` crate (BSL-1.0 OR Apache-2.0; its `soapysdr-sys` binding is
//! BSL-1.0). At runtime, *Soapy itself* discovers and loads whatever vendor
//! modules the **user** has installed on their own machine — SoapyUHD,
//! SoapyHackRF, SoapyRTLSDR and friends — through its plugin interface.
//!
//! Several of those vendor stacks are GPL: **UHD is GPL-3 and librtlsdr is
//! GPL-2** (D-003, D-016), and LC-2 forbids linking, vendoring, or bundling
//! either, ever. Nothing GPL enters this repository, this binary, or any
//! release artifact: we link BSL-1.0 SoapySDR, and the GPL driver is a
//! user-installed module that Soapy loads at runtime in the user's own
//! process space — the arm's-length arrangement D-016 records as the basis
//! for reaching GPL-driven hardware from an MIT binary. D-020 measured that
//! no subprocess fallback exists for UHD (`uhd-host` ships no capture
//! utility), so **Pattern C is the only route** to the owner's top-priority
//! hardware — if it fails, that is a stop-and-surface, never a licence
//! compromise. `cargo-deny` gates what we link; this comment records why the
//! reach beyond the link boundary is clean.
//!
//! ## Feature gating
//!
//! The device half ([`SoapySource`], `device.rs`) is behind the **`soapy`
//! cargo feature, off by default**: CI has no SoapySDR installed and the
//! default build must not need it. This module's pure-Rust half —
//! configuration, device-argument parsing, device selection, error text, and
//! the [`SourceMeta`] mapping — is compiled unconditionally: it has no
//! external dependency, it lets a featureless binary explain precisely how to
//! enable the backend, and it keeps the lane's hardware-free seal tests
//! running in the default CI build.
//!
//! ## Behaviour contract (device half)
//!
//! * **Metadata is device readback, never an echo of the request**
//!   (clarification C5): after applying the requested rate / centre / gain,
//!   the backend reads the actual values back from the device and reports
//!   those. A hardware source knows its rate, so both rate and centre are
//!   always `Some`.
//! * **Overflow is a counted EVENT, never an inferred quantity (D-050,
//!   amending D-048).** `SOAPY_SDR_OVERFLOW` means the device lost an
//!   *unknown* amount of data; exactly that is reported — the event is
//!   counted in [`SoapySourceStats`], handed to
//!   [`SampleSink::device_overflow`], and noted on stderr (rate-limited).
//!   No sample count is derived from `time_ns`: the binding surfaces no
//!   read-path validity flag for it, and a fabricated loss count is the
//!   mirror image of the 100%-while-dropping dishonesty D-048 forbids.
//!   Should a binding ever report a trustworthy count, D-048's
//!   received-and-dropped rule applies through
//!   [`SampleSink::device_lost`], which stands gated on that evidence.
//!   Sustained overload of the *display* pipeline still lands where it
//!   always did: this backend never stalls on its sink, so ring
//!   backpressure is shed and counted at the producer (D-013/NFR-P3).
//! * **Disconnect ends the stream as a real error** naming the device — a
//!   vanished USB radio is a failed stream, not a silent gap.
//!
//! [`SoapySource`]: crate::soapy::SoapySource
//! [`SourceMeta`]: crate::source::SourceMeta
//! [`SampleSink::device_lost`]: crate::source::SampleSink::device_lost
//! [`SampleSink::device_overflow`]: crate::source::SampleSink::device_overflow
//! [`SoapySourceStats`]: crate::soapy::SoapySourceStats

#[cfg(feature = "soapy")]
mod device;

#[cfg(feature = "soapy")]
pub use device::{SoapySource, SoapySourceStats};

use crate::source::{SourceError, SourceMeta};

/// Configuration for a SoapySDR source. The CLI layer parses
/// `--source soapy[:device-args]`, `--device-args`, `--rate`, `--center`, and
/// `--gain` into this; the backend never parses command lines.
///
/// This type is compiled whether or not the `soapy` feature is on, so the CLI
/// can always represent the request (and a build without the feature can say
/// exactly how to get one with it).
#[derive(Debug, Clone, PartialEq)]
pub struct SoapySourceConfig {
    /// SoapySDR device arguments as `key=value` pairs separated by commas,
    /// e.g. `driver=uhd` or `driver=uhd,serial=EXAMPLE1`. Empty means "any
    /// device": enumeration runs over every installed Soapy module.
    pub device_args: String,
    /// Sample rate to ask of the device, Hz. `None` leaves the device at its
    /// own default; either way the reported metadata is the device's readback.
    pub sample_rate_hz: Option<f64>,
    /// Centre frequency to tune to, Hz. `None` leaves the device untouched.
    pub center_freq_hz: Option<f64>,
    /// Overall RX gain to set, dB. `None` leaves the device's default gain.
    /// Per-stage gains are FR-S7 territory (M3).
    pub gain_db: Option<f64>,
}

impl SoapySourceConfig {
    /// A config selecting by device args only, everything else left at the
    /// device's own defaults.
    pub fn new(device_args: impl Into<String>) -> SoapySourceConfig {
        SoapySourceConfig {
            device_args: device_args.into(),
            sample_rate_hz: None,
            center_freq_hz: None,
            gain_db: None,
        }
    }

    /// Reject unusable numbers before any device is touched, naming the field
    /// and the offending value (spec §8.4).
    pub fn validate(&self) -> Result<(), SourceError> {
        if let Some(rate) = self.sample_rate_hz {
            if !rate.is_finite() || rate <= 0.0 {
                return Err(SourceError::InvalidConfig(format!(
                    "sample rate must be finite and positive, got {rate} Hz"
                )));
            }
        }
        if let Some(center) = self.center_freq_hz {
            if !center.is_finite() {
                return Err(SourceError::InvalidConfig(format!(
                    "center frequency must be finite, got {center} Hz"
                )));
            }
        }
        if let Some(gain) = self.gain_db {
            if !gain.is_finite() {
                return Err(SourceError::InvalidConfig(format!(
                    "gain must be finite, got {gain} dB"
                )));
            }
        }
        Ok(())
    }
}

/// Whether this build carries the SoapySDR device backend (the `soapy` cargo
/// feature). The CLI uses this to turn `--source soapy` in a featureless
/// build into an §8.4-grade error naming the exact rebuild flag, instead of a
/// mystery failure.
pub const fn compiled() -> bool {
    cfg!(feature = "soapy")
}

/// Open a SoapySDR source, boxed for the app's source thread — the seam
/// `pipeline::open_source` consumes. Compiled whether or not the `soapy`
/// feature is on, so the app needs no feature plumbing of its own: without
/// the feature this returns the §8.4-grade rebuild error instead of a device.
#[cfg(feature = "soapy")]
pub fn open(
    config: &SoapySourceConfig,
) -> Result<Box<dyn crate::source::SampleSource + Send>, SourceError> {
    Ok(Box::new(SoapySource::new(config.clone())?))
}

/// Feature-off half of [`open`]: names the exact rebuild flag (spec §8.4).
#[cfg(not(feature = "soapy"))]
pub fn open(
    config: &SoapySourceConfig,
) -> Result<Box<dyn crate::source::SampleSource + Send>, SourceError> {
    let _ = config;
    Err(SourceError::InvalidConfig(
        "this build has no SoapySDR support: rebuild with `cargo build --features \
         phosphene-sources/soapy` (off by default so plain builds need no SoapySDR \
         install; the backend then reaches radios through the user's installed \
         Soapy modules — spec §4.5 Pattern C)"
            .to_owned(),
    ))
}

/// Parse `driver=uhd,serial=XYZ`-style device arguments into pairs.
///
/// Rules: segments are comma-separated and whitespace-trimmed; empty segments
/// (e.g. a trailing comma) are ignored; each segment must be `key=value` with
/// a non-empty key; a repeated key is an error. Values containing `,` or `=`
/// cannot be expressed in this syntax — no Soapy device identifier needs
/// them.
pub fn parse_device_args(s: &str) -> Result<Vec<(String, String)>, String> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for segment in s.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let Some((key, value)) = segment.split_once('=') else {
            return Err(format!(
                "device argument \"{segment}\" is not key=value \
                 (device args look like driver=uhd,serial=XYZ)"
            ));
        };
        let (key, value) = (key.trim(), value.trim());
        if key.is_empty() {
            return Err(format!("device argument \"{segment}\" has an empty key"));
        }
        if pairs.iter().any(|(existing, _)| existing == key) {
            return Err(format!("device argument key \"{key}\" is given twice"));
        }
        pairs.push((key.to_owned(), value.to_owned()));
    }
    Ok(pairs)
}

/// The canonical `key=value,key=value` form of parsed device args — what the
/// backend hands to Soapy's enumeration and what error messages quote as
/// "what was searched".
pub fn format_device_args(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The outcome of choosing among enumerated devices — see [`select_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Use the device at this index of the enumeration result.
    Chosen {
        /// Index into the enumerated device list.
        index: usize,
    },
    /// Every match was Soapy's `audio` (soundcard) pseudo-device and the user
    /// did not ask for it — treated as "no radio found" rather than silently
    /// displaying a soundcard spectrum.
    OnlyAudio {
        /// How many audio devices were skipped.
        count: usize,
    },
    /// The enumeration returned nothing at all.
    None,
}

/// Choose which enumerated device to open (FR-S8: one discovery path for
/// every driver).
///
/// The first match wins, with one deliberate exception: Soapy's `audio`
/// driver (a soundcard, present on most machines) is skipped unless the user
/// explicitly asked for `driver=audio` — otherwise a bare `--source soapy`
/// on a machine with a dummy soundcard would silently show audio instead of
/// radio. Devices are considered in Soapy's own enumeration order.
pub fn select_device(requested: &[(String, String)], found: &[Vec<(String, String)>]) -> Selection {
    if found.is_empty() {
        return Selection::None;
    }
    let audio_requested = requested.iter().any(|(k, v)| k == "driver" && v == "audio");
    if audio_requested {
        return Selection::Chosen { index: 0 };
    }
    match found
        .iter()
        .position(|device| driver_of(device) != Some("audio"))
    {
        Some(index) => Selection::Chosen { index },
        None => Selection::OnlyAudio { count: found.len() },
    }
}

fn driver_of(pairs: &[(String, String)]) -> Option<&str> {
    pairs
        .iter()
        .find(|(k, _)| k == "driver")
        .map(|(_, v)| v.as_str())
}

/// The zero-devices error text: names exactly what was searched (the lane
/// contract — never a panic, never a silent hang).
pub fn no_devices_message(searched: &str) -> String {
    if searched.is_empty() {
        "no SoapySDR devices found: enumeration over every installed Soapy module \
         returned nothing (is the device plugged in? `SoapySDRUtil --find` shows \
         what Soapy can see)"
            .to_owned()
    } else {
        format!(
            "no SoapySDR devices found matching \"{searched}\": enumeration over \
             the installed Soapy modules returned nothing for those arguments \
             (`SoapySDRUtil --find` shows what Soapy can see)"
        )
    }
}

/// The only-audio-devices error text — the searched args are named just like
/// [`no_devices_message`], plus the way to opt into the soundcard deliberately.
pub fn only_audio_message(searched: &str, count: usize) -> String {
    let matching = if searched.is_empty() {
        String::new()
    } else {
        format!(" matching \"{searched}\"")
    };
    format!(
        "no SoapySDR radio found{matching}: only the Soapy audio (soundcard) \
         pseudo-device answered ({count} found); pass driver=audio explicitly \
         if a soundcard really is the source you want"
    )
}

/// The FR-S8 "clear message about which was chosen" when several devices
/// matched.
pub fn chosen_message(total: usize, chosen: &[(String, String)]) -> String {
    format!(
        "{total} SoapySDR devices matched; using {} — narrow the match with \
         --device-args (e.g. driver=…, serial=…)",
        SoapyDeviceInfo::from_pairs(chosen).describe()
    )
}

/// Identity of an enumerated Soapy device, extracted from its kwargs — the
/// input to the [`SourceMeta`] mapping and to every error message that names
/// the device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SoapyDeviceInfo {
    /// The Soapy driver key (`uhd`, `hackrf`, …), if the kwargs carried one.
    pub driver: Option<String>,
    /// The device's human-readable label, if the kwargs carried one.
    pub label: Option<String>,
    /// The device serial, if the kwargs carried one.
    pub serial: Option<String>,
}

impl SoapyDeviceInfo {
    /// Extract identity from enumerated device kwargs.
    pub fn from_pairs(pairs: &[(String, String)]) -> SoapyDeviceInfo {
        let get = |wanted: &str| {
            pairs
                .iter()
                .find(|(k, _)| k == wanted)
                .map(|(_, v)| v.clone())
                .filter(|v| !v.is_empty())
        };
        SoapyDeviceInfo {
            driver: get("driver"),
            label: get("label"),
            serial: get("serial"),
        }
    }

    /// How the device is named in errors and diagnostics, most-specific form
    /// first: `driver=uhd serial=EXAMPLE1`, else the label, else a generic
    /// name.
    pub fn describe(&self) -> String {
        match (&self.driver, &self.serial) {
            (Some(driver), Some(serial)) => format!("driver={driver} serial={serial}"),
            (Some(driver), None) => format!("driver={driver}"),
            (None, _) => self
                .label
                .clone()
                .unwrap_or_else(|| "SoapySDR device".to_owned()),
        }
    }
}

/// The [`SourceMeta`] mapping for a Soapy device.
///
/// `sample_rate_hz` and `center_freq_hz` here are **device readback**, not the
/// requested values (clarification C5: report what the hardware says, never
/// invent — and a hardware source always knows, so both are `Some`).
pub fn source_meta(info: &SoapyDeviceInfo, sample_rate_hz: f64, center_freq_hz: f64) -> SourceMeta {
    let label = info
        .label
        .clone()
        .unwrap_or_else(|| match (&info.driver, &info.serial) {
            (Some(driver), Some(serial)) => format!("{driver} {serial}"),
            (Some(driver), None) => driver.clone(),
            (None, _) => "soapy".to_owned(),
        });
    SourceMeta {
        sample_rate_hz: Some(sample_rate_hz),
        center_freq_hz: Some(center_freq_hz),
        label,
        provenance: format!("soapysdr {} (cf32)", info.describe()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn device_args_parse_and_canonicalise() {
        assert_eq!(parse_device_args("").unwrap(), vec![]);
        assert_eq!(parse_device_args("  ").unwrap(), vec![]);
        let parsed = parse_device_args(" driver=uhd , serial=EXAMPLE1 ,").unwrap();
        assert_eq!(parsed, pairs(&[("driver", "uhd"), ("serial", "EXAMPLE1")]));
        assert_eq!(format_device_args(&parsed), "driver=uhd,serial=EXAMPLE1");
    }

    #[test]
    fn device_args_errors_name_the_offending_segment() {
        let err = parse_device_args("driver=uhd,serialXYZ").unwrap_err();
        assert!(
            err.contains("serialXYZ") && err.contains("key=value"),
            "unhelpful: {err}"
        );
        let err = parse_device_args("=uhd").unwrap_err();
        assert!(err.contains("empty key"), "unhelpful: {err}");
        let err = parse_device_args("driver=uhd,driver=hackrf").unwrap_err();
        assert!(
            err.contains("driver") && err.contains("twice"),
            "unhelpful: {err}"
        );
    }

    #[test]
    fn config_validation_names_the_field() {
        let mut config = SoapySourceConfig::new("driver=uhd");
        config.sample_rate_hz = Some(-1.0);
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("sample rate") && err.contains("-1"),
            "unhelpful: {err}"
        );

        let mut config = SoapySourceConfig::new("");
        config.center_freq_hz = Some(f64::NAN);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("center frequency"), "unhelpful: {err}");

        let mut config = SoapySourceConfig::new("");
        config.gain_db = Some(f64::INFINITY);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("gain"), "unhelpful: {err}");

        let mut config = SoapySourceConfig::new("");
        config.sample_rate_hz = Some(2.048e6);
        config.center_freq_hz = Some(100e6);
        config.gain_db = Some(40.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn selection_prefers_a_radio_over_the_soundcard() {
        let audio = pairs(&[("driver", "audio"), ("label", "Dummy Output")]);
        let uhd = pairs(&[("driver", "uhd"), ("serial", "EXAMPLE1")]);
        // Soapy's enumeration order puts the soundcard first on real machines.
        let found = vec![audio.clone(), uhd.clone()];
        assert_eq!(select_device(&[], &found), Selection::Chosen { index: 1 });
        // Explicitly asking for audio gets audio.
        let req = pairs(&[("driver", "audio")]);
        assert_eq!(select_device(&req, &found), Selection::Chosen { index: 0 });
        // Only the soundcard present: not silently used.
        assert_eq!(
            select_device(&[], &[audio]),
            Selection::OnlyAudio { count: 1 }
        );
        assert_eq!(select_device(&[], &[]), Selection::None);
    }

    #[test]
    fn zero_devices_error_names_what_was_searched() {
        let msg = no_devices_message("driver=uhd,serial=999");
        assert!(msg.contains("driver=uhd,serial=999"), "unhelpful: {msg}");
        assert!(msg.contains("SoapySDRUtil --find"), "unactionable: {msg}");
        let msg = no_devices_message("");
        assert!(
            msg.contains("every installed Soapy module"),
            "unhelpful: {msg}"
        );
        let msg = only_audio_message("", 1);
        assert!(
            msg.contains("audio") && msg.contains("driver=audio"),
            "unhelpful: {msg}"
        );
    }

    #[test]
    fn multi_match_message_names_the_chosen_device() {
        let msg = chosen_message(2, &pairs(&[("driver", "uhd"), ("serial", "EXAMPLE1")]));
        assert!(msg.contains('2') && msg.contains("driver=uhd serial=EXAMPLE1"));
        assert!(msg.contains("--device-args"), "unactionable: {msg}");
    }

    #[test]
    fn source_meta_reports_device_readback_never_invented() {
        // The rate/centre flowing into the meta are the readback values —
        // this test pins that they arrive untouched and always Some (C5:
        // hardware knows its rate).
        let info = SoapyDeviceInfo::from_pairs(&pairs(&[
            ("driver", "uhd"),
            ("serial", "EXAMPLE1"),
            ("label", "B200mini EXAMPLE1"),
        ]));
        let meta = source_meta(&info, 2_048_000.0, 100_000_000.0);
        assert_eq!(meta.sample_rate_hz, Some(2_048_000.0));
        assert_eq!(meta.center_freq_hz, Some(100_000_000.0));
        assert_eq!(meta.label, "B200mini EXAMPLE1");
        assert!(meta.provenance.contains("soapysdr"));
        assert!(meta.provenance.contains("driver=uhd serial=EXAMPLE1"));
        assert!(meta.provenance.contains("cf32"));
    }

    #[test]
    fn source_meta_label_falls_back_through_driver_and_serial() {
        let info = SoapyDeviceInfo::from_pairs(&pairs(&[("driver", "hackrf")]));
        let meta = source_meta(&info, 8e6, 433.92e6);
        assert_eq!(meta.label, "hackrf");
        let info = SoapyDeviceInfo::from_pairs(&[]);
        assert_eq!(info.describe(), "SoapySDR device");
    }
}
