// SPDX-License-Identifier: MIT

//! The owner's capture-filename convention (D-055) — and only that one.
//!
//! ```text
//! cap_<freq><unit>_<rate><unit>sps[_<anything>].<ext>
//! ```
//!
//! Real names from the corpus this was confirmed against:
//!
//! ```text
//! cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16
//! cap_632MHz_30p72Msps.sc16
//! cap_1p985GHz_30p72Msps_arfcn397000_gscn4961_20260609-143210-PDT.sc16
//! ```
//!
//! The rules, each of which exists because breaking it produces a plausible
//! wrong number rather than an obvious failure:
//!
//! * **`p` is the decimal separator.** A filename cannot carry a bare `.`,
//!   so the convention substitutes `p`: `1p985GHz` is 1.985 GHz and
//!   `30p72Msps` is 30.72 Msps.
//! * **Units are read, never assumed.** Both `GHz` and `MHz` occur in the
//!   corpus. Reading `632MHz` as GHz is a factor-of-1000 error that looks
//!   entirely reasonable on screen, so the multiplier comes from the name
//!   or the name does not match at all.
//! * **Everything after the rate field is ignored** — `arfcn…`, `gscn…`,
//!   timestamps. They are not parsed, not required, and their absence is not
//!   a failure: `cap_632MHz_30p72Msps.sc16` carries none.
//! * **Anchored at `cap_`.** A name that merely contains digits is not this
//!   convention.
//! * **All or nothing.** Both values come out or neither does. Recognising
//!   half the metadata would mean guessing the other half, which is what
//!   clarification C5 forbids — which is why D-055 names `b13_751` (a centre
//!   with no rate) and `gnb_n3` (neither) as shapes explicitly *not* inferred
//!   from.
//! * **A non-match is silent.** This is a hint, and a hint that does not fire
//!   is not an error: the caller falls through to the next rung of the
//!   precedence chain.

use std::path::Path;

/// The two values the convention encodes, in SI base units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureName {
    /// Centre frequency in Hz, from the `<freq><unit>Hz` field.
    pub center_freq_hz: f64,
    /// Sample rate in samples/s, from the `<rate><unit>sps` field.
    pub sample_rate_hz: f64,
}

/// The anchor every name of this convention starts with.
const ANCHOR: &str = "cap_";

/// The SI multiplier a prefix letter denotes, shared with any other place
/// the product reads a frequency with units (D-058's entry box uses the same
/// family, so the product has one way of writing a frequency rather than
/// two).
///
/// Deliberately **case-sensitive except for kilo**: `M` is mega and `G` is
/// giga, while lower-case `m` is *milli* in SI and is therefore rejected
/// rather than charitably read as mega — that charity is exactly the
/// factor-of-a-billion mistake this parser exists to avoid. `k` and `K` both
/// mean kilo because no SI prefix claims the capital.
pub fn si_multiplier(prefix: char) -> Option<f64> {
    match prefix {
        'k' | 'K' => Some(1e3),
        'M' => Some(1e6),
        'G' => Some(1e9),
        _ => None,
    }
}

/// Parse a path's file name against the convention, or `None`.
///
/// The extension is stripped first ([`Path::file_stem`]), so `.sc16`,
/// `.cs16` and an extension-less name all read the same. A name that is not
/// valid UTF-8 never matches.
pub fn parse_path(path: &Path) -> Option<CaptureName> {
    parse_stem(path.file_stem()?.to_str()?)
}

/// Parse an already-extension-stripped basename against the convention.
///
/// Exposed separately so the table tests can state the exact string under
/// test rather than a path that has to be de-extensioned first.
pub fn parse_stem(stem: &str) -> Option<CaptureName> {
    let rest = stem.strip_prefix(ANCHOR)?;
    let mut fields = rest.split('_');
    let freq_field = fields.next()?;
    let rate_field = fields.next()?;
    // Everything after the rate field is ignored, present or not.

    let center_freq_hz = parse_scaled(freq_field.strip_suffix("Hz")?)?;
    let sample_rate_hz = parse_scaled(rate_field.strip_suffix("sps")?)?;

    // A zero or non-finite rate is not a rate; a negative centre cannot be
    // written in this grammar, but the finiteness check is what keeps a
    // pathological name out of the display's arithmetic.
    if !(sample_rate_hz.is_finite() && sample_rate_hz > 0.0) || !center_freq_hz.is_finite() {
        return None;
    }
    Some(CaptureName {
        center_freq_hz,
        sample_rate_hz,
    })
}

/// `<digits>[p<digits>][<k|M|G>]` → a value in base units, or `None`.
///
/// Written out rather than delegated to `f64::from_str` on a patched string
/// so that the accepted grammar is exactly what D-055 describes: no signs,
/// no exponents, no whitespace, no bare `.`, and at most one `p`.
fn parse_scaled(field: &str) -> Option<f64> {
    let (digits, scale) = match field.chars().next_back() {
        Some(last) if !last.is_ascii_digit() => (
            &field[..field.len() - last.len_utf8()],
            si_multiplier(last)?,
        ),
        _ => (field, 1.0),
    };
    let mut parts = digits.split('p');
    let whole = parts.next()?;
    let frac = parts.next();
    if parts.next().is_some() {
        // A second `p` is not this grammar.
        return None;
    }
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if let Some(frac) = frac {
        if frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    let joined = match frac {
        Some(frac) => format!("{whole}.{frac}"),
        None => whole.to_owned(),
    };
    Some(joined.parse::<f64>().ok()? * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The real corpus shapes D-055 was confirmed against, including the
    /// unit variation that makes assuming a scale a factor-of-1000 bug.
    #[test]
    fn the_corpus_shapes_parse_with_the_units_they_state() {
        let cases: [(&str, f64, f64); 6] = [
            (
                "cap_1p985GHz_30p72Msps_arfcn397000_gscn4961.sc16",
                1.985e9,
                30.72e6,
            ),
            // MHz, and no trailing fields at all.
            ("cap_632MHz_30p72Msps.sc16", 632e6, 30.72e6),
            // A timestamp tail, which is ignored like every other tail.
            (
                "cap_1p985GHz_30p72Msps_arfcn397000_gscn4961_20260609-143210-PDT.sc16",
                1.985e9,
                30.72e6,
            ),
            (
                "cap_3p70992GHz_30p72Msps_gscn7992_20260706-173148-PDT.sc16",
                3.70992e9,
                30.72e6,
            ),
            ("cap_634p54MHz_7p68Msps.sc16", 634.54e6, 7.68e6),
            // A different extension changes nothing; so does having none.
            ("cap_634p54MHz_15p36Msps.fc32", 634.54e6, 15.36e6),
        ];
        for (name, center, rate) in cases {
            let got = parse_path(&PathBuf::from(name))
                .unwrap_or_else(|| panic!("{name} must match the convention"));
            assert_eq!(got.center_freq_hz, center, "{name} centre");
            assert_eq!(got.sample_rate_hz, rate, "{name} rate");
        }
    }

    /// `632MHz` read as GHz is the failure this parser exists to prevent:
    /// the unit is read out of the name, every time.
    #[test]
    fn every_unit_in_the_k_m_g_family_is_read_not_assumed() {
        let cases = [
            ("cap_632Hz_1000sps", 632.0, 1000.0),
            ("cap_632kHz_48ksps", 632e3, 48e3),
            ("cap_632MHz_1Msps", 632e6, 1e6),
            ("cap_632GHz_1Gsps", 632e9, 1e9),
            ("cap_632KHz_48Ksps", 632e3, 48e3),
        ];
        for (stem, center, rate) in cases {
            let got = parse_stem(stem).unwrap_or_else(|| panic!("{stem} must match"));
            assert_eq!(
                (got.center_freq_hz, got.sample_rate_hz),
                (center, rate),
                "{stem}"
            );
        }
        // Lower-case `m` is milli in SI. Charitably reading it as mega is a
        // factor of a billion; a non-match is the honest answer.
        assert_eq!(parse_stem("cap_632mHz_30p72msps"), None);
    }

    /// D-055 names these explicitly: recognising half the metadata would
    /// mean guessing the other half (C5). Plus the malformed shapes that
    /// must fall through in silence.
    #[test]
    fn the_shapes_that_must_not_match_do_not_match() {
        let negatives = [
            // D-055's two named negatives: a centre with no rate, and neither.
            "b13_751",
            "gnb_n3",
            // Not anchored at `cap_` — merely containing digits is not the
            // convention.
            "recording_1p985GHz_30p72Msps",
            "gscn7992_3mod_3p70992GHz_30p72Msps",
            // Anchored, but the fields are a different convention entirely
            // (no `Hz`, no `sps`).
            "cap_c634p54M_s7p68M",
            // The SDRangel shape D-055 explicitly did not adopt.
            "cap_1876954_7680KSPS",
            // Half the metadata.
            "cap_1p985GHz",
            "cap_1p985GHz_",
            // Malformed numbers.
            "cap_1p9p85GHz_30p72Msps",
            "cap_pGHz_30p72Msps",
            "cap_1p985GHz_p72Msps",
            "cap_1p985GHz_30p72Zsps",
            "cap_-1p985GHz_30p72Msps",
            "cap_1.985GHz_30.72Msps",
            "cap_1e9Hz_30p72Msps",
            "cap_1p985GHz_0sps",
            "cap_ GHz_30p72Msps",
            "cap_",
            "cap",
            "",
        ];
        for stem in negatives {
            assert_eq!(parse_stem(stem), None, "{stem:?} must not match");
        }
    }

    /// A hint that does not fire is not an error: the parser has no error
    /// channel at all, by construction, and a path with no name or a
    /// non-UTF-8 name simply does not match.
    #[test]
    fn a_non_match_is_silent_and_returns_no_error() {
        assert_eq!(parse_path(&PathBuf::from("/")), None);
        assert_eq!(parse_path(&PathBuf::from("..")), None);
        assert_eq!(parse_path(&PathBuf::from("/tmp/capture.bin")), None);
        // A directory prefix is irrelevant — only the file name is read.
        let deep = PathBuf::from("/x/cap_ignored/cap_632MHz_30p72Msps.sc16");
        assert!(parse_path(&deep).is_some());
    }

    #[test]
    fn the_si_family_is_the_one_d058_shares() {
        assert_eq!(si_multiplier('k'), Some(1e3));
        assert_eq!(si_multiplier('K'), Some(1e3));
        assert_eq!(si_multiplier('M'), Some(1e6));
        assert_eq!(si_multiplier('G'), Some(1e9));
        for c in ['m', 'g', 'T', 'u', 'p', 'x', '1'] {
            assert_eq!(si_multiplier(c), None, "{c}");
        }
    }
}
