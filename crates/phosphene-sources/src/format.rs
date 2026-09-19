// SPDX-License-Identifier: MIT

//! Raw IQ sample formats (FR-S1) and their conversion to/from `cf32`.
//!
//! ## The one calibration (lane M1-E's reason to exist)
//!
//! Every format normalises to the same **D-006 full-scale convention** at
//! ingest (spec §8.2): a full-scale tone in *any* of the five formats decodes
//! to a complex sinusoid of magnitude 1.0 and therefore reads **0.0 dBFS**
//! through `phosphene-core`. Concretely, per component:
//!
//! | Format | Native sample     | Full scale → ±1.0        | Decode map |
//! |--------|-------------------|--------------------------|------------|
//! | `cf32` | `f32` LE pair     | 1.0 (already normalised) | identity |
//! | `cs32` | `i32` LE pair     | ±2 147 483 647           | `x / 2147483647` |
//! | `cs16` | `i16` LE pair     | ±32767                   | `x / 32767` |
//! | `cs8`  | `i8` pair         | ±127                     | `x / 127` |
//! | `cu8`  | `u8` pair         | 0 / 255 (mid 127.5)      | `(x − 127.5) / 127.5` |
//!
//! Signed formats normalise by the maximum *positive* code so that full scale
//! is exact in both directions and encode→decode round-trips are exact at the
//! rails; the single most-negative code (`−32768`, `−128`) decodes slightly
//! below −1.0, which is honest about what the converter produced. `cu8` uses
//! the RTL-SDR convention of a 127.5 midpoint, so 0/255 decode to exactly
//! ∓1.0 and the DC offset of the unsigned encoding cancels.
//!
//! `cs32` is the same rule one width up (D-062), and it is the only format
//! whose native precision exceeds the pipeline's: an `i32` carries 31
//! magnitude bits and the `f32` it decodes into carries 24, so the bottom
//! bits are rounded away at ingest. That is a deliberate consequence of spec
//! §7.7 (`f32` DSP), not a defect — 24 bits is ≈145 dB of dynamic range,
//! far past what any converter that writes `ci32` actually delivers — and it
//! is why the `cs32` scale divides by `i32::MAX` even though `i32::MAX` is
//! not itself representable in `f32`: both the code and the divisor round to
//! 2³¹, so full scale still decodes to exactly ±1.0.
//!
//! Multi-byte formats (`cf32`, `cs32`, `cs16`) are **little-endian**, the GNU
//! Radio on-disk convention. Samples are interleaved I then Q.
//!
//! ## `cs32` assumes full scale at the container maximum (D-071)
//!
//! The table above says `x / 2147483647` for `cs32`, and that choice needs
//! stating out loud because — unlike `cs16` or `cu8` — **the format does not
//! carry it**. SigMF's `ci32_le` names the *container*, not where full scale
//! sits inside it: two files with identical bytes and identical metadata can
//! mean different levels, and nothing in either file says which.
//!
//! So `cs32` normalises by the **container's** full scale, `i32::MAX`. It is
//! the only value derivable from `ci32_le` alone, and it keeps D-006 exactly
//! true for a genuine 32-bit full-scale capture.
//!
//! **The known consequence: SDRangel writes 24-bit values in a 32-bit
//! container**, so its `ci32_le` recordings display **≈48 dB low**
//! (`20·log₁₀(2⁸)` = 48.16). That is the correct reading of an ambiguous
//! input, not a defect to patch around. In particular this crate does **not**
//! sniff `core:recorder`, and does **not** infer a scale from the observed
//! peak level: a heuristic that lifts a genuinely quiet capture by 48 dB is
//! far worse than a level that is honestly low, and is undetectable when it
//! is wrong. An explicit user-supplied override (`--full-scale-bits`) is the
//! right ergonomic fix and is on the backlog; guessing is not.
//!
//! ## Aliases (D-055, D-062)
//!
//! `sc16` — the SDR world's *signed complex 16* — names the identical layout
//! this crate calls `cs16`; `ci32` and `sc32` likewise name `cs32`, which is
//! how SigMF (`ci32_le`) and the SDR world respectively spell it. They are
//! accepted wherever a format is *read* ([`IqFormat::from_name`],
//! [`IqFormat::from_extension`], `FromStr`) and never *written*:
//! [`IqFormat::name`] has exactly one answer per layout, so nothing
//! downstream has two names to keep in step.
//!
//! Decoding never panics and never invents data: [`decode_append`] converts
//! only whole samples and reports how many bytes it consumed, so a truncated
//! stream simply leaves a partial-sample remainder for the caller to carry or
//! count (lane M1-E: a partial sample at EOF is a normal condition).

use std::fmt;
use std::str::FromStr;

use phosphene_core::Complex;

/// Per-component scale of `cs32` (D-062, D-071): full scale is the maximum
/// positive code of the **container**, `i32::MAX` = 2 147 483 647.
///
/// `ci32_le` names the container and not the convention inside it, so this is
/// an assumption rather than a fact read off the file — the only one
/// derivable from the datatype alone, and the one that keeps D-006 true for a
/// genuine 32-bit full-scale capture. A recorder that writes narrower values
/// into the same container (SDRangel writes 24-bit ones) therefore reads
/// ≈48 dB low, and is left reading low: see the module docs for why guessing
/// is worse.
///
/// Written as `i32::MAX as f32` rather than as a literal because the value is
/// *not* representable in `f32` — it rounds to 2³¹ — and spelling it out
/// would hide that. Nothing downstream is harmed by the rounding: the
/// full-scale code rounds to the same 2³¹ when it is converted for the
/// division, so `±2 147 483 647` still decodes to exactly `±1.0`.
const CS32_SCALE: f32 = i32::MAX as f32;
/// Per-component scale of `cs16`: full scale is the maximum positive code.
const CS16_SCALE: f32 = 32767.0;
/// Per-component scale of `cs8`.
const CS8_SCALE: f32 = 127.0;
/// Midpoint and scale of `cu8` (RTL-SDR convention): `(x − 127.5) / 127.5`.
const CU8_MID: f32 = 127.5;

/// A raw IQ sample format on the wire / on disk (FR-S1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IqFormat {
    /// Interleaved little-endian `f32` I/Q — GNU Radio `.cfile` native.
    Cf32,
    /// Interleaved little-endian `i32` I/Q, 8 bytes per sample (D-062).
    /// SigMF spells this layout `ci32_le`. Full scale is assumed to be the
    /// container maximum (D-071) — see the module docs.
    Cs32,
    /// Interleaved little-endian `i16` I/Q.
    Cs16,
    /// Interleaved `i8` I/Q — HackRF native.
    Cs8,
    /// Interleaved `u8` I/Q with a 127.5 midpoint — RTL-SDR native.
    Cu8,
}

impl IqFormat {
    /// All formats, in FR-S1's order.
    pub const ALL: [IqFormat; 5] = [
        IqFormat::Cf32,
        IqFormat::Cs32,
        IqFormat::Cs16,
        IqFormat::Cs8,
        IqFormat::Cu8,
    ];

    /// Alternative spellings of a format that already exists (D-055, D-062).
    ///
    /// `sc16` is the SDR world's name — *signed complex 16* — for the
    /// identical byte layout this crate calls `cs16`: interleaved
    /// little-endian `i16` I/Q. `ci32` (SigMF's spelling, minus the
    /// endianness suffix) and `sc32` name `cs32` the same way. They are
    /// **aliases, not variants**: two names for one layout would be a second
    /// source of truth for nothing, so [`IqFormat::name`] keeps returning the
    /// one canonical name and only the *readers*
    /// ([`IqFormat::from_name`], [`IqFormat::from_extension`]) accept the
    /// alias.
    pub const ALIASES: &'static [(&'static str, IqFormat)] = &[
        ("sc16", IqFormat::Cs16),
        ("ci32", IqFormat::Cs32),
        ("sc32", IqFormat::Cs32),
    ];

    /// Parse a format name, canonical or aliased, case-sensitively.
    ///
    /// Returns `None` for anything that is not one of the five FR-S1 formats
    /// or an entry of [`IqFormat::ALIASES`].
    pub fn from_name(name: &str) -> Option<IqFormat> {
        IqFormat::ALL
            .into_iter()
            .find(|f| f.name() == name)
            .or_else(|| {
                IqFormat::ALIASES
                    .iter()
                    .find(|(alias, _)| *alias == name)
                    .map(|&(_, f)| f)
            })
    }

    /// The format a recognised file extension implies, or `None`.
    ///
    /// `ext` is matched case-insensitively and without its leading dot.
    /// Beyond the format names themselves this knows GNU Radio's `.cfile`
    /// (native `cf32`) and, per D-055 and D-062, `.sc16`, `.ci32` and
    /// `.sc32`.
    ///
    /// An unrecognised extension is `None`, never a guess: raw IQ has no
    /// self-describing header, and inventing a layout shows garbage that
    /// looks like signal.
    pub fn from_extension(ext: &str) -> Option<IqFormat> {
        let ext = ext.to_ascii_lowercase();
        match ext.as_str() {
            "cfile" => Some(IqFormat::Cf32),
            other => IqFormat::from_name(other),
        }
    }

    /// Bytes per complex sample (I and Q together).
    pub fn bytes_per_sample(self) -> usize {
        match self {
            IqFormat::Cf32 => 8,
            IqFormat::Cs32 => 8,
            IqFormat::Cs16 => 4,
            IqFormat::Cs8 => 2,
            IqFormat::Cu8 => 2,
        }
    }

    /// The format's conventional name (`cf32`, `cs32`, `cs16`, `cs8`, `cu8`).
    pub fn name(self) -> &'static str {
        match self {
            IqFormat::Cf32 => "cf32",
            IqFormat::Cs32 => "cs32",
            IqFormat::Cs16 => "cs16",
            IqFormat::Cs8 => "cs8",
            IqFormat::Cu8 => "cu8",
        }
    }
}

impl fmt::Display for IqFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A format name that is not one of the five FR-S1 formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFormat(pub String);

impl fmt::Display for UnknownFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown IQ format {:?}; expected one of cf32, cs32, cs16, cs8, cu8 \
             (sc16 is an alias of cs16; ci32 and sc32 are aliases of cs32)",
            self.0
        )
    }
}

impl std::error::Error for UnknownFormat {}

impl FromStr for IqFormat {
    type Err = UnknownFormat;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        IqFormat::from_name(s).ok_or_else(|| UnknownFormat(s.to_string()))
    }
}

/// Decode every *whole* sample in `bytes` to full-scale-normalised `cf32`,
/// appending to `out`, and return the number of bytes consumed (always a
/// multiple of [`IqFormat::bytes_per_sample`]).
///
/// Any remainder shorter than one sample is left unconsumed — a truncated
/// stream is the caller's to carry across reads or to count at EOF, never a
/// panic. `cf32` bytes are passed through as-is (garbage in, garbage out —
/// this crate does not sanitise the user's capture).
pub fn decode_append(format: IqFormat, bytes: &[u8], out: &mut Vec<Complex<f32>>) -> usize {
    let bps = format.bytes_per_sample();
    let consumed = bytes.len() - bytes.len() % bps;
    let whole = &bytes[..consumed];
    out.reserve(consumed / bps);
    match format {
        IqFormat::Cf32 => {
            for s in whole.as_chunks::<8>().0 {
                let re = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                let im = f32::from_le_bytes([s[4], s[5], s[6], s[7]]);
                out.push(Complex::new(re, im));
            }
        }
        IqFormat::Cs32 => {
            for s in whole.as_chunks::<8>().0 {
                let re = i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / CS32_SCALE;
                let im = i32::from_le_bytes([s[4], s[5], s[6], s[7]]) as f32 / CS32_SCALE;
                out.push(Complex::new(re, im));
            }
        }
        IqFormat::Cs16 => {
            for s in whole.as_chunks::<4>().0 {
                let re = i16::from_le_bytes([s[0], s[1]]) as f32 / CS16_SCALE;
                let im = i16::from_le_bytes([s[2], s[3]]) as f32 / CS16_SCALE;
                out.push(Complex::new(re, im));
            }
        }
        IqFormat::Cs8 => {
            for s in whole.as_chunks::<2>().0 {
                let re = s[0] as i8 as f32 / CS8_SCALE;
                let im = s[1] as i8 as f32 / CS8_SCALE;
                out.push(Complex::new(re, im));
            }
        }
        IqFormat::Cu8 => {
            for s in whole.as_chunks::<2>().0 {
                let re = (s[0] as f32 - CU8_MID) / CU8_MID;
                let im = (s[1] as f32 - CU8_MID) / CU8_MID;
                out.push(Complex::new(re, im));
            }
        }
    }
    consumed
}

/// Encode `samples` into `format`, appending the bytes to `out`.
///
/// The inverse of [`decode_append`] under the same full-scale convention.
/// Integer formats clamp to [−1.0, 1.0] before quantising (round to nearest),
/// so encoding never wraps; a decode of the encode is within half a
/// quantisation step of the clamped input. `cf32` is exact.
pub fn encode_append(format: IqFormat, samples: &[Complex<f32>], out: &mut Vec<u8>) {
    out.reserve(samples.len() * format.bytes_per_sample());
    match format {
        IqFormat::Cf32 => {
            for s in samples {
                out.extend_from_slice(&s.re.to_le_bytes());
                out.extend_from_slice(&s.im.to_le_bytes());
            }
        }
        IqFormat::Cs32 => {
            for s in samples {
                // `1.0 * CS32_SCALE` is 2³¹, one past `i32::MAX`; the `as`
                // cast saturates, so full scale encodes to the full-scale
                // code rather than wrapping to the negative rail.
                let re = (s.re.clamp(-1.0, 1.0) * CS32_SCALE).round() as i32;
                let im = (s.im.clamp(-1.0, 1.0) * CS32_SCALE).round() as i32;
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
            }
        }
        IqFormat::Cs16 => {
            for s in samples {
                let re = (s.re.clamp(-1.0, 1.0) * CS16_SCALE).round() as i16;
                let im = (s.im.clamp(-1.0, 1.0) * CS16_SCALE).round() as i16;
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
            }
        }
        IqFormat::Cs8 => {
            for s in samples {
                out.push((s.re.clamp(-1.0, 1.0) * CS8_SCALE).round() as i8 as u8);
                out.push((s.im.clamp(-1.0, 1.0) * CS8_SCALE).round() as i8 as u8);
            }
        }
        IqFormat::Cu8 => {
            for s in samples {
                out.push((s.re.clamp(-1.0, 1.0) * CU8_MID + CU8_MID).round() as u8);
                out.push((s.im.clamp(-1.0, 1.0) * CU8_MID + CU8_MID).round() as u8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_and_display_round_trip() {
        for f in IqFormat::ALL {
            assert_eq!(f.name().parse::<IqFormat>(), Ok(f));
            assert_eq!(f.to_string(), f.name());
        }
        let err = "iq16".parse::<IqFormat>().unwrap_err();
        let msg = err.to_string();
        for name in ["cf32", "cs32", "cs16", "cs8", "cu8"] {
            assert!(msg.contains(name), "unhelpful message: {msg}");
        }
    }

    /// D-055: `sc16` is a second *spelling*, never a second variant — it
    /// resolves to the very same `IqFormat::Cs16`, and the canonical name
    /// stays `cs16` so nothing downstream sees two names for one layout.
    #[test]
    fn sc16_is_an_alias_of_cs16_and_never_a_variant() {
        assert_eq!(IqFormat::ALL.len(), 5, "the alias must not add a variant");
        assert_eq!("sc16".parse::<IqFormat>(), Ok(IqFormat::Cs16));
        assert_eq!(IqFormat::from_name("sc16"), Some(IqFormat::Cs16));
        assert_eq!(IqFormat::Cs16.name(), "cs16");
        assert_eq!(IqFormat::Cs16.to_string(), "cs16");
    }

    /// The lane's explicit seal: the same bytes read as `.sc16` and as
    /// `.cs16` decode to the same samples. If they ever diverge, the alias
    /// is wrong.
    #[test]
    fn sc16_and_cs16_decode_identically() {
        let sc16 = IqFormat::from_extension("sc16").expect("sc16 extension");
        let cs16 = IqFormat::from_extension("cs16").expect("cs16 extension");
        assert_eq!(sc16, cs16);
        let bytes: Vec<u8> = (0..64u16)
            .flat_map(|v| (v.wrapping_mul(1031) as i16).to_le_bytes())
            .collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert_eq!(
            decode_append(sc16, &bytes, &mut a),
            decode_append(cs16, &bytes, &mut b)
        );
        assert!(!a.is_empty());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(
                (x.re.to_bits(), x.im.to_bits()),
                (y.re.to_bits(), y.im.to_bits())
            );
        }
    }

    #[test]
    fn extensions_map_case_insensitively_and_never_guess() {
        let cases = [
            ("cf32", IqFormat::Cf32),
            ("cfile", IqFormat::Cf32),
            ("CFile", IqFormat::Cf32),
            ("cs16", IqFormat::Cs16),
            ("sc16", IqFormat::Cs16),
            ("SC16", IqFormat::Cs16),
            ("cs32", IqFormat::Cs32),
            ("sc32", IqFormat::Cs32),
            ("SC32", IqFormat::Cs32),
            ("ci32", IqFormat::Cs32),
            ("cs8", IqFormat::Cs8),
            ("cu8", IqFormat::Cu8),
        ];
        for (ext, want) in cases {
            assert_eq!(IqFormat::from_extension(ext), Some(want), "{ext}");
        }
        for ext in ["bin", "iq", "dat", "fc32", "ci24", "cs64", "sigmf-data", ""] {
            assert_eq!(
                IqFormat::from_extension(ext),
                None,
                "{ext} must not be guessed"
            );
        }
    }

    #[test]
    fn full_scale_codes_decode_to_exactly_unit_magnitude() {
        let cases: [(IqFormat, Vec<u8>); 4] = [
            (
                IqFormat::Cs32,
                [i32::MAX, -i32::MAX]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            (
                IqFormat::Cs16,
                [32767i16, -32767i16]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            (IqFormat::Cs8, vec![127u8, 129u8]), // +127, −127 as two's complement
            (IqFormat::Cu8, vec![255u8, 0u8]),
        ];
        for (format, bytes) in cases {
            let mut out = Vec::new();
            assert_eq!(decode_append(format, &bytes, &mut out), bytes.len());
            assert_eq!(out.len(), 1, "{format}");
            assert_eq!(out[0].re, 1.0, "{format} full-scale positive");
            assert_eq!(out[0].im, -1.0, "{format} full-scale negative");
        }
    }

    #[test]
    fn decode_leaves_a_partial_trailing_sample_unconsumed() {
        for format in IqFormat::ALL {
            let bps = format.bytes_per_sample();
            let bytes = vec![0u8; 3 * bps + (bps - 1)];
            let mut out = Vec::new();
            let consumed = decode_append(format, &bytes, &mut out);
            assert_eq!(consumed, 3 * bps, "{format}");
            assert_eq!(out.len(), 3, "{format}");
        }
    }

    #[test]
    fn encode_clamps_out_of_range_values_to_full_scale() {
        let over = [Complex::new(2.0f32, -2.0f32)];
        let expect_unit = |format: IqFormat| {
            let mut bytes = Vec::new();
            encode_append(format, &over, &mut bytes);
            let mut back = Vec::new();
            decode_append(format, &bytes, &mut back);
            assert_eq!(back[0].re, 1.0, "{format}");
            assert_eq!(back[0].im, -1.0, "{format}");
        };
        expect_unit(IqFormat::Cs32);
        expect_unit(IqFormat::Cs16);
        expect_unit(IqFormat::Cs8);
        expect_unit(IqFormat::Cu8);
    }

    #[test]
    fn encode_decode_error_is_within_half_a_quantisation_step() {
        // A deterministic spread of values across [−1, 1].
        let samples: Vec<Complex<f32>> = (0..512)
            .map(|i| {
                let v = i as f32 / 511.0 * 2.0 - 1.0;
                Complex::new(v, -v)
            })
            .collect();
        let tolerances = [
            (IqFormat::Cs32, 0.5 / CS32_SCALE),
            (IqFormat::Cs16, 0.5 / CS16_SCALE),
            (IqFormat::Cs8, 0.5 / CS8_SCALE),
            (IqFormat::Cu8, 0.5 / CU8_MID),
        ];
        for (format, tol) in tolerances {
            let mut bytes = Vec::new();
            encode_append(format, &samples, &mut bytes);
            let mut back = Vec::new();
            decode_append(format, &bytes, &mut back);
            for (i, (a, b)) in samples.iter().zip(&back).enumerate() {
                assert!(
                    (a.re - b.re).abs() <= tol + f32::EPSILON
                        && (a.im - b.im).abs() <= tol + f32::EPSILON,
                    "{format} sample {i}: {a} vs {b}"
                );
            }
        }
        // cf32 is bit-exact.
        let mut bytes = Vec::new();
        encode_append(IqFormat::Cf32, &samples, &mut bytes);
        let mut back = Vec::new();
        decode_append(IqFormat::Cf32, &bytes, &mut back);
        for (a, b) in samples.iter().zip(&back) {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits())
            );
        }
    }

    /// D-062: `ci32` and `sc32` are spellings of the one `Cs32` variant — the
    /// mistake this test exists to catch is a second variant for the same
    /// byte layout.
    #[test]
    fn ci32_and_sc32_are_spellings_of_the_one_cs32_variant() {
        for spelling in ["cs32", "ci32", "sc32"] {
            assert_eq!(
                spelling.parse::<IqFormat>(),
                Ok(IqFormat::Cs32),
                "{spelling}"
            );
            assert_eq!(IqFormat::from_name(spelling), Some(IqFormat::Cs32));
            assert_eq!(IqFormat::from_extension(spelling), Some(IqFormat::Cs32));
        }
        // One layout, one canonical name: nothing downstream ever sees the
        // alias spelled back at it.
        assert_eq!(IqFormat::Cs32.name(), "cs32");
        assert_eq!(IqFormat::Cs32.to_string(), "cs32");
        assert_eq!(
            IqFormat::ALL
                .iter()
                .filter(|f| f.bytes_per_sample() == 8 && **f != IqFormat::Cf32)
                .count(),
            1,
            "cs32 must be the only 8-byte integer variant"
        );
    }

    /// The D-062 scale, asserted on the codes rather than on the constant:
    /// full scale is `±i32::MAX` and it decodes to exactly `±1.0`, so a
    /// full-scale tone reads 0.0 dBFS (D-006) with no fudge anywhere.
    #[test]
    fn cs32_decodes_at_the_documented_scale() {
        let codes = [
            (i32::MAX, 1.0f32),
            (-i32::MAX, -1.0),
            (0, 0.0),
            (i32::MAX / 2, 0.5),
            (-(i32::MAX / 2), -0.5),
        ];
        for (code, want) in codes {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&code.to_le_bytes());
            bytes.extend_from_slice(&code.to_le_bytes());
            let mut out = Vec::new();
            assert_eq!(decode_append(IqFormat::Cs32, &bytes, &mut out), 8);
            assert!(
                (out[0].re - want).abs() <= 0.5 / CS32_SCALE,
                "code {code} decoded to {}, expected {want}",
                out[0].re
            );
        }
        assert_eq!(IqFormat::Cs32.bytes_per_sample(), 8);
        // The i32 layout is little-endian and I-then-Q: 1 in I, 2 in Q.
        let mut out = Vec::new();
        let bytes = [1u8, 0, 0, 0, 2, 0, 0, 0];
        decode_append(IqFormat::Cs32, &bytes, &mut out);
        assert!(out[0].re > 0.0 && out[0].im > out[0].re);
    }

    /// The lane's round-trip seal at unit level: the same samples written as
    /// `cs32` and as `cf32` decode to the same values within the `cs32`
    /// quantisation bound. If the scale were wrong the two would differ by a
    /// constant factor, which is exactly the plausible-looking defect D-062
    /// names.
    #[test]
    fn cs32_and_cf32_decode_to_the_same_values() {
        let samples: Vec<Complex<f32>> = (0..1024)
            .map(|i| {
                let v = i as f32 / 1023.0 * 2.0 - 1.0;
                Complex::new(v, -v * 0.5)
            })
            .collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        encode_append(IqFormat::Cs32, &samples, &mut a);
        encode_append(IqFormat::Cf32, &samples, &mut b);
        let (mut from_cs32, mut from_cf32) = (Vec::new(), Vec::new());
        decode_append(IqFormat::Cs32, &a, &mut from_cs32);
        decode_append(IqFormat::Cf32, &b, &mut from_cf32);
        assert_eq!(from_cs32.len(), samples.len());
        for (i, (x, y)) in from_cs32.iter().zip(&from_cf32).enumerate() {
            assert!(
                (x.re - y.re).abs() <= 0.5 / CS32_SCALE && (x.im - y.im).abs() <= 0.5 / CS32_SCALE,
                "sample {i}: cs32 {x} vs cf32 {y}"
            );
        }
    }
}
