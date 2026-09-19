// SPDX-License-Identifier: MIT

//! FR-D8 colormaps: the four intensity→colour ramps, and the LUT layout that
//! makes D-009's live switch free.
//!
//! ## Provenance and licences (the LC-2 audit trail)
//!
//! `cargo deny` cannot audit a copied data table, so the record lives here:
//!
//! * **`p7`** — in-house, ours by construction (spec §2 branding note): a
//!   blue-flash low end into the yellow-green CRT-afterglow high end, named
//!   for the WWII radar phosphor. The anchor colours below were designed for
//!   this project and derive from no other colormap.
//! * **`inferno`** (the FR-D8 "*inferno*-like" ramp) and **`viridis`** — piecewise-linear
//!   interpolations of the widely published 10-anchor discretizations of
//!   matplotlib's colormaps. matplotlib releases these colormap data **CC0**
//!   (public domain), a licence LC-2 permits.
//! * **`turbo`** — piecewise-linear interpolation of the published 9-anchor
//!   discretization of Google's Turbo colormap, released under
//!   **Apache-2.0**, a licence LC-2 permits.
//!
//! **No fosphor table was copied, consulted, or transcribed** (LC-1/§4.3):
//! the two permissively licensed ramps come from matplotlib and Google as
//! recorded above, and the other two are original.
//!
//! ## Where the LUT lives, and why (D-009)
//!
//! D-009 requires the active colormap to be switchable live with **no
//! pipeline rebuild and no visible stall**. That is a constraint on where the
//! LUT data lives, so the choice is deliberate: all four ramps are baked into
//! **one 256×4 `Rgba8UnormSrgb` texture** ([`lut_atlas_rgba8`], one row per
//! map in [`Colormap::ALL`] order) uploaded once when the surface renderer is
//! built. The active map is nothing but a row index carried in the per-panel
//! uniform buffer the frame already writes — so a switch writes four bytes of
//! uniform data and touches no pipeline, no shader, no texture and no bind
//! group. The no-rebuild requirement is met structurally, not by careful
//! scheduling.

/// Number of entries in each colormap's lookup table.
pub const LUT_SIZE: usize = 256;

/// The four shipped colormaps (FR-D8). `P7` is the flagship default (D-010).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Colormap {
    /// In-house long-persistence ramp: blue flash into yellow-green CRT
    /// afterglow. The default (D-010).
    #[default]
    P7,
    /// The FR-D8 "inferno-like" ramp (from matplotlib's CC0 inferno).
    Inferno,
    /// Colorblind-safe ramp (from matplotlib's CC0 viridis).
    Viridis,
    /// High-contrast rainbow ramp (from Google's Apache-2.0 turbo).
    Turbo,
}

impl Colormap {
    /// Every colormap, in atlas-row and cycle order.
    pub const ALL: [Colormap; 4] = [
        Colormap::P7,
        Colormap::Inferno,
        Colormap::Viridis,
        Colormap::Turbo,
    ];

    /// Row of this map in the LUT atlas ([`lut_atlas_rgba8`]).
    pub fn index(self) -> usize {
        match self {
            Colormap::P7 => 0,
            Colormap::Inferno => 1,
            Colormap::Viridis => 2,
            Colormap::Turbo => 3,
        }
    }

    /// The next map in cycle order, wrapping — the D-009 keyboard-cycle step.
    pub fn next(self) -> Colormap {
        Self::ALL[(self.index() + 1) % Self::ALL.len()]
    }

    /// Lower-case display name.
    pub fn name(self) -> &'static str {
        match self {
            Colormap::P7 => "p7",
            Colormap::Inferno => "inferno",
            Colormap::Viridis => "viridis",
            Colormap::Turbo => "turbo",
        }
    }

    /// Anchor points `(position 0..=1, sRGB colour)` the LUT interpolates.
    fn anchors(self) -> &'static [(f32, [u8; 3])] {
        match self {
            // Ours by construction: dark ground, deep-blue flash for the
            // rarely-hit cells, saturating through green into the sustained
            // yellow-green afterglow that gives the phosphor its name.
            Colormap::P7 => &[
                (0.00, [0x02, 0x04, 0x09]),
                (0.10, [0x10, 0x26, 0x5e]),
                (0.22, [0x23, 0x50, 0xc8]),
                (0.38, [0x2e, 0x86, 0xa8]),
                (0.55, [0x3f, 0xbf, 0x6e]),
                (0.72, [0x8f, 0xe0, 0x45]),
                (0.88, [0xd9, 0xf7, 0x6a]),
                (1.00, [0xf8, 0xff, 0xc8]),
            ],
            // matplotlib inferno, 10 evenly spaced anchors (CC0).
            Colormap::Inferno => &[
                (0.0, [0x00, 0x00, 0x04]),
                (1.0 / 9.0, [0x16, 0x0b, 0x39]),
                (2.0 / 9.0, [0x42, 0x0a, 0x68]),
                (3.0 / 9.0, [0x6a, 0x17, 0x6e]),
                (4.0 / 9.0, [0x93, 0x26, 0x67]),
                (5.0 / 9.0, [0xbc, 0x37, 0x54]),
                (6.0 / 9.0, [0xdd, 0x51, 0x3a]),
                (7.0 / 9.0, [0xf3, 0x78, 0x19]),
                (8.0 / 9.0, [0xfc, 0xa5, 0x0a]),
                (1.0, [0xfc, 0xff, 0xa4]),
            ],
            // matplotlib viridis, 10 evenly spaced anchors (CC0).
            Colormap::Viridis => &[
                (0.0, [0x44, 0x01, 0x54]),
                (1.0 / 9.0, [0x48, 0x28, 0x78]),
                (2.0 / 9.0, [0x3e, 0x4a, 0x89]),
                (3.0 / 9.0, [0x31, 0x68, 0x8e]),
                (4.0 / 9.0, [0x26, 0x82, 0x8e]),
                (5.0 / 9.0, [0x1f, 0x9e, 0x89]),
                (6.0 / 9.0, [0x35, 0xb7, 0x79]),
                (7.0 / 9.0, [0x6d, 0xcd, 0x59]),
                (8.0 / 9.0, [0xb4, 0xde, 0x2c]),
                (1.0, [0xfd, 0xe7, 0x25]),
            ],
            // Google turbo, 9 evenly spaced anchors (Apache-2.0).
            Colormap::Turbo => &[
                (0.0, [0x30, 0x12, 0x3b]),
                (1.0 / 8.0, [0x46, 0x62, 0xd7]),
                (2.0 / 8.0, [0x36, 0xaa, 0xf9]),
                (3.0 / 8.0, [0x1a, 0xe4, 0xb6]),
                (4.0 / 8.0, [0x72, 0xfe, 0x5e]),
                (5.0 / 8.0, [0xc7, 0xef, 0x34]),
                (6.0 / 8.0, [0xfa, 0xba, 0x39]),
                (7.0 / 8.0, [0xf6, 0x6b, 0x19]),
                (1.0, [0x7a, 0x04, 0x03]),
            ],
        }
    }

    /// The full [`LUT_SIZE`]-entry sRGB lookup table for this map.
    pub fn lut(self) -> Vec<[u8; 3]> {
        let anchors = self.anchors();
        (0..LUT_SIZE)
            .map(|i| {
                let t = i as f32 / (LUT_SIZE - 1) as f32;
                sample_anchors(anchors, t)
            })
            .collect()
    }
}

/// Piecewise-linear interpolation of `anchors` (sorted by position) at `t`.
fn sample_anchors(anchors: &[(f32, [u8; 3])], t: f32) -> [u8; 3] {
    let (first, last) = (anchors[0], anchors[anchors.len() - 1]);
    if t <= first.0 {
        return first.1;
    }
    if t >= last.0 {
        return last.1;
    }
    let hi = anchors.partition_point(|a| a.0 <= t).min(anchors.len() - 1);
    let (t0, c0) = anchors[hi - 1];
    let (t1, c1) = anchors[hi];
    let f = (t - t0) / (t1 - t0);
    let mut out = [0u8; 3];
    for ch in 0..3 {
        let v = c0[ch] as f32 + (c1[ch] as f32 - c0[ch] as f32) * f;
        out[ch] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// The LUT atlas: `LUT_SIZE × Colormap::ALL.len()` RGBA8 texels, row-major,
/// one row per colormap in [`Colormap::ALL`] order, alpha 255. Uploaded once
/// into the `Rgba8UnormSrgb` LUT texture (see the module docs for why).
pub fn lut_atlas_rgba8() -> Vec<u8> {
    let mut out = Vec::with_capacity(LUT_SIZE * Colormap::ALL.len() * 4);
    for map in Colormap::ALL {
        for rgb in map.lut() {
            out.extend_from_slice(&rgb);
            out.push(255);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_endpoints_hit_the_anchor_ends() {
        for map in Colormap::ALL {
            let lut = map.lut();
            assert_eq!(lut.len(), LUT_SIZE);
            let anchors = map.anchors();
            assert_eq!(lut[0], anchors[0].1, "{} zero end", map.name());
            assert_eq!(
                lut[LUT_SIZE - 1],
                anchors[anchors.len() - 1].1,
                "{} top end",
                map.name()
            );
        }
    }

    #[test]
    fn maps_differ_pairwise() {
        // A copy-paste error that aliased two ramps would produce identical
        // LUTs; require every pair to differ substantially somewhere.
        for (i, a) in Colormap::ALL.iter().enumerate() {
            for b in &Colormap::ALL[i + 1..] {
                let (la, lb) = (a.lut(), b.lut());
                let max_diff = la
                    .iter()
                    .zip(&lb)
                    .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)))
                    .max()
                    .unwrap();
                assert!(
                    max_diff > 32,
                    "{} and {} are near-identical",
                    a.name(),
                    b.name()
                );
            }
        }
    }

    #[test]
    fn cycle_visits_all_and_wraps() {
        let mut m = Colormap::default();
        assert_eq!(m, Colormap::P7, "D-010: p7 is the flagship default");
        let mut seen = Vec::new();
        for _ in 0..Colormap::ALL.len() {
            seen.push(m);
            m = m.next();
        }
        assert_eq!(seen, Colormap::ALL.to_vec());
        assert_eq!(m, Colormap::P7, "cycle wraps back to the start");
    }

    #[test]
    fn atlas_rows_match_map_index() {
        let atlas = lut_atlas_rgba8();
        assert_eq!(atlas.len(), LUT_SIZE * Colormap::ALL.len() * 4);
        for map in Colormap::ALL {
            let row = map.index();
            let lut = map.lut();
            for (i, rgb) in lut.iter().enumerate() {
                let o = (row * LUT_SIZE + i) * 4;
                assert_eq!(&atlas[o..o + 3], rgb.as_slice());
                assert_eq!(atlas[o + 3], 255);
            }
        }
    }

    #[test]
    fn ramps_are_monotonically_brightening_where_claimed() {
        // p7, inferno, viridis are perceptual "dark → bright" ramps: a crude
        // luma proxy must never decrease by more than rounding noise. (Turbo
        // is deliberately not monotone — it ends on dark red.)
        for map in [Colormap::P7, Colormap::Inferno, Colormap::Viridis] {
            let lut = map.lut();
            let luma =
                |c: &[u8; 3]| 0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32;
            for w in lut.windows(2) {
                assert!(
                    luma(&w[1]) >= luma(&w[0]) - 1.5,
                    "{} luma dips: {:?} -> {:?}",
                    map.name(),
                    w[0],
                    w[1]
                );
            }
        }
    }
}
