// SPDX-License-Identifier: MIT

//! Display-side frequency zoom (FR-D6, D-011).
//!
//! A [`ZoomSpan`] is the visible window of the full frequency span, held as a
//! pair of fractions of that span. It **reinterprets the existing FFT** —
//! there is no device retune, no re-tasking of the source, and no coupling to
//! device control anywhere in this type or its consumers (D-011 struck
//! FR-D6's optional recompute clause for exactly that reason).
//!
//! Every constructor sanitizes: whatever sequence of drags, scrolls and
//! resets is applied, the span is always finite, non-inverted, at least
//! [`MIN_WIDTH`] wide and inside `[0, 1]`. [`ZoomSpan::FULL`] is the exact
//! identity, so reset is bit-for-bit reversible.

/// Minimum visible width as a fraction of the full span (a 10 000× ceiling on
/// zoom). Deep enough that a 32 768-bin FFT can be read bin-by-bin, shallow
/// enough that arithmetic keeps plenty of float headroom.
pub const MIN_WIDTH: f64 = 1e-4;

/// The visible window of the full frequency span, as fractions `0 ≤ lo < hi
/// ≤ 1` of it. `FULL` (0..1) is the unzoomed identity.
///
/// Fields are private so no un-sanitized span can exist; construct via
/// [`ZoomSpan::new`], [`ZoomSpan::zoom_to`] or [`ZoomSpan::zoom_about`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoomSpan {
    lo: f64,
    hi: f64,
}

impl Default for ZoomSpan {
    fn default() -> Self {
        Self::FULL
    }
}

impl ZoomSpan {
    /// The unzoomed identity: the full span, exactly.
    pub const FULL: ZoomSpan = ZoomSpan { lo: 0.0, hi: 1.0 };

    /// Sanitizing constructor: non-finite input yields [`Self::FULL`];
    /// endpoints are ordered, clamped into `[0, 1]` and widened to
    /// [`MIN_WIDTH`] (about their midpoint) if the request is narrower.
    pub fn new(a: f64, b: f64) -> Self {
        if !a.is_finite() || !b.is_finite() {
            return Self::FULL;
        }
        let (mut lo, mut hi) = if a <= b { (a, b) } else { (b, a) };
        lo = lo.clamp(0.0, 1.0);
        hi = hi.clamp(0.0, 1.0);
        if hi - lo < MIN_WIDTH {
            let mid = ((lo + hi) * 0.5).clamp(MIN_WIDTH * 0.5, 1.0 - MIN_WIDTH * 0.5);
            lo = mid - MIN_WIDTH * 0.5;
            hi = mid + MIN_WIDTH * 0.5;
        }
        Self { lo, hi }
    }

    /// Lower edge of the window, as a fraction of the full span.
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Upper edge of the window, as a fraction of the full span.
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// Visible width as a fraction of the full span. Always in
    /// `[MIN_WIDTH, 1]`.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Whether this is exactly the unzoomed full span.
    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    /// Map a position across the *visible* window (`t` in 0..1, left to
    /// right) to a fraction of the *full* span. This is the one mapping the
    /// axis labels, the cursor readout and the data surfaces all share.
    pub fn view_to_full(&self, t: f64) -> f64 {
        self.lo + t * self.width()
    }

    /// Zoom to the sub-window `a..b` of the **current view** (both in 0..1
    /// view fractions) — the drag-select gesture. Zooms compose: the result
    /// is expressed in full-span fractions.
    pub fn zoom_to(&self, a: f64, b: f64) -> Self {
        Self::new(self.view_to_full(a), self.view_to_full(b))
    }

    /// Zoom by `factor` (> 1 zooms in) about the view fraction `t` — the
    /// scroll-wheel gesture. The full-span point under the cursor stays under
    /// the cursor. Non-finite or non-positive factors are ignored.
    pub fn zoom_about(&self, t: f64, factor: f64) -> Self {
        if !factor.is_finite() || factor <= 0.0 || !t.is_finite() {
            return *self;
        }
        let t = t.clamp(0.0, 1.0);
        let new_width = (self.width() / factor).clamp(MIN_WIDTH, 1.0);
        let anchor = self.view_to_full(t);
        let mut lo = anchor - t * new_width;
        // Keep the window inside the full span by shifting, never shrinking.
        lo = lo.clamp(0.0, 1.0 - new_width);
        Self {
            lo,
            hi: lo + new_width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(z: &ZoomSpan) -> bool {
        z.lo.is_finite()
            && z.hi.is_finite()
            && z.lo >= 0.0
            && z.hi <= 1.0
            && (z.hi - z.lo) >= MIN_WIDTH * 0.999
    }

    #[test]
    fn reset_is_exact() {
        // The seal: reset returns to exactly the pre-zoom span.
        let z = ZoomSpan::FULL.zoom_to(0.2, 0.7).zoom_about(0.5, 3.0);
        assert!(!z.is_full());
        assert_eq!(ZoomSpan::FULL, ZoomSpan::new(0.0, 1.0));
        assert!(ZoomSpan::FULL.is_full());
        assert_eq!(ZoomSpan::FULL.lo(), 0.0);
        assert_eq!(ZoomSpan::FULL.hi(), 1.0);
    }

    #[test]
    fn no_sequence_produces_an_invalid_span() {
        // The seal: no zoom sequence can produce an inverted, zero-width or
        // NaN span. Drive a mix of hostile and ordinary operations.
        let mut z = ZoomSpan::FULL;
        let ops: &[(f64, f64, bool)] = &[
            (0.9, 0.1, true),            // inverted drag
            (0.5, 0.5, true),            // zero-width drag
            (f64::NAN, 0.3, true),       // NaN drag
            (0.2, 0.8, true),            // ordinary drag
            (-3.0, 7.0, true),           // out-of-range drag
            (0.5, 10.0, false),          // deep scroll in
            (0.5, f64::INFINITY, false), // infinite factor
            (0.0, 0.5, false),           // scroll out at the left edge
            (1.0, 1e9, false),           // absurd zoom at the right edge
            (0.5, -2.0, false),          // negative factor
            (f64::NAN, 2.0, false),      // NaN anchor
            (0.001, 0.0011, true),       // sub-minimum drag
        ];
        for &(a, b, is_drag) in ops {
            z = if is_drag {
                z.zoom_to(a, b)
            } else {
                z.zoom_about(a, b)
            };
            assert!(valid(&z), "invalid span after op ({a}, {b}): {z:?}");
        }
        // Repeated deep zoom saturates at MIN_WIDTH, still valid.
        for _ in 0..200 {
            z = z.zoom_about(0.37, 2.0);
            assert!(valid(&z));
        }
        assert!((z.width() - MIN_WIDTH).abs() < MIN_WIDTH * 0.01);
        // And zooming all the way back out lands exactly on the full span.
        for _ in 0..200 {
            z = z.zoom_about(0.5, 0.5);
        }
        assert!(z.is_full(), "zoom-out did not saturate at FULL: {z:?}");
    }

    #[test]
    fn zoom_about_keeps_the_anchor_fixed() {
        let z = ZoomSpan::new(0.2, 0.8);
        let t = 0.25;
        let anchor = z.view_to_full(t);
        let zi = z.zoom_about(t, 2.0);
        assert!((zi.view_to_full(t) - anchor).abs() < 1e-12);
        assert!((zi.width() - z.width() / 2.0).abs() < 1e-12);
    }

    #[test]
    fn zooms_compose_in_full_span_fractions() {
        // Selecting the middle half twice = the middle quarter of the span.
        let z = ZoomSpan::FULL.zoom_to(0.25, 0.75).zoom_to(0.25, 0.75);
        assert!((z.lo() - 0.375).abs() < 1e-12);
        assert!((z.hi() - 0.625).abs() < 1e-12);
    }

    #[test]
    fn view_to_full_endpoints() {
        let z = ZoomSpan::new(0.3, 0.6);
        assert!((z.view_to_full(0.0) - 0.3).abs() < 1e-12);
        assert!((z.view_to_full(1.0) - 0.6).abs() < 1e-12);
        assert!((z.view_to_full(0.5) - 0.45).abs() < 1e-12);
    }
}
