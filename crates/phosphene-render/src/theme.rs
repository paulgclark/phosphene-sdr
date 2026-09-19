// SPDX-License-Identifier: MIT

//! The D-010 visual identity: modern, striking, slightly hacker-ish.
//!
//! Dark ground, high-contrast luminous trace, crisp monospace chrome, minimal
//! borders — the data does the talking. The trace hue leans yellow-green in a
//! nod to the P7 phosphor heritage the product trades on, but this is a
//! palette, not a CRT costume: no scanlines, no bezels, no fake glass (D-010
//! non-goal), and never at the cost of legibility.
//!
//! All chrome text is set in JetBrains Mono (OFL-1.1 — the LC-2 permitted font
//! license), vendored under `assets/fonts/`. egui's bundled default fonts are
//! disabled because they include a face under Ubuntu-font-1.0, which is not in
//! the LC-2 permitted set.

use std::sync::Arc;

use egui::{Color32, FontData, FontDefinitions, FontFamily, FontId, TextStyle};

/// JetBrains Mono Regular v2.304, vendored from the upstream release
/// (see `assets/fonts/OFL.txt` for its license).
pub const MONO_FONT: &[u8] = include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");

/// Name under which [`MONO_FONT`] is registered with egui.
pub const MONO_FONT_NAME: &str = "JetBrainsMono";

/// The phosphene color palette (sRGB).
///
/// Kept as plain data so both the egui chrome and the wgpu data surface draw
/// from one source of truth.
#[derive(Debug, Clone, Copy)]
pub struct Theme {
    /// Window / data-surface ground: near-black with a faint green cast.
    pub bg: Color32,
    /// Chrome panel fill (top and status bars) — a step darker than `bg` so
    /// the data surface reads as the luminous element.
    pub panel_bg: Color32,
    /// Hairline separating chrome from the data surface.
    pub panel_edge: Color32,
    /// Minor grid lines (one per division).
    pub grid_minor: Color32,
    /// Grid frame (the border of the data surface).
    pub grid_frame: Color32,
    /// Crisp core of the live trace.
    pub trace_core: Color32,
    /// Wide low-alpha halo under the core line.
    pub trace_glow: Color32,
    /// Top of the gradient fill under the trace (fades to transparent).
    pub trace_fill: Color32,
    /// The max-hold trace line (FR-D3): warm amber, visually distinct from
    /// the yellow-green live trace and deliberately subordinate to it
    /// (D-010) — drawn as a bare thin stroke, no fill, no glow.
    pub max_hold_line: Color32,
    /// Primary chrome text.
    pub text: Color32,
    /// De-emphasized chrome text (labels, units).
    pub text_dim: Color32,
    /// Accent text — readouts, the wordmark.
    pub accent: Color32,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: Color32::from_rgb(7, 10, 9),
            panel_bg: Color32::from_rgb(4, 6, 5),
            panel_edge: Color32::from_rgb(26, 38, 32),
            grid_minor: Color32::from_rgba_unmultiplied(130, 200, 160, 44),
            grid_frame: Color32::from_rgba_unmultiplied(130, 200, 160, 96),
            trace_core: Color32::from_rgb(214, 255, 128),
            trace_glow: Color32::from_rgba_unmultiplied(170, 245, 110, 56),
            trace_fill: Color32::from_rgba_unmultiplied(140, 230, 100, 46),
            max_hold_line: Color32::from_rgba_unmultiplied(255, 176, 96, 170),
            text: Color32::from_rgb(168, 196, 178),
            text_dim: Color32::from_rgb(104, 128, 114),
            accent: Color32::from_rgb(214, 255, 128),
        }
    }
}

impl Theme {
    /// `bg` as a linear-space wgpu clear color.
    pub fn clear_color(&self) -> wgpu::Color {
        let [r, g, b, _] = srgb_to_linear(self.bg);
        wgpu::Color {
            r: r as f64,
            g: g as f64,
            b: b as f64,
            a: 1.0,
        }
    }

    /// Install fonts and widget styling on an egui context. Idempotent; call
    /// once after creating the context (windowed and headless paths alike).
    pub fn install(&self, ctx: &egui::Context) {
        let mut fonts = FontDefinitions::empty();
        fonts.font_data.insert(
            MONO_FONT_NAME.to_owned(),
            Arc::new(FontData::from_static(MONO_FONT)),
        );
        // One face everywhere: the chrome is monospace by design (D-010).
        fonts
            .families
            .insert(FontFamily::Proportional, vec![MONO_FONT_NAME.to_owned()]);
        fonts
            .families
            .insert(FontFamily::Monospace, vec![MONO_FONT_NAME.to_owned()]);
        ctx.set_fonts(fonts);

        // The identity is one deliberate dark look — install it on every
        // theme slot so the host theme preference cannot restyle us.
        let this = *self;
        ctx.all_styles_mut(move |style| {
            style.text_styles = [
                (TextStyle::Heading, FontId::monospace(16.0)),
                (TextStyle::Body, FontId::monospace(12.0)),
                (TextStyle::Monospace, FontId::monospace(12.0)),
                (TextStyle::Button, FontId::monospace(12.0)),
                (TextStyle::Small, FontId::monospace(10.0)),
            ]
            .into();

            let visuals = &mut style.visuals;
            *visuals = egui::Visuals::dark();
            visuals.panel_fill = this.panel_bg;
            visuals.window_fill = this.panel_bg;
            visuals.extreme_bg_color = this.bg;
            visuals.override_text_color = Some(this.text);
            // Minimal borders (D-010): kill the default widget chrome.
            visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
            visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, this.text);
        });
    }
}

/// Convert an sRGB-encoded [`Color32`] to a linear-space, straight-alpha RGBA
/// quadruple for use as wgpu vertex color.
pub fn srgb_to_linear(c: Color32) -> [f32; 4] {
    fn channel(v: u8) -> f32 {
        let v = v as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }
    [
        channel(c.r()),
        channel(c.g()),
        channel(c.b()),
        c.a() as f32 / 255.0,
    ]
}

/// Uppercase small-caps-style chrome text helper: the D-010 chrome idiom.
pub fn chrome_text(text: &str, size: f32, color: Color32) -> egui::RichText {
    egui::RichText::new(text.to_uppercase())
        .font(FontId::monospace(size))
        .color(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_conversion_endpoints() {
        let black = srgb_to_linear(Color32::from_rgb(0, 0, 0));
        assert_eq!(&black[..3], &[0.0, 0.0, 0.0]);
        let white = srgb_to_linear(Color32::from_rgb(255, 255, 255));
        for ch in &white[..3] {
            assert!((ch - 1.0).abs() < 1e-6);
        }
        // Mid grey: sRGB 128 is ~0.2158 linear, definitely not 0.5.
        let mid = srgb_to_linear(Color32::from_rgb(128, 128, 128));
        assert!((mid[0] - 0.2158).abs() < 1e-3);
    }

    #[test]
    fn alpha_passes_through_unchanged() {
        let c = srgb_to_linear(Color32::from_rgba_unmultiplied(10, 20, 30, 51));
        assert!((c[3] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn theme_installs_mono_font_on_context() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        // Laying out text forces font atlas creation — this panics if no
        // usable font is registered (we disabled egui's default fonts).
        let out = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.label("0 DBFS");
        });
        out.drop_without_applying_deltas();
    }
}
