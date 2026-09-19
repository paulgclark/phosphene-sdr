// SPDX-License-Identifier: MIT
// The two textured data surfaces (spec §7.3 render half + §7.6): the
// persistence histogram and the waterfall. One quad each, generated from the
// uniform rect; all sampling is explicit textureLoad (no sampler, no
// filterable-float feature requirement) with manual interpolation.

struct SurfaceUniforms {
    // NDC quad rect: x0 (left), y_top, x1 (right), y_bottom.
    rect: vec4<f32>,
    // x: colormap LUT row, y: histogram gamma (§7.3), z: epsilon floor,
    // w: waterfall gamma γ_wf (§7.6).
    sel: vec4<f32>,
    // Histogram: x = levels, y = bins, z = zoom window lo, w = zoom width.
    // Waterfall: x = bins, y = ring rows H, z = head (next write row),
    //            w = visible rows.
    dims: vec4<f32>,
    // x: dB at colour-scale bottom, y: dB at top (waterfall mapping),
    // z: zoom window lo, w: zoom width (waterfall — dims is full there).
    range: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SurfaceUniforms;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Four-vertex triangle strip over the uniform rect; uv.y = 0 at the top.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let ux = f32(vi & 1u);
    let uy = f32(vi >> 1u);
    var out: VsOut;
    out.pos = vec4<f32>(mix(u.rect.x, u.rect.z, ux), mix(u.rect.y, u.rect.w, uy), 0.0, 1.0);
    out.uv = vec2<f32>(ux, uy);
    return out;
}

// Linear-interpolated LUT lookup on the row the uniform selects. The LUT
// texture is sRGB, so the loaded texels are already linear-light.
fn cmap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 255.0;
    let i0 = i32(floor(x));
    let i1 = min(i0 + 1, 255);
    let row = i32(u.sel.x + 0.5);
    let c0 = textureLoad(lut_tex, vec2<i32>(i0, row), 0).rgb;
    let c1 = textureLoad(lut_tex, vec2<i32>(i1, row), 0).rgb;
    return mix(c0, c1, x - f32(i0));
}

// Bilinear sample of the intensity grid. Texel (x = level, y = bin) — the
// row-major-by-bin layout PersistenceHistogram::intensity() hands over.
fn sample_intensity(level_c: f32, bin_c: f32) -> f32 {
    let l0 = i32(floor(level_c));
    let b0 = i32(floor(bin_c));
    let l1 = min(l0 + 1, i32(u.dims.x) - 1);
    let b1 = min(b0 + 1, i32(u.dims.y) - 1);
    let fl = level_c - f32(l0);
    let fb = bin_c - f32(b0);
    let v00 = textureLoad(data_tex, vec2<i32>(l0, b0), 0).r;
    let v10 = textureLoad(data_tex, vec2<i32>(l1, b0), 0).r;
    let v01 = textureLoad(data_tex, vec2<i32>(l0, b1), 0).r;
    let v11 = textureLoad(data_tex, vec2<i32>(l1, b1), 0).r;
    return mix(mix(v00, v10, fl), mix(v01, v11, fl), fb);
}

// §7.3 render half: colour = cmap(gamma_adjust(I)); cells below ε render as
// background — output alpha 0 so the cleared ground (and the grid beneath)
// shows through instead of the colormap's zero end.
@fragment
fn fs_histogram(in: VsOut) -> @location(0) vec4<f32> {
    let level_c = (1.0 - in.uv.y) * (u.dims.x - 1.0);
    // Display-side zoom (FR-D6, D-011): reinterpret the existing grid
    // through the visible window — the same mapping the axis labels use.
    let bin_c = (u.dims.z + in.uv.x * u.dims.w) * (u.dims.y - 1.0);
    let i = sample_intensity(level_c, bin_c);
    if i < u.sel.z {
        return vec4<f32>(0.0);
    }
    let g = pow(clamp(i, 0.0, 1.0), u.sel.y);
    return vec4<f32>(cmap(g), 1.0);
}

// §7.6: newest row at the top; the scroll is the ring offset computed here —
// rows are written once and never rewritten to move.
@fragment
fn fs_waterfall(in: VsOut) -> @location(0) vec4<f32> {
    let bins = u.dims.x;
    let ring_rows = i32(u.dims.y + 0.5);
    let vis = max(u.dims.w, 1.0);
    let screen_row = min(floor(in.uv.y * vis), vis - 1.0);
    let idx = i32(u.dims.z + 0.5) - 1 - i32(screen_row);
    let ring_row = ((idx % ring_rows) + ring_rows) % ring_rows;

    // Linear interpolation along frequency only — each row is one instant;
    // blending adjacent rows would smear transients across time. The zoom
    // window (range.zw) applies exactly as on the histogram.
    let bin_c = (u.range.z + in.uv.x * u.range.w) * (bins - 1.0);
    let b0 = i32(floor(bin_c));
    let b1 = min(b0 + 1, i32(bins) - 1);
    let fb = bin_c - f32(b0);
    let d0 = textureLoad(data_tex, vec2<i32>(b0, ring_row), 0).r;
    let d1 = textureLoad(data_tex, vec2<i32>(b1, ring_row), 0).r;
    let db = mix(d0, d1, fb);

    // §7.6 colour mapping, in two steps. Normalise the row's dBFS against
    // the colour-scale range — shared with the histogram, or the auto-range
    // over the retained rows — then shape the normalised value with the
    // power-law intensity control before the LUT lookup. Applying γ_wf
    // AFTER the normalisation is what makes it meaningful in both range
    // modes: the range decides which dB window the colours span, γ_wf
    // decides how they are distributed inside it.
    let t = clamp((db - u.range.x) / max(u.range.y - u.range.x, 1e-6), 0.0, 1.0);
    return vec4<f32>(cmap(pow(t, u.sel.w)), 1.0);
}
