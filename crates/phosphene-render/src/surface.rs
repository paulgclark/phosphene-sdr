// SPDX-License-Identifier: MIT

//! GPU resources for the two textured data surfaces: the persistence
//! histogram (FR-D1, §7.3 render half) and the waterfall (FR-D4, §7.6).
//!
//! Both surfaces draw one quad each through pipelines built **once** at
//! construction; per-frame work is texture uploads and 64-byte uniform
//! writes. The colormap LUT is a single 256×4 atlas texture holding all four
//! ramps (see [`crate::colormap`]), so switching the active map is a uniform
//! row index — no pipeline rebuild, no stall (D-009). The
//! [`pipeline_builds`](SurfaceRenderer::pipeline_builds) counter exists so
//! the seal can assert that structurally.
//!
//! The waterfall is a **ring texture** of [`RING_ROWS`] rows (§7.6: H ≥
//! 1024): each new row is written once at the head cursor and the scroll is a
//! read-side offset in the shader — rewriting every row per frame to scroll
//! is the defect the lane text names, and it is impossible by construction
//! here.

use bytemuck::{Pod, Zeroable};

use crate::colormap::{self, Colormap};
use crate::layout::DisplayParams;
use crate::scene::GridPx;
use crate::waterfall::ROW_FLOOR_DBFS;

/// Height of the waterfall ring texture (§7.6 requires H ≥ 1024).
pub const RING_ROWS: u32 = 1024;

/// Intensity cells below this render as background rather than as the
/// colormap's zero end (§7.3: "cells below a small ε render as background" —
/// a floor that glows is a lie about the noise floor). Starting value; the
/// M1 visual-tuning pass may adjust it.
pub const INTENSITY_FLOOR_EPSILON: f32 = 0.004;

/// Default `gamma_adjust` exponent. §7.3 renders `cmap(gamma_adjust(I))`
/// with the intensity/gamma control marked *tune*; `gamma_adjust` is the
/// standard power-law gamma `I^γ`, and γ < 1 lifts the rare, low-intensity
/// cells the persistence display exists to show. Starting value for the M1
/// visual-tuning pass.
pub const DEFAULT_GAMMA: f32 = 0.7;

/// Bounds on either intensity exponent — §7.3's persistence `gamma_adjust`
/// and §7.6's waterfall `γ_wf`. One pair of bounds for both, so the two
/// controls cover exactly the same ground; outside this range a power law
/// is either a no-op or a hard threshold, and neither is a display.
pub const GAMMA_MIN: f32 = 0.05;
/// Upper bound on either intensity exponent — see [`GAMMA_MIN`].
pub const GAMMA_MAX: f32 = 5.0;

/// Default §7.6 waterfall intensity exponent `γ_wf`.
///
/// **1.0 is the identity**, and that is the point: the waterfall's
/// power→colour mapping was linear in dB before this control existed
/// (D-065 §3, as corrected — §7.3's gamma is read only by the histogram
/// fragment shader and never reached the waterfall at all), so shipping the
/// control at 1.0 adds a control without changing the picture. What the
/// picture *does* change with is D-065 §2's auto-range default; keeping
/// this at the identity is what makes that attribution unambiguous.
pub const DEFAULT_WF_GAMMA: f32 = 1.0;

/// Uniform block shared by both surface pipelines; must match
/// `SurfaceUniforms` in `surface.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SurfaceUniforms {
    /// NDC quad rect: x0, y_top, x1, y_bottom.
    rect: [f32; 4],
    /// x: colormap LUT row, y: §7.3 histogram gamma, z: epsilon floor,
    /// w: §7.6 waterfall gamma `γ_wf`.
    sel: [f32; 4],
    /// Histogram: [levels, bins, zoom lo, zoom width].
    /// Waterfall: [bins, H, head, visible].
    dims: [f32; 4],
    /// x: dB at scale bottom, y: dB at top (waterfall only);
    /// z, w: zoom lo / zoom width (waterfall — dims is full there).
    range: [f32; 4],
}

/// One surface's GPU-side state: its data texture, uniform buffer, and the
/// bind group tying them to the shared LUT.
struct Panel {
    uniforms: wgpu::Buffer,
    tex: wgpu::Texture,
    bind: wgpu::BindGroup,
    w: u32,
    h: u32,
    /// Whether real data has been uploaded (dummy 1×1 textures never draw).
    ready: bool,
    /// Whether this frame's layout gives the panel non-zero area.
    visible: bool,
}

/// Renders the persistence histogram and waterfall surfaces.
pub struct SurfaceRenderer {
    hist_pipeline: wgpu::RenderPipeline,
    wf_pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    lut: wgpu::Texture,
    lut_uploaded: bool,
    lut_view: wgpu::TextureView,
    hist: Panel,
    wf: Panel,
    wf_head: u32,
    wf_written: u32,
    /// Per-ring-row (min, max) over bins above the −200 dB gap filler;
    /// `(INFINITY, NEG_INFINITY)` marks an empty row. Feeds auto-range.
    wf_row_range: Vec<(f32, f32)>,
    colormap: Colormap,
    gamma: f32,
    wf_gamma: f32,
    auto_range: bool,
    pipeline_builds: u32,
}

fn create_data_texture(device: &wgpu::Device, label: &str, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

impl SurfaceRenderer {
    /// Build both pipelines and the colormap LUT atlas texture. The atlas
    /// bytes upload on the first frame (construction has no queue), after
    /// which every colormap a switch could select is resident — that is the
    /// D-009 design.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phosphene-surface-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("surface.wgsl").into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phosphene-surface-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("phosphene-surface-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let mut pipeline_builds = 0;
        let mut build = |entry: &str, label: &str| {
            pipeline_builds += 1;
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipe_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let hist_pipeline = build("fs_histogram", "phosphene-histogram-pipeline");
        let wf_pipeline = build("fs_waterfall", "phosphene-waterfall-pipeline");

        // The LUT atlas: all four ramps, uploaded once, selected by uniform.
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("phosphene-colormap-lut"),
            size: wgpu::Extent3d {
                width: colormap::LUT_SIZE as u32,
                height: Colormap::ALL.len() as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lut_view = lut.create_view(&wgpu::TextureViewDescriptor::default());

        let hist = Panel::new(device, &bind_layout, &lut_view, "phosphene-histogram", 1, 1);
        let wf = Panel::new(device, &bind_layout, &lut_view, "phosphene-waterfall", 1, 1);
        Self {
            hist_pipeline,
            wf_pipeline,
            bind_layout,
            lut,
            lut_uploaded: false,
            lut_view,
            hist,
            wf,
            wf_head: 0,
            wf_written: 0,
            wf_row_range: Vec::new(),
            colormap: Colormap::default(),
            gamma: DEFAULT_GAMMA,
            wf_gamma: DEFAULT_WF_GAMMA,
            auto_range: false,
            pipeline_builds,
        }
    }

    /// Number of render pipelines built so far. Stays constant across
    /// colormap switches — the D-009 no-rebuild seal instrument.
    pub fn pipeline_builds(&self) -> u32 {
        self.pipeline_builds
    }

    /// Total waterfall rows written into the ring since construction.
    pub fn waterfall_rows_written(&self) -> u32 {
        self.wf_written
    }

    /// The active colormap.
    pub fn colormap(&self) -> Colormap {
        self.colormap
    }

    /// Select a colormap. Takes effect next frame via the uniform write the
    /// frame performs anyway.
    pub fn set_colormap(&mut self, map: Colormap) {
        self.colormap = map;
    }

    /// Advance to the next colormap (D-009 cycle seam), returning it.
    pub fn cycle_colormap(&mut self) -> Colormap {
        self.colormap = self.colormap.next();
        self.colormap
    }

    /// The `gamma_adjust` exponent.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Set the `gamma_adjust` exponent (live control, §7.3). Clamped to a
    /// sane range; non-finite input keeps the current value.
    pub fn set_gamma(&mut self, gamma: f32) {
        if gamma.is_finite() {
            self.gamma = gamma.clamp(GAMMA_MIN, GAMMA_MAX);
        }
    }

    /// The §7.6 waterfall intensity exponent `γ_wf`.
    pub fn waterfall_gamma(&self) -> f32 {
        self.wf_gamma
    }

    /// Set the §7.6 waterfall intensity exponent (live control). Clamped to
    /// the same range as §7.3's gamma; non-finite input keeps the current
    /// value.
    ///
    /// Distinct from [`set_gamma`](Self::set_gamma) by construction: this
    /// one reaches the waterfall fragment shader and that one reaches the
    /// histogram's, and neither shader reads the other's exponent.
    pub fn set_waterfall_gamma(&mut self, gamma: f32) {
        if gamma.is_finite() {
            self.wf_gamma = gamma.clamp(GAMMA_MIN, GAMMA_MAX);
        }
    }

    /// Whether the waterfall auto-ranges its colour scale (§7.6 toggle).
    pub fn waterfall_auto_range(&self) -> bool {
        self.auto_range
    }

    /// Toggle waterfall auto-range: on, the colour scale spans the retained
    /// rows' min..max; off, it shares the histogram's displayed dB range.
    pub fn set_waterfall_auto_range(&mut self, on: bool) {
        self.auto_range = on;
    }

    /// Upload the persistence intensity grid (`bins × levels`, row-major by
    /// bin, values in `[0,1]` — the `PersistenceHistogram::intensity()`
    /// layout). An empty slice marks the surface not-ready and nothing draws.
    ///
    /// Panics if `data.len() != bins * levels` — a programmer error.
    pub fn upload_intensity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[f32],
        bins: usize,
        levels: usize,
    ) {
        if data.is_empty() || bins == 0 || levels == 0 {
            self.hist.ready = false;
            return;
        }
        assert_eq!(
            data.len(),
            bins * levels,
            "intensity grid {} != bins {bins} × levels {levels}",
            data.len()
        );
        let (w, h) = (levels as u32, bins as u32);
        if self.hist.w != w || self.hist.h != h {
            self.hist = Panel::new(
                device,
                &self.bind_layout,
                &self.lut_view,
                "phosphene-histogram",
                w,
                h,
            );
        }
        queue.write_texture(
            self.hist.tex.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.hist.ready = true;
    }

    /// Append a **batch** of aggregated waterfall rows (flat `rows × bins`,
    /// oldest first, dBFS per bin). The batch is written in at most two
    /// `write_texture` calls — one contiguous run up to the ring seam and
    /// one after the wrap — never one GPU write per row; D-047 names the
    /// per-row upload as the real per-row expense, and at the fast mode's
    /// cadence (~17 rows per 60 fps frame at 1 ms/row) a per-row write would
    /// be the wrong shape. The scroll stays a read-side offset in the
    /// shader. A batch wider than the ring keeps only its newest
    /// [`RING_ROWS`] rows — the older ones would be overwritten before ever
    /// being scanned out. A new row width recreates the ring (a resize, not
    /// a per-frame event).
    ///
    /// Panics if `rows.len()` is not a multiple of `bins` — a programmer
    /// error.
    pub fn push_waterfall_rows(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rows: &[f32],
        bins: usize,
    ) {
        if rows.is_empty() || bins == 0 {
            return;
        }
        assert_eq!(
            rows.len() % bins,
            0,
            "row batch of {} values is not a multiple of {bins} bins",
            rows.len()
        );
        let w = bins as u32;
        if self.wf.w != w || self.wf.h != RING_ROWS {
            self.wf = Panel::new(
                device,
                &self.bind_layout,
                &self.lut_view,
                "phosphene-waterfall",
                w,
                RING_ROWS,
            );
            // Blank history reads as the −200 dB floor, not as 0 dBFS (an
            // all-zero float texture would render full-scale hot).
            let blank = vec![ROW_FLOOR_DBFS; (w * RING_ROWS) as usize];
            queue.write_texture(
                self.wf.tex.as_image_copy(),
                bytemuck::cast_slice(&blank),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: w,
                    height: RING_ROWS,
                    depth_or_array_layers: 1,
                },
            );
            self.wf_head = 0;
            self.wf_written = 0;
            self.wf_row_range = vec![(f32::INFINITY, f32::NEG_INFINITY); RING_ROWS as usize];
            self.wf.ready = true;
        }
        // Keep only the newest RING_ROWS rows of an oversized batch.
        let total = rows.len() / bins;
        let kept = total.min(RING_ROWS as usize);
        let rows = &rows[(total - kept) * bins..];

        // At most two contiguous writes: up to the seam, then from row 0.
        let mut written = 0usize;
        while written < kept {
            let at = (self.wf_head as usize + written) % RING_ROWS as usize;
            let run = (kept - written).min(RING_ROWS as usize - at);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.wf.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: at as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&rows[written * bins..(written + run) * bins]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: w,
                    height: run as u32,
                    depth_or_array_layers: 1,
                },
            );
            written += run;
        }

        for (i, row) in rows.chunks_exact(bins).enumerate() {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in row {
                // The −200 dB gap filler is "no data" and must not drag the
                // auto-range floor down.
                if v.is_finite() && v > ROW_FLOOR_DBFS + 0.5 {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let at = (self.wf_head as usize + i) % RING_ROWS as usize;
            self.wf_row_range[at] = (lo, hi);
        }
        self.wf_head = (self.wf_head + kept as u32) % RING_ROWS;
        self.wf_written = self.wf_written.saturating_add(kept as u32);
    }

    /// The waterfall's colour-scale dB range for this frame (§7.6: shared
    /// with the histogram, or independent auto-range).
    fn waterfall_db_range(&self, params: &DisplayParams) -> (f32, f32) {
        if self.auto_range && self.wf_written > 0 {
            let rows = (self.wf_written.min(RING_ROWS)) as usize;
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            // While filling, rows 0..written hold data; once wrapped, all do.
            for &(rlo, rhi) in self.wf_row_range.iter().take(rows) {
                lo = lo.min(rlo);
                hi = hi.max(rhi);
            }
            if lo.is_finite() && hi.is_finite() {
                // Guard a degenerate span (single tone, single row).
                if hi - lo < 6.0 {
                    let mid = (hi + lo) * 0.5;
                    return (mid - 3.0, mid + 3.0);
                }
                return (lo, hi);
            }
        }
        (params.db_bottom, params.db_top)
    }

    /// Write both panels' uniforms for this frame. `None` rects mark a panel
    /// invisible (the FR-D4 split endpoints); an invisible panel draws
    /// nothing.
    pub fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        hist_rect: Option<GridPx>,
        wf_rect: Option<GridPx>,
        size_px: [u32; 2],
        params: &DisplayParams,
    ) {
        if !self.lut_uploaded {
            let atlas = colormap::lut_atlas_rgba8();
            queue.write_texture(
                self.lut.as_image_copy(),
                &atlas,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(colormap::LUT_SIZE as u32 * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: colormap::LUT_SIZE as u32,
                    height: Colormap::ALL.len() as u32,
                    depth_or_array_layers: 1,
                },
            );
            self.lut_uploaded = true;
        }
        let sel = [
            self.colormap.index() as f32,
            self.gamma,
            INTENSITY_FLOOR_EPSILON,
            self.wf_gamma,
        ];
        // Display-side zoom window (FR-D6, D-011): both surfaces reinterpret
        // their existing textures through the same window the axis labels
        // and the trace use.
        let (zoom_lo, zoom_w) = (params.zoom.lo() as f32, params.zoom.width() as f32);
        self.hist.visible = hist_rect.is_some();
        if let Some(rect) = hist_rect {
            let u = SurfaceUniforms {
                rect: ndc_rect(rect, size_px),
                sel,
                dims: [self.hist.w as f32, self.hist.h as f32, zoom_lo, zoom_w],
                range: [0.0; 4],
            };
            queue.write_buffer(&self.hist.uniforms, 0, bytemuck::bytes_of(&u));
        }
        self.wf.visible = wf_rect.is_some();
        if let Some(rect) = wf_rect {
            let visible_rows = (rect.max_y - rect.min_y)
                .round()
                .clamp(1.0, RING_ROWS as f32);
            let (lo, hi) = self.waterfall_db_range(params);
            let u = SurfaceUniforms {
                rect: ndc_rect(rect, size_px),
                sel,
                dims: [
                    self.wf.w as f32,
                    RING_ROWS as f32,
                    self.wf_head as f32,
                    visible_rows,
                ],
                range: [lo, hi, zoom_lo, zoom_w],
            };
            queue.write_buffer(&self.wf.uniforms, 0, bytemuck::bytes_of(&u));
        }
    }

    /// Draw the histogram surface, if visible and fed.
    pub fn draw_histogram(&self, rpass: &mut wgpu::RenderPass<'_>) {
        if self.hist.visible && self.hist.ready {
            rpass.set_pipeline(&self.hist_pipeline);
            rpass.set_bind_group(0, &self.hist.bind, &[]);
            rpass.draw(0..4, 0..1);
        }
    }

    /// Draw the waterfall surface, if visible and at least one row exists.
    pub fn draw_waterfall(&self, rpass: &mut wgpu::RenderPass<'_>) {
        if self.wf.visible && self.wf.ready {
            rpass.set_pipeline(&self.wf_pipeline);
            rpass.set_bind_group(0, &self.wf.bind, &[]);
            rpass.draw(0..4, 0..1);
        }
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Physical-pixel rect → the NDC `[x0, y_top, x1, y_bottom]` the vertex
/// shader expands into a quad.
fn ndc_rect(rect: GridPx, size_px: [u32; 2]) -> [f32; 4] {
    let (w, h) = (size_px[0] as f32, size_px[1] as f32);
    [
        rect.min_x / w * 2.0 - 1.0,
        1.0 - rect.min_y / h * 2.0,
        rect.max_x / w * 2.0 - 1.0,
        1.0 - rect.max_y / h * 2.0,
    ]
}

impl Panel {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        lut_view: &wgpu::TextureView,
        label: &str,
        w: u32,
        h: u32,
    ) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<SurfaceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex = create_data_texture(device, label, w, h);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
            ],
        });
        Self {
            uniforms,
            tex,
            bind,
            w,
            h,
            ready: false,
            visible: false,
        }
    }
}
