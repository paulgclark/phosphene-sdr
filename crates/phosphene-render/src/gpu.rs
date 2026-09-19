// SPDX-License-Identifier: MIT

//! The wgpu side: one colored-vertex alpha-blended pipeline for the data
//! surface, plus the frame composer that layers the egui chrome on top and
//! renders into any target view — the window surface and the headless
//! offscreen texture share this path, so what CI's golden PNG sees is what a
//! user sees.

use std::mem;

use egui::TexturesDelta;
use egui_wgpu::ScreenDescriptor;

use crate::colormap::Colormap;
use crate::keymap::{ChromeRequests, GAMMA_STEP};
use crate::layout::{Layout, ViewState};
use crate::scene::{self, GridPx, Vertex};
use crate::surface::SurfaceRenderer;
use crate::theme::Theme;

/// Renders the data-surface geometry ([`Vertex`] triangle lists).
pub struct SceneRenderer {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    capacity: u64,
    count: u32,
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phosphene-scene-vertices"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl SceneRenderer {
    /// Build the pipeline for a given target format.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phosphene-scene-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("phosphene-scene-pipeline-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("phosphene-scene-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let capacity = 64 * 1024;
        Self {
            pipeline,
            vbuf: create_vertex_buffer(device, capacity),
            capacity,
            count: 0,
        }
    }

    /// Upload this frame's vertices, growing the buffer if needed.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[Vertex]) {
        let bytes: &[u8] = bytemuck::cast_slice(verts);
        if (bytes.len() as u64) > self.capacity {
            self.capacity = (bytes.len() as u64).next_power_of_two();
            self.vbuf = create_vertex_buffer(device, self.capacity);
        }
        queue.write_buffer(&self.vbuf, 0, bytes);
        self.count = verts.len() as u32;
    }

    /// Record the draw into an open render pass.
    pub fn draw(&self, rpass: &mut wgpu::RenderPass<'_>) {
        self.draw_range(rpass, 0..self.count);
    }

    /// Draw a sub-range of the prepared vertices — lets one prepared buffer
    /// contribute geometry both under and over the textured surfaces.
    pub fn draw_range(&self, rpass: &mut wgpu::RenderPass<'_>, range: std::ops::Range<u32>) {
        let end = range.end.min(self.count);
        if range.start >= end {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_vertex_buffer(0, self.vbuf.slice(..));
        rpass.draw(range.start..end, 0..1);
    }
}

/// Tessellated egui output for one frame, produced by the caller (windowed
/// and headless paths tessellate identically).
pub struct EguiFrame {
    /// `Context::tessellate` output.
    pub primitives: Vec<egui::ClippedPrimitive>,
    /// Texture changes to apply this frame.
    pub textures_delta: TexturesDelta,
    /// Pixels per point the shapes were tessellated at.
    pub pixels_per_point: f32,
}

/// Everything needed to draw the full RTSA data surface for one frame:
/// persistence histogram (FR-D1), waterfall (FR-D4), and the live (FR-D2)
/// and max-hold (FR-D3) trace overlays. Consumed by
/// [`FrameComposer::render_rtsa_frame`].
pub struct RtsaInput<'a> {
    /// DC-centered dBFS/bin spectrum for the live trace overlay (FR-D2).
    /// Empty skips the trace.
    pub live_bins: &'a [f32],
    /// DC-centered dBFS/bin max-hold trace to overlay (FR-D3), same
    /// convention as `live_bins`: **empty means "not shown"** — a caller
    /// with no max hold (hidden, or none accumulated) is still correct.
    /// Drawn under the live trace in the subordinate
    /// [`crate::scene::TraceStyle::max_hold`] style (D-010).
    pub max_hold_bins: &'a [f32],
    /// Persistence intensity grid: `bins × levels` values in `[0,1]`,
    /// row-major by bin — exactly what `PersistenceHistogram::intensity()`
    /// yields. Empty skips the histogram surface.
    pub intensity: &'a [f32],
    /// Frequency-bin count of `intensity`.
    pub intensity_bins: usize,
    /// Power-level count of `intensity` (level 0 = bottom of the dB range).
    pub intensity_levels: usize,
    /// The full data-surface rect from [`crate::layout::Layout`], in egui
    /// points; the histogram/waterfall regions are split off it internally
    /// by the FR-D4 ratio in `view.split`.
    pub surface_points: egui::Rect,
    /// The view state (axis parameters, zoom, split, pause) — the same
    /// struct the chrome read this frame, so geometry cannot diverge from
    /// the labels.
    pub view: &'a ViewState,
    /// The palette.
    pub theme: &'a Theme,
}

/// Composes one frame: clear → grid + trace → egui chrome, into any view.
pub struct FrameComposer {
    scene: SceneRenderer,
    surface: SurfaceRenderer,
    egui: egui_wgpu::Renderer,
    verts: Vec<Vertex>,
    /// Reused trace-geometry scratch (§7.7, D-035): both trace overlays
    /// build through it sequentially, so the per-frame path allocates
    /// nothing once warm.
    trace_scratch: scene::TraceScratch,
    /// Retained copy of the last unpaused spectrum, so the FR-D12 freeze
    /// keeps rendering the *held data* through the live geometry — labels
    /// and scale changes stay truthful of what is on screen.
    held_bins: Vec<f32>,
    /// Retained copy of the last unpaused max-hold trace (FR-D3) — the
    /// FR-D12 freeze holds it exactly as it holds `held_bins`. An empty
    /// `max_hold_bins` input clears it even while paused: hiding the trace
    /// is a view command, not new data.
    held_max_bins: Vec<f32>,
    paused: bool,
    /// The cadence of the retained waterfall rows, as stated by the
    /// producer at the row-push seam ([`Self::push_waterfall_rows`]) — its
    /// interval AND its time base (seconds, or spectra for a rate-less
    /// source, D-054). The FR-D5 labels read it back via
    /// [`Self::waterfall_row_interval`] / [`Self::waterfall_row_spectra`],
    /// so they state the cadence the rows actually carried (D-031) — never
    /// a value someone typed elsewhere.
    wf_cadence: Option<crate::waterfall::RowCadence>,
}

impl FrameComposer {
    /// `format` must match the target view the composer will render into.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            scene: SceneRenderer::new(device, format),
            surface: SurfaceRenderer::new(device, format),
            egui: egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default()),
            verts: Vec::new(),
            trace_scratch: scene::TraceScratch::default(),
            held_bins: Vec::new(),
            held_max_bins: Vec::new(),
            paused: false,
            wf_cadence: None,
        }
    }

    /// Apply the chrome's per-frame requests (from
    /// [`crate::chrome::ChromeResponse`]): the colormap cycle goes through
    /// [`cycle_colormap`](Self::cycle_colormap) (the D-009 no-rebuild seam)
    /// and persistence steps through [`set_gamma`](Self::set_gamma).
    pub fn apply_requests(&mut self, r: &ChromeRequests) {
        if r.cycle_colormap {
            self.cycle_colormap();
        }
        if r.gamma_steps != 0 {
            self.set_gamma(self.gamma() * GAMMA_STEP.powi(r.gamma_steps));
        }
    }

    /// Track the FR-D12 pause state for one retained trace buffer: refresh
    /// it from `incoming` unless the display is paused with data already
    /// held.
    fn hold(held: &mut Vec<f32>, incoming: &[f32], paused: bool) {
        if !paused || held.is_empty() {
            held.clear();
            held.extend_from_slice(incoming);
        }
    }

    /// Render one full RTSA frame — persistence histogram (FR-D1),
    /// waterfall (FR-D4), grid, live trace, egui chrome — into `view` and
    /// submit.
    ///
    /// Draw order: grid and panel frames first, then the two blended
    /// surfaces (histogram cells below the ε floor are transparent, so the
    /// grid shows through where nothing accumulated — the floor renders as
    /// background, §7.3), then the max-hold trace (FR-D3, subordinate),
    /// then the live trace on top, then chrome.
    pub fn render_rtsa_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        size_px: [u32; 2],
        input: RtsaInput<'_>,
        mut egui_frame: EguiFrame,
    ) {
        let ppp = egui_frame.pixels_per_point;
        let view_state = input.view;
        self.paused = view_state.paused;
        let regions = Layout {
            grid: input.surface_points,
        }
        .split(view_state.split);
        let hist_px = GridPx::from_points(regions.histogram, ppp);
        let wf_px = GridPx::from_points(regions.waterfall, ppp);
        let visible = |r: &GridPx| (r.max_x - r.min_x) >= 1.0 && (r.max_y - r.min_y) >= 1.0;
        let hist_visible = visible(&hist_px);
        let wf_visible = visible(&wf_px);

        // FR-D12 freeze: while paused, the retained textures and the held
        // trace keep rendering; new data is not taken (the source keeps
        // draining upstream).
        if !self.paused {
            self.surface.upload_intensity(
                device,
                queue,
                input.intensity,
                input.intensity_bins,
                input.intensity_levels,
            );
        }
        Self::hold(&mut self.held_bins, input.live_bins, self.paused);
        if input.max_hold_bins.is_empty() {
            // Empty means "not shown" — an explicit hide, honoured even
            // while the FR-D12 freeze holds the data itself.
            self.held_max_bins.clear();
        } else {
            Self::hold(&mut self.held_max_bins, input.max_hold_bins, self.paused);
        }
        self.surface.prepare_frame(
            queue,
            hist_visible.then_some(hist_px),
            wf_visible.then_some(wf_px),
            size_px,
            &view_state.params,
        );

        self.verts.clear();
        if hist_visible {
            scene::build_grid(
                &mut self.verts,
                hist_px,
                size_px,
                &view_state.params,
                input.theme,
            );
        }
        if wf_visible {
            scene::build_frame(&mut self.verts, wf_px, size_px, input.theme);
        }
        let under = self.verts.len() as u32;
        // FR-D3: the max-hold overlay draws before (under) the live trace —
        // subordinate by style and by stacking (D-010).
        if hist_visible && !self.held_max_bins.is_empty() {
            scene::build_trace_styled(
                &mut self.verts,
                &self.held_max_bins,
                hist_px,
                size_px,
                &view_state.params,
                &scene::TraceStyle::max_hold(input.theme),
                &mut self.trace_scratch,
            );
        }
        if hist_visible && !self.held_bins.is_empty() {
            scene::build_trace(
                &mut self.verts,
                &self.held_bins,
                hist_px,
                size_px,
                &view_state.params,
                input.theme,
                &mut self.trace_scratch,
            );
        }
        self.scene.prepare(device, queue, &self.verts);

        let screen = ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point: ppp,
        };
        for (id, deltas) in &egui_frame.textures_delta.set {
            for delta in deltas {
                self.egui.update_texture(device, queue, *id, delta);
            }
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("phosphene-rtsa-frame"),
        });
        let user_cmds =
            self.egui
                .update_buffers(device, queue, &mut encoder, &egui_frame.primitives, &screen);
        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("phosphene-rtsa-frame-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(input.theme.clear_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.scene.draw_range(&mut rpass, 0..under);
            self.surface.draw_histogram(&mut rpass);
            self.surface.draw_waterfall(&mut rpass);
            self.scene
                .draw_range(&mut rpass, under..self.verts.len() as u32);
            self.egui
                .render(&mut rpass, &egui_frame.primitives, &screen);
        }
        queue.submit(user_cmds.into_iter().chain([encoder.finish()]));
        for id in &egui_frame.textures_delta.free {
            self.egui.free_texture(id);
        }
        // Every delta has been applied above; epaint panics on drop otherwise.
        egui_frame.textures_delta.clear();
    }

    /// The active colormap (FR-D8).
    pub fn colormap(&self) -> Colormap {
        self.surface.colormap()
    }

    /// Select the active colormap. Live: takes effect next frame with no
    /// pipeline rebuild and no stall (D-009) — see [`crate::colormap`] for
    /// how the LUT layout guarantees that.
    pub fn set_colormap(&mut self, map: Colormap) {
        self.surface.set_colormap(map);
    }

    /// Cycle to the next colormap and return it — the seam M1-C's keyboard
    /// map binds (D-009).
    pub fn cycle_colormap(&mut self) -> Colormap {
        self.surface.cycle_colormap()
    }

    /// The `gamma_adjust` exponent (§7.3 intensity/gamma control).
    pub fn gamma(&self) -> f32 {
        self.surface.gamma()
    }

    /// Set the `gamma_adjust` exponent — a live control (§7.3).
    pub fn set_gamma(&mut self, gamma: f32) {
        self.surface.set_gamma(gamma);
    }

    /// Whether the waterfall auto-ranges its colour scale (§7.6 toggle).
    pub fn waterfall_auto_range(&self) -> bool {
        self.surface.waterfall_auto_range()
    }

    /// Toggle the waterfall's independent auto-range (§7.6): on, the colour
    /// scale spans the retained rows; off, it shares the histogram's dB
    /// range.
    pub fn set_waterfall_auto_range(&mut self, on: bool) {
        self.surface.set_waterfall_auto_range(on);
    }

    /// The §7.6 waterfall intensity exponent `γ_wf` (D-065 §3).
    pub fn waterfall_gamma(&self) -> f32 {
        self.surface.waterfall_gamma()
    }

    /// Set the §7.6 waterfall intensity exponent — a live control, mirrored
    /// from [`crate::layout::ViewState::wf_gamma`] every frame exactly as
    /// the auto-range toggle above is.
    pub fn set_waterfall_gamma(&mut self, gamma: f32) {
        self.surface.set_waterfall_gamma(gamma);
    }

    /// Append one aggregated waterfall row (dBFS per bin), stating the
    /// seconds of signal time it represents — the single-row, seconds-only
    /// convenience over [`Self::push_waterfall_rows`]; `None` states no
    /// time base at all.
    pub fn push_waterfall_row(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        row: &[f32],
        row_interval_s: Option<f32>,
    ) {
        let cadence = row_interval_s
            .filter(|s| s.is_finite() && *s > 0.0)
            .map(crate::waterfall::RowCadence::seconds_per_row);
        self.push_waterfall_rows(device, queue, row, row.len(), cadence);
    }

    /// Append one frame's **batch** of aggregated waterfall rows (flat
    /// `rows × bins`, oldest first), stating the cadence they were
    /// aggregated at — interval **and** time base, so a rate-less source's
    /// spectra cadence (D-054) travels with its rows and the FR-D5 labels
    /// can state the axis in the one honest unit; a seconds figure is never
    /// invented. Batching is the D-047 shape: the rows land in at most two
    /// texture writes, never one GPU write per row. Rows are discarded
    /// while the display is paused (FR-D12) — the frozen picture stays
    /// frozen.
    pub fn push_waterfall_rows(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rows: &[f32],
        bins: usize,
        cadence: Option<crate::waterfall::RowCadence>,
    ) {
        if self.paused || rows.is_empty() {
            return;
        }
        self.wf_cadence = cadence;
        self.surface.push_waterfall_rows(device, queue, rows, bins);
    }

    /// Seconds of signal time per waterfall row, as the producer stated it
    /// with the most recent row push. The production frame loop mirrors
    /// this into [`crate::layout::DisplayParams::wf_row_interval_s`] each
    /// frame, so the FR-D5 time labels state the cadence the rows actually
    /// carried. `None` until rows arrive, and `None` for a rate-less
    /// source — see [`Self::waterfall_row_spectra`].
    pub fn waterfall_row_interval(&self) -> Option<f32> {
        self.wf_cadence.and_then(|c| c.row_interval_s())
    }

    /// Spectra per waterfall row — the rate-less mirror of
    /// [`Self::waterfall_row_interval`] (D-054), mirrored into
    /// [`crate::layout::DisplayParams::wf_row_spectra`] so the FR-D5 axis
    /// labels the waterfall in rows of spectra. At most one of the two
    /// accessors answers `Some`: a cadence has exactly one time base.
    pub fn waterfall_row_spectra(&self) -> Option<f32> {
        self.wf_cadence.and_then(|c| c.spectra_per_row())
    }

    /// Number of surface render pipelines built since construction. The
    /// D-009 seal instrument: it must not grow when the colormap switches.
    pub fn surface_pipeline_builds(&self) -> u32 {
        self.surface.pipeline_builds()
    }

    /// Total waterfall rows written into the ring since construction. The
    /// D-032 seal instrument: driving the real windowed/headless frame loop
    /// must make this grow — a waterfall no production code feeds is a dead
    /// feature, however green its unit tests.
    pub fn waterfall_rows_written(&self) -> u32 {
        self.surface.waterfall_rows_written()
    }
}

/// Format used for the headless render target.
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// An offscreen render target with RGBA readback — the headless path
/// (FR-C4 subset) that M0-D's CI golden consumes, and the FR-D12 screenshot
/// path, which renders in the window's own surface format.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    /// View to render into.
    pub view: wgpu::TextureView,
    /// Size in physical pixels.
    pub size: [u32; 2],
    format: wgpu::TextureFormat,
}

impl OffscreenTarget {
    /// Create a `width` × `height` target in [`OFFSCREEN_FORMAT`].
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self::with_format(device, width, height, OFFSCREEN_FORMAT)
    }

    /// Create a target in an explicit 8-bit color format — the FR-D12
    /// screenshot renders in the same format as the window surface so the
    /// composer's pipelines (built for that format) can draw into it.
    /// BGRA formats are swizzled to RGBA on readback.
    pub fn with_format(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("phosphene-offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size: [width, height],
            format,
        }
    }

    /// Read the rendered image back as tightly-packed RGBA8 (sRGB-encoded)
    /// rows — BGRA-format targets are swizzled so the result is always
    /// RGBA byte order. Blocks until the GPU work completes.
    pub fn read_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let [width, height] = self.size;
        let unpadded = width as usize * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phosphene-readback"),
            size: (padded * height as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("phosphene-readback-encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while waiting for readback");
        rx.recv()
            .expect("map_async callback dropped")
            .expect("readback buffer mapping failed");

        let data = slice
            .get_mapped_range()
            .expect("readback buffer range not mapped");
        let mut out = Vec::with_capacity(unpadded * height as usize);
        for row in 0..height as usize {
            let start = row * padded;
            out.extend_from_slice(&data[start..start + unpadded]);
        }
        drop(data);
        buffer.unmap();
        if matches!(
            self.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in out.as_chunks_mut::<4>().0 {
                px.swap(0, 2);
            }
        }
        out
    }
}
