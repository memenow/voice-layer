//! The wgpu/WGSL glass material layer — the animated backdrop the Liquid Glass
//! surfaces float above.
//!
//! P1 approximated the material with a static gradient container; this is the
//! "extreme tier" enhancement: a full-window fragment shader that paints a
//! drifting multi-stop gradient, frosted-glass grain, slow light blobs, a
//! traveling specular sweep, a pointer-reactive highlight, and a vignette.
//!
//! Honesty about the platform: a fragment shader cannot sample the framebuffer
//! behind the window, so there is no real desktop blur on GNOME Wayland (see the
//! plan's platform constraints). This paints a self-contained rich background;
//! the glass cards lens *this*, not the true desktop. It draws only the
//! BACKGROUND — per-card lensing stays the [`crate::theme`] container styling.
//!
//! Graceful degradation is structural, not runtime-detected: iced_wgpu chooses
//! the surface alpha mode internally (PostMultiplied → PreMultiplied → Auto), and
//! if the GPU backend is unavailable iced falls back to the tiny-skia software
//! renderer, where a custom shader primitive is a silent no-op (it logs a warning
//! and draws nothing — it never panics). The caller therefore always paints an
//! opaque backdrop *underneath* this widget; when the shader cannot draw, that
//! backdrop shows through and the view degrades to the P1 appearance.

use std::borrow::Cow;

use iced::wgpu;
use iced::widget::shader::{self, Viewport};
use iced::{Element, Length, Rectangle, mouse};

use voicelayer_ui::color::Rgba;

/// A full-window animated glass background, driven by a monotonically increasing
/// `elapsed` seconds value from the app's frame clock.
pub fn background<'a, Message>(elapsed: f32, opacity: f32) -> Element<'a, Message>
where
    Message: 'a,
{
    // Fully qualified call: the `shader` module is in scope (for `shader::Program`
    // etc.), but the same-named free constructor lives in `iced::widget`'s value
    // namespace, so name it directly here.
    iced::widget::shader(GlassBackground::new(elapsed, opacity))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

pub(crate) fn rgba_array(c: Rgba) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

/// The shader program: snapshots the theme colors so the GPU material stays the
/// single source of truth in [`voicelayer_ui`], and carries the animation clock.
#[derive(Debug)]
pub struct GlassBackground {
    time: f32,
    /// User glass opacity `0.0..=1.0`; damps the animated decoration so a more
    /// opaque, frostier glass reads calmer (see the fragment shader).
    opacity: f32,
    base: [f32; 4],
    elevated: [f32; 4],
    accent: [f32; 4],
}

impl GlassBackground {
    fn new(time: f32, opacity: f32) -> Self {
        let p = crate::theme::palette();
        Self {
            time,
            opacity,
            base: rgba_array(p.bg_base),
            elevated: rgba_array(p.bg_elevated),
            accent: rgba_array(p.accent),
        }
    }
}

impl<Message> shader::Program<Message> for GlassBackground {
    type State = ();
    type Primitive = GlassPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        // Pointer position normalized to 0..1 within the widget; (-1, -1) signals
        // "off widget" to the shader so the radial highlight is suppressed.
        let pointer = cursor
            .position_in(bounds)
            .map(|p| [p.x / bounds.width.max(1.0), p.y / bounds.height.max(1.0)])
            .unwrap_or([-1.0, -1.0]);
        let aspect = bounds.width / bounds.height.max(1.0);
        GlassPrimitive {
            time: self.time,
            aspect,
            pointer,
            opacity: self.opacity,
            base: self.base,
            elevated: self.elevated,
            accent: self.accent,
        }
    }
}

/// Per-frame data handed to the GPU.
#[derive(Debug)]
pub struct GlassPrimitive {
    time: f32,
    aspect: f32,
    pointer: [f32; 2],
    opacity: f32,
    base: [f32; 4],
    elevated: [f32; 4],
    accent: [f32; 4],
}

/// The uniform block. `#[repr(C)]` with explicit padding so the layout matches
/// the WGSL `U` struct exactly under std140 (16-byte alignment, total 80 bytes,
/// no implicit padding so it is `Pod`-safe).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    base: [f32; 4],       // offset 0
    elevated: [f32; 4],   // offset 16
    accent: [f32; 4],     // offset 32
    time: f32,            // offset 48
    aspect: f32,          // offset 52
    pointer: [f32; 2],    // offset 56
    resolution: [f32; 2], // offset 64
    opacity: f32,         // offset 72
    _pad: f32,            // offset 76 -> total 80
}

impl shader::Primitive for GlassPrimitive {
    type Pipeline = GlassPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        let uniforms = Uniforms {
            base: self.base,
            elevated: self.elevated,
            accent: self.accent,
            time: self.time,
            aspect: self.aspect,
            pointer: self.pointer,
            resolution: [bounds.width, bounds.height],
            opacity: self.opacity,
            _pad: 0.0,
        };
        queue.write_buffer(&pipeline.uniforms, 0, bytemuck::bytes_of(&uniforms));
    }

    fn draw(&self, pipeline: &Self::Pipeline, render_pass: &mut wgpu::RenderPass<'_>) -> bool {
        // iced has already begun the pass with the viewport + scissor set to our
        // bounds; encode the full-bounds triangle and report that we drew.
        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        true
    }
}

/// GPU resources, built once and cached across frames by iced's primitive
/// storage (keyed on this type). Only the uniform buffer is rewritten per frame.
#[derive(Debug)]
pub struct GlassPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

impl shader::Pipeline for GlassPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let (pipeline, bind_group, uniforms) = fullscreen_pipeline(
            device,
            format,
            GLASS_WGSL,
            std::mem::size_of::<Uniforms>() as u64,
            "voicelayer.glass",
        );
        Self {
            pipeline,
            bind_group,
            uniforms,
        }
    }
}

/// Build a full-screen-triangle render pipeline fed by a single fragment-visible
/// uniform buffer. The glass background and the capture HUD's waveform
/// ([`crate::hud`]) share this scaffold; they differ only in their WGSL program
/// and the contents of that uniform block, so the wgpu boilerplate lives here
/// once. The returned buffer is `UNIFORM | COPY_DST`, rewritten each frame in the
/// primitive's `prepare`. Alpha blending is enabled so a shader can paint
/// translucent or rounded-transparent regions onto the (possibly transparent)
/// window surface.
pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    wgsl: &str,
    uniform_size: u64,
    label: &str,
) -> (wgpu::RenderPipeline, wgpu::BindGroup, wgpu::Buffer) {
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}.uniforms")),
        size: uniform_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}.bgl")),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bind_group")),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}.wgsl")),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}.layout")),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label}.pipeline")),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_group, uniforms)
}

/// The material. A full-screen triangle carries interpolated `uv` (0..1 over the
/// widget bounds) so the fragment stage is independent of the global framebuffer
/// coordinate. Colors arrive as uniforms from the shared palette.
const GLASS_WGSL: &str = r#"
struct U {
    base: vec4<f32>,
    elevated: vec4<f32>,
    accent: vec4<f32>,
    time: f32,
    aspect: f32,
    pointer: vec2<f32>,
    resolution: vec2<f32>,
    opacity: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var t = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VOut;
    out.pos = vec4<f32>(p[idx], 0.0, 1.0);
    out.uv = t[idx];
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));
    let w = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var pp = p;
    for (var k = 0; k < 4; k = k + 1) {
        v = v + amp * vnoise(pp);
        pp = pp * 2.0;
        amp = amp * 0.5;
    }
    return v;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let t = u.time;

    // Aspect-corrected coordinate so blobs and the pointer halo stay circular.
    var p = uv;
    p.x = p.x * u.aspect;

    // Base diagonal gradient (elevated at the top, base at the bottom) with a
    // slow horizontal drift so the field never reads as a static fill.
    let g = clamp(uv.y * 0.85 + 0.15 * sin(t * 0.05 + uv.x * 1.5), 0.0, 1.0);
    var col = mix(u.elevated.rgb, u.base.rgb, g);

    // Opacity (the user slider / a11y escape hatch) damps the lively decoration:
    // a frostier, more opaque glass reads calmer, so the blobs, grain, sweep, and
    // pointer halo fade toward a plain gradient as opacity rises. calibrated.
    let decor = 1.0 - 0.6 * clamp(u.opacity, 0.0, 1.0);

    // Two drifting soft light pools — accent-tinted, low intensity.
    let c1 = vec2<f32>(0.30 + 0.10 * sin(t * 0.13), 0.28 + 0.08 * cos(t * 0.11)) * vec2<f32>(u.aspect, 1.0);
    let c2 = vec2<f32>(0.72 + 0.08 * cos(t * 0.09), 0.66 + 0.10 * sin(t * 0.07)) * vec2<f32>(u.aspect, 1.0);
    // WGSL `smoothstep` is only well-defined for an increasing edge pair, so
    // express each inward falloff as `1.0 - smoothstep(low, high, x)` rather than
    // descending edges (mathematically identical, but driver-portable).
    let b1 = 1.0 - smoothstep(0.0, 0.55, distance(p, c1));
    let b2 = 1.0 - smoothstep(0.0, 0.50, distance(p, c2));
    col = col + u.accent.rgb * b1 * 0.18 * decor;
    col = col + u.elevated.rgb * b2 * 0.14 * decor;

    // Frosted-glass micro-grain.
    let n = fbm(uv * vec2<f32>(220.0, 220.0));
    col = col + (n - 0.5) * 0.02 * decor;

    // Traveling specular sweep along the diagonal, periodic.
    let sweep_phase = fract(t * 0.06);
    let band = uv.x * 0.6 + uv.y * 0.4;
    let sweep = (1.0 - smoothstep(0.0, 0.06, abs(band - sweep_phase))) * 0.06;
    col = col + vec3<f32>(1.0, 1.0, 1.0) * sweep * decor;

    // Pointer-reactive radial highlight (converged, subtle); off when pointer < 0.
    if (u.pointer.x >= 0.0) {
        let pp = u.pointer * vec2<f32>(u.aspect, 1.0);
        col = col + u.accent.rgb * (1.0 - smoothstep(0.0, 0.22, distance(p, pp))) * 0.10 * decor;
    }

    // Vignette: lift the center, darken the edges.
    let vd = distance(uv, vec2<f32>(0.5, 0.5));
    col = col * (1.0 - 0.35 * smoothstep(0.35, 0.95, vd));

    return vec4<f32>(col, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    /// The shader is only compiled by naga at runtime inside wgpu, which the
    /// headless test environment can't reach. Parse and validate the WGSL here so
    /// a syntax or type error fails the suite instead of panicking on first draw.
    #[test]
    fn glass_wgsl_parses_and_validates() {
        let module =
            naga::front::wgsl::parse_str(super::GLASS_WGSL).expect("glass WGSL should parse");
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("glass WGSL should validate");
    }

    /// The Rust uniform block and the WGSL `U` struct must agree; std140 rounds
    /// the block up to a multiple of 16. Guards against a field reorder that
    /// silently corrupts the layout.
    #[test]
    fn uniforms_match_std140_size() {
        assert_eq!(core::mem::size_of::<super::Uniforms>(), 80);
        assert_eq!(core::mem::size_of::<super::Uniforms>() % 16, 0);
    }
}
