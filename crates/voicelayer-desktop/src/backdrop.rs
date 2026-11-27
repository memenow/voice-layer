//! Animated wgpu/WGSL window backdrop and shared full-screen shader pipeline.
//!
//! Renders an in-window gradient with light pools, subtle grain, and a pointer
//! highlight. Under the software fallback renderer (tiny-skia), custom shader
//! primitives are no-ops, so callers paint an opaque [`crate::theme::backdrop`]
//! underneath.

use std::borrow::Cow;

use iced::wgpu;
use iced::widget::shader::{self, Viewport};
use iced::{Element, Length, Rectangle, mouse};

use voicelayer_ui::color::Rgba;

pub fn background<'a, Message>(elapsed: f32, opacity: f32) -> Element<'a, Message>
where
    Message: 'a,
{
    iced::widget::shader(GlassBackground::new(elapsed, opacity))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

pub(crate) fn rgba_array(c: Rgba) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

#[derive(Debug)]
pub struct GlassBackground {
    time: f32,
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

/// std140-aligned uniform block matching WGSL `struct U` (80 bytes, 16-byte aligned).
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
        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        true
    }
}

#[derive(Debug)]
pub struct GlassPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

impl shader::Pipeline for GlassPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let wgsl = glass_wgsl();
        let (pipeline, bind_group, uniforms) = fullscreen_pipeline(
            device,
            format,
            &wgsl,
            std::mem::size_of::<Uniforms>() as u64,
            "voicelayer.backdrop",
        );
        Self {
            pipeline,
            bind_group,
            uniforms,
        }
    }
}

/// Build a full-screen-triangle render pipeline with one fragment uniform buffer
/// and alpha blending enabled. Shared by the backdrop and capture HUD shaders.
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

/// Shared full-screen triangle vertex shader and hash helper used by both WGSL programs.
pub(crate) const WGSL_COMMON: &str = r#"
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
"#;

const GLASS_UNIFORMS_WGSL: &str = r#"
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
"#;

const GLASS_FRAGMENT_WGSL: &str = r#"
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

    var p = uv;
    p.x = p.x * u.aspect;

    let g = clamp(uv.y * 0.85 + 0.15 * sin(t * 0.05 + uv.x * 1.5), 0.0, 1.0);
    var col = mix(u.elevated.rgb, u.base.rgb, g);

    // Higher opacity dampens animated effects toward a simpler gradient.
    let decor = 1.0 - 0.6 * clamp(u.opacity, 0.0, 1.0);

    let c1 = vec2<f32>(0.30 + 0.10 * sin(t * 0.13), 0.28 + 0.08 * cos(t * 0.11)) * vec2<f32>(u.aspect, 1.0);
    let c2 = vec2<f32>(0.72 + 0.08 * cos(t * 0.09), 0.66 + 0.10 * sin(t * 0.07)) * vec2<f32>(u.aspect, 1.0);
    // Use increasing smoothstep bounds for WGSL portability.
    let b1 = 1.0 - smoothstep(0.0, 0.55, distance(p, c1));
    let b2 = 1.0 - smoothstep(0.0, 0.50, distance(p, c2));
    col = col + u.accent.rgb * b1 * 0.18 * decor;
    col = col + u.elevated.rgb * b2 * 0.14 * decor;

    let n = fbm(uv * vec2<f32>(220.0, 220.0));
    col = col + (n - 0.5) * 0.02 * decor;

    let sweep_phase = fract(t * 0.06);
    let band = uv.x * 0.6 + uv.y * 0.4;
    let sweep = (1.0 - smoothstep(0.0, 0.06, abs(band - sweep_phase))) * 0.06;
    col = col + vec3<f32>(1.0, 1.0, 1.0) * sweep * decor;

    if (u.pointer.x >= 0.0) {
        let pp = u.pointer * vec2<f32>(u.aspect, 1.0);
        col = col + u.accent.rgb * (1.0 - smoothstep(0.0, 0.22, distance(p, pp))) * 0.10 * decor;
    }

    let vd = distance(uv, vec2<f32>(0.5, 0.5));
    col = col * (1.0 - 0.35 * smoothstep(0.35, 0.95, vd));

    return vec4<f32>(col, 1.0);
}
"#;

fn glass_wgsl() -> String {
    format!("{GLASS_UNIFORMS_WGSL}{WGSL_COMMON}{GLASS_FRAGMENT_WGSL}")
}

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn glass_wgsl_parses_and_validates() {
        let source = super::glass_wgsl();
        let module = naga::front::wgsl::parse_str(&source).expect("glass WGSL should parse");
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("glass WGSL should validate");
    }

    #[test]
    fn uniforms_match_std140_size() {
        assert_eq!(core::mem::size_of::<super::Uniforms>(), 80);
        assert_eq!(core::mem::size_of::<super::Uniforms>() % 16, 0);
    }
}
