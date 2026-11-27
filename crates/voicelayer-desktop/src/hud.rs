//! Floating capture HUD overlay shown while microphone capture is active.
//!
//! Opens as a borderless, transparent secondary window. On GNOME Wayland, window
//! placement and always-on-top hints are controlled by the compositor. The
//! waveform animation reflects capture state energy (not live microphone levels).

use iced::widget::shader::{self, Viewport};
use iced::widget::{Space, column, container, row, stack, text};
use iced::{Element, Length, Rectangle, alignment, mouse, wgpu};

use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::backdrop::{WGSL_COMMON, fullscreen_pipeline, rgba_array};
use crate::components::{self, Capsule};
use crate::state::SessionStage;
use crate::theme;

struct HudStatus {
    label: String,
    live: bool,
    energy: f32,
    can_stop: bool,
}

fn status_of(app: &App) -> HudStatus {
    if app.capture_in_flight {
        return HudStatus {
            label: format!("Capturing · {} s", app.preferences.capture_seconds),
            live: true,
            energy: 0.7,
            can_stop: false,
        };
    }
    match app.session.stage {
        SessionStage::Listening => HudStatus {
            label: "Listening, speak now".to_owned(),
            live: true,
            energy: 1.0,
            can_stop: true,
        },
        SessionStage::Starting => HudStatus {
            label: "Starting…".to_owned(),
            live: false,
            energy: 0.5,
            can_stop: false,
        },
        SessionStage::Stopping => HudStatus {
            label: "Stopping…".to_owned(),
            live: false,
            energy: 0.4,
            can_stop: false,
        },
        SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => HudStatus {
            label: "Idle".to_owned(),
            live: false,
            energy: 0.25,
            can_stop: false,
        },
    }
}

pub(crate) fn view(app: &App) -> Element<'_, Message> {
    let p = theme::palette();
    let status = status_of(app);

    let dot_color = if status.live {
        p.accent
    } else {
        p.text_secondary
    };
    let label_color = if status.live {
        p.text_primary
    } else {
        p.text_secondary
    };

    let header = row![
        text("●")
            .size(tokens::text::BODY)
            .color(theme::color(dot_color)),
        text(status.label)
            .font(theme::font(Weight::Semibold))
            .size(tokens::text::TITLE)
            .color(theme::color(label_color)),
    ]
    .spacing(tokens::space::SM)
    .align_y(alignment::Vertical::Center);

    let action: Element<'_, Message> = if status.can_stop {
        row![
            Space::new().width(Length::Fill),
            components::capsule("Stop · F9", Capsule::Primary).on_press(Message::StopPressed),
        ]
        .into()
    } else {
        Space::new().height(Length::Fixed(0.0)).into()
    };

    let body = column![header, Space::new().height(Length::Fill), action]
        .spacing(tokens::space::SM)
        .width(Length::Fill)
        .height(Length::Fill);

    // Layer order: fallback card (for software rendering), waveform shader, overlay controls.
    stack![
        container(Space::new().width(Length::Fill).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style({
                let a11y = app.accessibility();
                move |_theme| theme::glass_card(&a11y)
            }),
        waveform(app.elapsed, status.energy),
        container(body)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(theme::pad(tokens::space::MD, tokens::space::LG)),
    ]
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn waveform<'a, Message: 'a>(elapsed: f32, energy: f32) -> Element<'a, Message> {
    iced::widget::shader(Waveform::new(elapsed, energy))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

#[derive(Debug)]
struct Waveform {
    time: f32,
    energy: f32,
    base: [f32; 4],
    elevated: [f32; 4],
    accent: [f32; 4],
    corner_radius: f32,
}

impl Waveform {
    fn new(time: f32, energy: f32) -> Self {
        let p = theme::palette();
        Self {
            time,
            energy,
            base: rgba_array(p.bg_base),
            elevated: rgba_array(p.bg_elevated),
            accent: rgba_array(p.accent),
            corner_radius: tokens::radius::CARD,
        }
    }
}

impl<Message> shader::Program<Message> for Waveform {
    type State = ();
    type Primitive = WaveformPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        WaveformPrimitive {
            time: self.time,
            energy: self.energy,
            base: self.base,
            elevated: self.elevated,
            accent: self.accent,
            corner_radius: self.corner_radius,
        }
    }
}

#[derive(Debug)]
struct WaveformPrimitive {
    time: f32,
    energy: f32,
    base: [f32; 4],
    elevated: [f32; 4],
    accent: [f32; 4],
    corner_radius: f32,
}

/// std140-aligned uniform block matching WGSL `struct U` (80 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveUniforms {
    base: [f32; 4],       // offset 0
    elevated: [f32; 4],   // offset 16
    accent: [f32; 4],     // offset 32
    time: f32,            // offset 48
    energy: f32,          // offset 52
    corner_radius: f32,   // offset 56
    _pad0: f32,           // offset 60
    resolution: [f32; 2], // offset 64
    _pad1: [f32; 2],      // offset 72 -> total 80
}

impl shader::Primitive for WaveformPrimitive {
    type Pipeline = WaveformPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        let uniforms = WaveUniforms {
            base: self.base,
            elevated: self.elevated,
            accent: self.accent,
            time: self.time,
            energy: self.energy,
            corner_radius: self.corner_radius,
            _pad0: 0.0,
            resolution: [bounds.width, bounds.height],
            _pad1: [0.0, 0.0],
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
struct WaveformPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

impl shader::Pipeline for WaveformPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let wgsl = waveform_wgsl();
        let (pipeline, bind_group, uniforms) = fullscreen_pipeline(
            device,
            format,
            &wgsl,
            std::mem::size_of::<WaveUniforms>() as u64,
            "voicelayer.hud",
        );
        Self {
            pipeline,
            bind_group,
            uniforms,
        }
    }
}

const WAVEFORM_UNIFORMS_WGSL: &str = r#"
struct U {
    base: vec4<f32>,
    elevated: vec4<f32>,
    accent: vec4<f32>,
    time: f32,
    energy: f32,
    corner_radius: f32,
    pad0: f32,
    resolution: vec2<f32>,
    pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
"#;

const WAVEFORM_FRAGMENT_WGSL: &str = r#"
fn sd_round_rect(q: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let d = abs(q) - b + vec2<f32>(r, r);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0, 0.0))) - r;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let res = u.resolution;
    let t = u.time;

    let pix = uv * res;
    let center = res * 0.5;
    let margin = 2.0;
    let half = center - vec2<f32>(margin, margin);
    let radius = min(u.corner_radius, min(half.x, half.y));
    let d = sd_round_rect(pix - center, half, radius);

    let mask = clamp(0.5 - d / 1.5, 0.0, 1.0);
    if (mask <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var col = mix(u.elevated.rgb, u.base.rgb, clamp(uv.y, 0.0, 1.0));

    // Synthetic activity waveform driven by capture state energy.
    let mid = 0.5;
    let amp = 0.06 + 0.20 * u.energy;
    let wave = sin(uv.x * 18.0 - t * 3.0) * 0.6
             + sin(uv.x * 7.0 + t * 1.7) * 0.3
             + sin(uv.x * 33.0 - t * 5.0) * 0.1;
    let env = smoothstep(0.0, 0.18, uv.x) * (1.0 - smoothstep(0.82, 1.0, uv.x));
    let h = amp * wave * env;

    let dist = abs((uv.y - mid) - h);
    let line = 1.0 - smoothstep(0.0, 0.035, dist);
    let glow = (1.0 - smoothstep(0.0, 0.18, dist)) * 0.35;
    col = col + u.accent.rgb * (line * 0.9 + glow) * (0.4 + 0.6 * u.energy);

    let dist2 = abs((uv.y - mid) + h);
    col = col + u.accent.rgb * (1.0 - smoothstep(0.0, 0.05, dist2)) * 0.15;

    let n = hash(floor(pix * 0.5));
    col = col + (n - 0.5) * 0.015;

    let rim = (1.0 - smoothstep(0.0, 2.5, abs(d))) * 0.10;
    col = col + vec3<f32>(1.0, 1.0, 1.0) * rim;

    return vec4<f32>(col, mask);
}
"#;

fn waveform_wgsl() -> String {
    format!("{WAVEFORM_UNIFORMS_WGSL}{WGSL_COMMON}{WAVEFORM_FRAGMENT_WGSL}")
}

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn waveform_wgsl_parses_and_validates() {
        let source = super::waveform_wgsl();
        let module = naga::front::wgsl::parse_str(&source).expect("waveform WGSL should parse");
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("waveform WGSL should validate");
    }

    #[test]
    fn wave_uniforms_match_std140_size() {
        assert_eq!(core::mem::size_of::<super::WaveUniforms>(), 80);
        assert_eq!(core::mem::size_of::<super::WaveUniforms>() % 16, 0);
    }
}
