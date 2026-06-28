//! The capture HUD overlay — a small floating glass pill that appears while a
//! dictation capture holds the microphone, and disappears when it ends.
//!
//! It is a second daemon window (see [`crate::app`]): borderless, transparent,
//! and best-effort always-on-top. Honesty about GNOME Wayland (see the plan's
//! platform constraints): a client cannot position itself (`window::move_to` is
//! unsupported) and `always-on-top` is silently ignored, so the compositor owns
//! placement — `Position::Centered` is only a hint. The layer-shell path that
//! would fix this on wlroots/KWin is left as a reserved interface, not built.
//!
//! The pill is drawn by a full-window wgpu/WGSL shader: a rounded-rectangle glass
//! body (alpha 0 outside the rounded region, so the window corners stay
//! see-through for the floating look) with an animated waveform across it. That
//! waveform is **synthetic** — the desktop client has no microphone amplitude
//! (the `/v1` event stream carries no levels), so it is a time-driven "listening"
//! animation whose height tracks the capture state's energy, not real audio.
//!
//! Degradation mirrors [`crate::glass`]: under the tiny-skia software renderer a
//! custom shader primitive is a silent no-op, so a translucent glass-card
//! container sits beneath the shader and shows the same pill shape when the GPU
//! material cannot draw.

use iced::widget::shader::{self, Viewport};
use iced::widget::{Space, column, container, row, stack, text};
use iced::{Element, Length, Rectangle, alignment, mouse, wgpu};

use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::components::{self, Capsule};
use crate::glass::{fullscreen_pipeline, rgba_array};
use crate::state::SessionStage;
use crate::theme;

/// What the HUD is currently reflecting, derived from the live capture state.
struct HudStatus {
    label: String,
    /// Accent vs. muted dot/label color (Listening is the only "live" state).
    live: bool,
    /// Waveform amplitude driver in `0.0..=1.0` — state energy, not audio level.
    energy: f32,
    /// Whether a stop action is offered. A one-shot fixed-window capture cannot
    /// be interrupted mid-recording, so it shows progress without a stop control.
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
            label: "Listening — speak now".to_owned(),
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
        // The HUD is only open while capture is active; render a calm line rather
        // than panic if a close race leaves it briefly visible.
        SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => HudStatus {
            label: "Idle".to_owned(),
            live: false,
            energy: 0.25,
            can_stop: false,
        },
    }
}

/// The capture HUD window's content: the waveform glass pill with a status line
/// and a quick stop control floating over it.
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

    // Bottom to top: the no-GPU fallback pill, the animated waveform shader, then
    // the status/stop overlay. Matches the main view's structural degradation —
    // if the shader is a no-op (tiny-skia), the glass-card pill still shows.
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

/// The animated waveform pill as a shader widget filling the HUD window.
fn waveform<'a, Message: 'a>(elapsed: f32, energy: f32) -> Element<'a, Message> {
    iced::widget::shader(Waveform::new(elapsed, energy))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

/// The shader program: snapshots the palette so the GPU pill stays sourced from
/// [`voicelayer_ui`], and carries the animation clock and state energy.
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

/// Per-frame data handed to the GPU.
#[derive(Debug)]
struct WaveformPrimitive {
    time: f32,
    energy: f32,
    base: [f32; 4],
    elevated: [f32; 4],
    accent: [f32; 4],
    corner_radius: f32,
}

/// The uniform block. `#[repr(C)]` with explicit padding so the layout matches
/// the WGSL `U` struct under std140 (16-byte alignment, total 80 bytes, no
/// implicit padding so it is `Pod`-safe).
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

/// GPU resources, built once and cached across frames by iced's primitive
/// storage (keyed on this type, distinct from [`crate::glass`]'s pipeline). Only
/// the uniform buffer is rewritten per frame.
#[derive(Debug)]
struct WaveformPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

impl shader::Pipeline for WaveformPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let (pipeline, bind_group, uniforms) = fullscreen_pipeline(
            device,
            format,
            WAVEFORM_WGSL,
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

/// The HUD material. A full-screen triangle carries interpolated `uv` (0..1 over
/// the widget bounds). The fragment stage cuts a rounded-rectangle glass pill
/// (transparent outside) and paints the synthetic waveform across it.
const WAVEFORM_WGSL: &str = r#"
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

// Signed distance to a rounded rectangle: q relative to center, b half-size, r radius.
fn sd_round_rect(q: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let d = abs(q) - b + vec2<f32>(r, r);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0, 0.0))) - r;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let res = u.resolution;
    let t = u.time;

    // Pixel space keeps the rounded corners crisp regardless of the pill aspect.
    let pix = uv * res;
    let center = res * 0.5;
    let margin = 2.0;
    let half = center - vec2<f32>(margin, margin);
    let radius = min(u.corner_radius, min(half.x, half.y));
    let d = sd_round_rect(pix - center, half, radius);

    // Opaque inside, ~1.5px anti-aliased edge, transparent outside so the window
    // corners stay see-through. The body is alpha 1.0, which makes the result
    // independent of the surface's premultiplied/straight alpha mode.
    let mask = clamp(0.5 - d / 1.5, 0.0, 1.0);
    if (mask <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Glass body: a vertical gradient, elevated at the top fading to base.
    var col = mix(u.elevated.rgb, u.base.rgb, clamp(uv.y, 0.0, 1.0));

    // Synthetic listening waveform (NOT real audio): a sum of sines drifting in
    // time, tapered at the horizontal ends, amplitude scaled by the state energy.
    let mid = 0.5;
    let amp = 0.06 + 0.20 * u.energy;
    let wave = sin(uv.x * 18.0 - t * 3.0) * 0.6
             + sin(uv.x * 7.0 + t * 1.7) * 0.3
             + sin(uv.x * 33.0 - t * 5.0) * 0.1;
    let env = smoothstep(0.0, 0.18, uv.x) * smoothstep(1.0, 0.82, uv.x);
    let h = amp * wave * env;

    let dist = abs((uv.y - mid) - h);
    let line = smoothstep(0.035, 0.0, dist);
    let glow = smoothstep(0.18, 0.0, dist) * 0.35;
    col = col + u.accent.rgb * (line * 0.9 + glow) * (0.4 + 0.6 * u.energy);

    // A faint mirrored ghost for a fuller "audio" feel.
    let dist2 = abs((uv.y - mid) + h);
    col = col + u.accent.rgb * smoothstep(0.05, 0.0, dist2) * 0.15;

    // Micro-grain so the body is not a flat fill.
    let n = hash(floor(pix * 0.5));
    col = col + (n - 0.5) * 0.015;

    // Specular rim just inside the rounded edge.
    let rim = smoothstep(2.5, 0.0, abs(d)) * 0.10;
    col = col + vec3<f32>(1.0, 1.0, 1.0) * rim;

    return vec4<f32>(col, mask);
}
"#;

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    /// wgpu compiles the WGSL with naga at runtime, which the headless test
    /// environment can't reach. Parse and validate it here so a syntax or type
    /// error fails the suite instead of panicking on the HUD's first draw.
    #[test]
    fn waveform_wgsl_parses_and_validates() {
        let module =
            naga::front::wgsl::parse_str(super::WAVEFORM_WGSL).expect("waveform WGSL should parse");
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("waveform WGSL should validate");
    }

    /// The Rust uniform block and the WGSL `U` struct must agree; std140 rounds
    /// the block up to a multiple of 16. Guards against a field reorder silently
    /// corrupting the layout.
    #[test]
    fn wave_uniforms_match_std140_size() {
        assert_eq!(core::mem::size_of::<super::WaveUniforms>(), 80);
        assert_eq!(core::mem::size_of::<super::WaveUniforms>() % 16, 0);
    }
}
