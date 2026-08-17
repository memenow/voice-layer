//! VoiceLayer desktop shell entry point.
//!
//! Wires the [`app::App`] controller into an `iced::daemon` (multi-window: the
//! main navigation window and the capture HUD overlay, with a settings window in
//! a later milestone). State, updates, and subscriptions live in [`app`];
//! the navigation shell and dictation/providers/doctor panels in [`view`], the
//! generative / history / settings panels in [`workflows`], and their form state
//! in [`forms`]; the `/v1` client and SSE reader in [`api`]; the shared Liquid
//! Glass tokens in [`theme`] / [`components`]; the animated wgpu glass backdrop
//! in [`glass`]; the floating capture overlay in [`hud`]; and, behind the
//! optional `tray` feature, a `StatusNotifierItem` system tray in `tray`.

mod a11y;
mod api;
mod app;
mod components;
mod config;
mod forms;
mod glass;
mod hud;
mod launcher;
mod portal;
mod state;
mod theme;
#[cfg(feature = "tray")]
mod tray;
mod view;
mod workflows;

use app::App;

fn init_tracing() {
    let filter = std::env::var("VOICELAYER_LOG").unwrap_or_else(|_| "vl_desktop=info".to_owned());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

/// Title resolver. A named function (not a closure) so it satisfies the
/// higher-ranked `for<'a> Fn(&'a App, window::Id)` bound the daemon builder
/// requires; an equivalent closure infers one concrete lifetime and is rejected.
fn resolve_title(_state: &App, _window: iced::window::Id) -> String {
    "VoiceLayer".to_owned()
}

/// Theme resolver. The daemon `.theme` closure returns `Option<Theme>` and
/// receives the window id; named for the same higher-ranked-lifetime reason.
fn resolve_theme(_state: &App, _window: iced::window::Id) -> Option<iced::Theme> {
    Some(theme::app_theme())
}

/// Window background resolver. The daemon `.style` closure takes `(&State,
/// &Theme)` — no window id — and returns the per-window background + text color.
fn resolve_style(_state: &App, _theme: &iced::Theme) -> iced::theme::Style {
    theme::app_style()
}

pub fn main() -> iced::Result {
    init_tracing();
    iced::daemon(App::boot, App::update, App::view)
        .title(resolve_title)
        .theme(resolve_theme)
        .style(resolve_style)
        // Register the bundled Inter static instances (SIL OFL 1.1, see
        // assets/fonts/OFL.txt) and make Inter the default family; fontdb resolves
        // each token weight from these static faces.
        .font(include_bytes!("../assets/fonts/Inter-Regular.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-Medium.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-SemiBold.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-Bold.ttf").as_slice())
        .default_font(iced::Font::with_name(theme::FONT_FAMILY))
        .subscription(App::subscription)
        .run()
}
