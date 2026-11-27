//! VoiceLayer desktop shell entry point.
//!
//! Runs [`app::App`] inside an `iced::daemon` managing the main navigation
//! window and the floating capture HUD overlay.

mod accessibility;
mod app;
mod backdrop;
mod components;
mod config;
mod daemon_client;
mod forms;
mod hotkeys;
mod hud;
mod launcher;
mod state;
mod theme;
#[cfg(feature = "tray")]
mod tray;
mod views;

use app::App;

fn init_tracing() {
    let filter = std::env::var("VOICELAYER_LOG").unwrap_or_else(|_| "vl_desktop=info".to_owned());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

/// Named function (not a closure) to satisfy iced's higher-ranked
/// `for<'a> Fn(&'a App, window::Id)` lifetime bound.
fn resolve_title(_state: &App, _window: iced::window::Id) -> String {
    "VoiceLayer".to_owned()
}

/// Named function for the same higher-ranked lifetime reason as [`resolve_title`].
fn resolve_theme(_state: &App, _window: iced::window::Id) -> Option<iced::Theme> {
    Some(theme::app_theme())
}

fn resolve_style(_state: &App, _theme: &iced::Theme) -> iced::theme::Style {
    theme::app_style()
}

pub fn main() -> iced::Result {
    init_tracing();
    iced::daemon(App::boot, App::update, App::view)
        .title(resolve_title)
        .theme(resolve_theme)
        .style(resolve_style)
        // Register bundled Inter static faces (SIL OFL 1.1; see assets/fonts/OFL.txt)
        // so fontdb resolves each weight independently.
        .font(include_bytes!("../assets/fonts/Inter-Regular.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-Medium.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-SemiBold.ttf").as_slice())
        .font(include_bytes!("../assets/fonts/Inter-Bold.ttf").as_slice())
        .default_font(iced::Font::with_name(theme::FONT_FAMILY))
        .subscription(App::subscription)
        .run()
}
