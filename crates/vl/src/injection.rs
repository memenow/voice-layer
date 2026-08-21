//! macOS GUI text injection: clipboard write + synthetic Cmd+V.
//!
//! macOS has no AT-SPI equivalent available without the Accessibility API;
//! the pragmatic injection path is writing the transcript to the clipboard
//! and posting a Command+V key event into the focused application via
//! CoreGraphics. Posting events requires the user to grant the host
//! terminal/app Input Monitoring (or Accessibility) permission.

use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

/// macOS virtual key code for the `V` key (kVK_ANSI_V).
const KEY_CODE_V: CGKeyCode = 9;

/// Post a Command+V keystroke into the focused application.
pub fn post_command_v() -> Result<(), Box<dyn std::error::Error>> {
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|()| "unable to create a CoreGraphics event source")?;

    let key_down = CGEvent::new_keyboard_event(source.clone(), KEY_CODE_V, true)
        .map_err(|()| "unable to create the Cmd+V key-down event")?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);
    let key_up = CGEvent::new_keyboard_event(source, KEY_CODE_V, false)
        .map_err(|()| "unable to create the Cmd+V key-up event")?;
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);

    key_down.post(CGEventTapLocation::HID);
    key_up.post(CGEventTapLocation::HID);
    Ok(())
}
