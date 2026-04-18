# Host Injection Strategy

## Principles

- Use the least invasive insertion mechanism that preserves correctness.
- Prefer semantic insertion over keyboard simulation.
- Keep terminal submission explicit. Never send Enter by default.

## GUI Targets

The preferred order for GUI insertion on Ubuntu GNOME Wayland is:

1. AT-SPI editable text insertion or replacement
2. Clipboard-assisted paste with restoration
3. Keyboard simulation fallback

### Why AT-SPI First

AT-SPI provides direct editable text operations and avoids timing races common with simulated input.
This is the preferred path for browser editors, GTK apps, Electron apps, and IDE surfaces that expose accessibility trees correctly.

### Why Clipboard Second

Clipboard-assisted paste works across a wide set of targets and is less layout-sensitive than per-key typing.
It still requires careful clipboard restore behavior and target-specific timing.

### Why Keyboard Simulation Last

Wayland intentionally limits unrestricted event injection.
`ydotool` and `wtype` remain useful escape hatches but should not define the product contract.

## Terminal and TUI Targets

The preferred order for terminal insertion is:

1. Bracketed paste payloads
2. Terminal-specific send-text adapters
3. Clipboard paste fallback when explicitly requested

### Default Safety Rule

Terminal injection must not submit text automatically.
If a user wants post-insert submission, that should be an explicit opt-in flag or per-target profile setting.

## Future Adapters

The adapter boundary is intentionally open for:

- kitty remote control
- tmux pane targeting
- Neovim RPC
- Terminal emulator-specific paste integrations

Those adapters should compose with the same `InjectionPlan` abstraction instead of creating parallel domain models.
