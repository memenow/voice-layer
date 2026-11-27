//! Background event subscriptions bridged into the iced runtime.

use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;

use iced::futures::SinkExt;
use iced::futures::channel::mpsc as futures_mpsc;
use iced::stream;
use tokio::sync::mpsc::UnboundedSender;
use voicelayer_core::EventEnvelope;

use crate::accessibility;
use crate::daemon_client;
use crate::hotkeys::{self, SHORTCUT_TOGGLE};
use crate::state::SystemA11y;

use super::Message;

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Bridge an unbounded tokio channel driven by a background future into an iced stream.
fn bridge_stream<T, F, Fut, M>(
    spawn_driver: F,
    mut map_event: M,
) -> impl iced::futures::Stream<Item = Message>
where
    T: Send + 'static,
    F: FnOnce(UnboundedSender<T>) -> Fut + 'static,
    Fut: Future<Output = ()> + Send + 'static,
    M: FnMut(T) -> Option<Message> + 'static,
{
    stream::channel(8, async move |mut output: futures_mpsc::Sender<Message>| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<T>();
        let _driver = AbortOnDrop(tokio::spawn(spawn_driver(tx)));
        while let Some(item) = rx.recv().await {
            let Some(message) = map_event(item) else {
                continue;
            };
            if output.send(message).await.is_err() {
                break;
            }
        }
    })
}

pub(super) fn hotkey_stream() -> impl iced::futures::Stream<Item = Message> {
    bridge_stream(
        |tx| async move {
            if let Err(error) = hotkeys::run_listener(tx).await {
                tracing::warn!(error = %error, "hotkey listener returned error");
            }
        },
        |shortcut_id| (shortcut_id == SHORTCUT_TOGGLE).then_some(Message::HotkeyReceived),
    )
}

pub(super) fn accessibility_stream() -> impl iced::futures::Stream<Item = Message> {
    bridge_stream(
        |tx| async move {
            if let Err(error) = accessibility::watch(tx).await {
                tracing::debug!(error = %error, "accessibility portal watch ended");
            }
        },
        |system: SystemA11y| Some(Message::SystemA11yChanged(system)),
    )
}

#[cfg(feature = "tray")]
pub(super) fn tray_stream() -> impl iced::futures::Stream<Item = Message> {
    use crate::tray::{self, TrayCommand};

    bridge_stream(
        |tx| async move {
            if let Err(error) = tray::run(tx).await {
                tracing::info!(error = %error, "system tray unavailable; continuing without it");
            }
        },
        |command| {
            Some(match command {
                TrayCommand::ToggleDictation => Message::HotkeyReceived,
                TrayCommand::ShowMain => Message::TrayShowRequested,
                TrayCommand::Quit => Message::TrayQuitRequested,
            })
        },
    )
}

/// Long-lived daemon SSE subscription that reconnects after stream drop.
pub(super) fn sse_stream(
    socket_path: PathBuf,
    daemon_epoch: u64,
) -> impl iced::futures::Stream<Item = Message> {
    stream::channel(
        64,
        async move |mut output: futures_mpsc::Sender<Message>| {
            loop {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<EventEnvelope>();
                let reader_path = socket_path.clone();
                let reader_guard = AbortOnDrop(tokio::spawn(async move {
                    if let Err(error) = daemon_client::stream_events(&reader_path, tx).await {
                        tracing::debug!(error = %error, "event stream ended");
                    }
                }));
                while let Some(event) = rx.recv().await {
                    if output
                        .send(Message::EventReceived(daemon_epoch, event))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                drop(reader_guard);
                tokio::time::sleep(Duration::from_millis(1500)).await;
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::AbortOnDrop;

    #[tokio::test]
    async fn abort_on_drop_cancels_a_pending_reader_task() {
        let reader = tokio::spawn(std::future::pending::<()>());
        let abort_handle = reader.abort_handle();
        let guard = AbortOnDrop(reader);

        drop(guard);
        tokio::time::timeout(Duration::from_millis(100), async {
            while !abort_handle.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping AbortOnDrop should cancel the reader task");
    }
}
