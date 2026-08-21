//! Typed daemon event bus backed by a tokio broadcast channel.

use tokio::sync::broadcast;
use voicelayer_core::{DaemonEvent, EventEnvelope};

const EVENT_CHANNEL_CAPACITY: usize = 128;

#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<EventEnvelope>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self { sender }
    }

    pub fn emit(&self, event: DaemonEvent) {
        // No subscribers is a normal state; drop the send result.
        let _ = self.sender.send(EventEnvelope::new(event));
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.sender.subscribe()
    }
}
