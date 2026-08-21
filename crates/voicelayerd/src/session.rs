//! Bounded in-memory session store.
//!
//! Sessions are process-local by design: the daemon is a single-user local
//! service and a restart starts a clean session table. The store is bounded
//! (capacity + terminal-state TTL) so a long-running daemon cannot grow it
//! without limit.

use std::{collections::HashMap, sync::Arc, time::Duration};

use tokio::sync::RwLock;
use uuid::Uuid;
use voicelayer_core::{CaptureSession, SessionState, now_epoch_millis};

const MAX_SESSIONS: usize = 256;
const TERMINAL_TTL: Duration = Duration::from_secs(600);

struct SessionEntry {
    session: CaptureSession,
    /// Epoch millis when the session reached a terminal state.
    terminal_since: Option<u64>,
}

#[derive(Clone, Default)]
pub struct SessionStore {
    entries: Arc<RwLock<HashMap<Uuid, SessionEntry>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn upsert(&self, session: CaptureSession) {
        let mut entries = self.entries.write().await;
        let now = now_epoch_millis();
        let terminal_since = if is_terminal(session.state) {
            Some(
                entries
                    .get(&session.session_id)
                    .and_then(|entry| entry.terminal_since)
                    .unwrap_or(now),
            )
        } else {
            None
        };
        entries.insert(
            session.session_id,
            SessionEntry {
                session,
                terminal_since,
            },
        );
        Self::enforce_bounds(&mut entries, now);
    }

    pub async fn get(&self, session_id: Uuid) -> Option<CaptureSession> {
        self.entries
            .read()
            .await
            .get(&session_id)
            .map(|entry| entry.session.clone())
    }

    pub async fn list(&self) -> Vec<CaptureSession> {
        self.entries
            .read()
            .await
            .values()
            .map(|entry| entry.session.clone())
            .collect()
    }

    fn enforce_bounds(entries: &mut HashMap<Uuid, SessionEntry>, now: u64) {
        let ttl_millis = TERMINAL_TTL.as_millis() as u64;
        entries.retain(|_, entry| {
            entry
                .terminal_since
                .is_none_or(|since| now.saturating_sub(since) < ttl_millis)
        });

        while entries.len() > MAX_SESSIONS {
            // Evict the oldest terminal session first; if everything is
            // still live, evict the oldest entry regardless of state.
            let victim = entries
                .iter()
                .filter(|(_, entry)| entry.terminal_since.is_some())
                .min_by_key(|(_, entry)| entry.session.created_at_millis)
                .or_else(|| {
                    entries
                        .iter()
                        .min_by_key(|(_, entry)| entry.session.created_at_millis)
                })
                .map(|(id, _)| *id);
            match victim {
                Some(id) => {
                    entries.remove(&id);
                }
                None => break,
            }
        }
    }
}

fn is_terminal(state: SessionState) -> bool {
    matches!(state, SessionState::Completed | SessionState::Failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voicelayer_core::{LanguageProfile, SessionMode, TriggerKind};

    fn session_at(created_at_millis: u64, state: SessionState) -> CaptureSession {
        let mut session = CaptureSession::new(
            SessionMode::Dictation,
            TriggerKind::Cli,
            LanguageProfile::default(),
        );
        session.created_at_millis = created_at_millis;
        session.state = state;
        session
    }

    #[tokio::test]
    async fn store_evicts_oldest_terminal_session_beyond_capacity() {
        let store = SessionStore::new();
        for index in 0..MAX_SESSIONS + 10 {
            let state = if index % 2 == 0 {
                SessionState::Completed
            } else {
                SessionState::Listening
            };
            store.upsert(session_at(index as u64, state)).await;
        }
        let entries = store.entries.read().await;
        assert!(entries.len() <= MAX_SESSIONS);
    }

    #[tokio::test]
    async fn upsert_preserves_first_terminal_timestamp() {
        let store = SessionStore::new();
        let mut session = session_at(1, SessionState::Listening);
        let id = session.session_id;
        store.upsert(session.clone()).await;
        session.state = SessionState::Completed;
        store.upsert(session.clone()).await;
        let first_terminal = store.entries.read().await[&id].terminal_since;
        session.state = SessionState::Completed;
        store.upsert(session).await;
        assert_eq!(
            store.entries.read().await[&id].terminal_since,
            first_terminal
        );
    }
}
