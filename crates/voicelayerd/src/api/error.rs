//! RFC 9457 problem+json error surface for the control API.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use uuid::Uuid;
use voicelayer_core::ProblemDetails;

use crate::worker::WorkerCallError;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("no dictation session exists for id {0}")]
    SessionNotFound(Uuid),
    #[error("audio capture failed: {0}")]
    Recording(String),
    #[error("worker call failed: {0}")]
    Worker(#[from] WorkerCallError),
    #[error("internal error: {0}")]
    Internal(String),
}

struct StatusMapping {
    status: StatusCode,
    problem_type: &'static str,
    title: &'static str,
}

impl ApiError {
    fn mapping(&self) -> StatusMapping {
        match self {
            Self::BadRequest(_) => StatusMapping {
                status: StatusCode::BAD_REQUEST,
                problem_type: "bad_request",
                title: "Bad request",
            },
            Self::SessionNotFound(_) => StatusMapping {
                status: StatusCode::NOT_FOUND,
                problem_type: "session_not_found",
                title: "Session not found",
            },
            Self::Recording(_) => StatusMapping {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                problem_type: "recording_failed",
                title: "Audio capture failed",
            },
            Self::Worker(error) => worker_mapping(error),
            Self::Internal(_) => StatusMapping {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                problem_type: "internal",
                title: "Internal error",
            },
        }
    }
}

fn worker_mapping(error: &WorkerCallError) -> StatusMapping {
    match error {
        WorkerCallError::TimedOut => StatusMapping {
            status: StatusCode::GATEWAY_TIMEOUT,
            problem_type: "worker_timeout",
            title: "Worker timed out",
        },
        WorkerCallError::Rpc(rpc) if rpc.is_provider_unavailable() => StatusMapping {
            status: StatusCode::SERVICE_UNAVAILABLE,
            problem_type: "provider_unavailable",
            title: "No provider is configured for this workflow",
        },
        WorkerCallError::Rpc(_) => StatusMapping {
            status: StatusCode::BAD_GATEWAY,
            problem_type: "provider_request_failed",
            title: "Provider request failed",
        },
        _ => StatusMapping {
            status: StatusCode::SERVICE_UNAVAILABLE,
            problem_type: "worker_unavailable",
            title: "Worker is unavailable",
        },
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mapping = self.mapping();
        let problem = ProblemDetails::new(
            format!("urn:voicelayer:problem:{}", mapping.problem_type),
            mapping.title,
            mapping.status.as_u16(),
            self.to_string(),
        );
        (
            mapping.status,
            [("content-type", "application/problem+json")],
            Json(problem),
        )
            .into_response()
    }
}
