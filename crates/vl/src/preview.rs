use voicelayer_core::{CompositionReceipt, PreviewArtifact, PreviewStatus, SessionMode};
use voicelayerd::WorkerPreviewPayload;

pub(crate) fn ready_receipt(preview: WorkerPreviewPayload) -> CompositionReceipt {
    CompositionReceipt {
        job_id: uuid::Uuid::new_v4(),
        preview: PreviewArtifact {
            artifact_id: uuid::Uuid::new_v4(),
            status: PreviewStatus::Ready,
            title: preview.title,
            generated_text: Some(preview.generated_text),
            notes: preview.notes,
        },
    }
}

pub(crate) fn worker_error_receipt(
    mode: SessionMode,
    title: &str,
    error: String,
) -> CompositionReceipt {
    let mut receipt = CompositionReceipt::needs_provider(title, mode);
    receipt
        .preview
        .notes
        .push(format!("Worker bridge detail: {error}"));
    receipt
}
