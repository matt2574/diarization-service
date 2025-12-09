"""Background worker for processing diarization jobs."""

import os
import time
import tempfile
import threading
import traceback
from typing import Any

import httpx

from config import get_settings
from queue import get_queue, JobStatus, Job
from diarization import get_diarization_service
from transcription import get_transcription_service
from alignment import align_transcript_with_speakers


def download_audio(url: str) -> str:
    """Download audio file to a temporary location."""
    response = httpx.get(url, timeout=120.0, follow_redirects=True)
    response.raise_for_status()

    # Determine file extension from content-type or URL
    content_type = response.headers.get("content-type", "")
    if "webm" in content_type or url.endswith(".webm"):
        ext = ".webm"
    elif "mp4" in content_type or "m4a" in content_type or url.endswith(".m4a"):
        ext = ".m4a"
    elif "wav" in content_type or url.endswith(".wav"):
        ext = ".wav"
    elif "ogg" in content_type or url.endswith(".ogg"):
        ext = ".ogg"
    else:
        ext = ".webm"  # Default

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        os.write(fd, response.content)
    finally:
        os.close(fd)

    return path


def process_job(job: Job) -> dict[str, Any]:
    """Process a single diarization job."""
    settings = get_settings()
    queue = get_queue()
    audio_path = None

    try:
        # Update status to processing
        queue.update_job(job.id, status=JobStatus.PROCESSING)

        # Download audio
        print(f"Downloading audio for job {job.id}: {job.audio_url[:100]}...")
        audio_path = download_audio(job.audio_url)
        print(f"Downloaded to: {audio_path}")

        # Run diarization
        print(f"Running diarization for job {job.id}...")
        diarization_service = get_diarization_service()
        diarization_segments = diarization_service.diarize(
            audio_path,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
        )
        print(f"Found {len(diarization_segments)} diarization segments")

        # Get unique speakers
        speakers = set(seg["speaker"] for seg in diarization_segments)
        speaker_count = len(speakers)
        print(f"Detected {speaker_count} speakers")

        # Run transcription
        print(f"Running transcription for job {job.id}...")
        transcription_service = get_transcription_service()
        transcript_segments = transcription_service.transcribe(audio_path)
        print(f"Got {len(transcript_segments)} transcript segments")

        # Align transcript with speakers
        print(f"Aligning transcript with speakers...")
        aligned_segments = align_transcript_with_speakers(
            transcript_segments, diarization_segments
        )
        print(f"Created {len(aligned_segments)} aligned segments")

        # Extract speaker embeddings
        print(f"Extracting speaker embeddings...")
        embeddings = diarization_service.extract_embeddings(
            audio_path, diarization_segments
        )
        print(f"Extracted embeddings for {len(embeddings)} speakers")

        # Build result
        result = {
            "recording_id": job.recording_id,
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                }
                for seg in aligned_segments
            ],
            "speaker_count": speaker_count,
            "embeddings": embeddings,
            "full_transcript": " ".join(seg["text"] for seg in aligned_segments),
        }

        # Update job as completed
        queue.update_job(job.id, status=JobStatus.COMPLETED, result=result)

        # Send webhook callback
        send_webhook(job.callback_url, job.id, result, success=True)

        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Job {job.id} failed: {error_msg}")

        # Update job as failed
        queue.update_job(job.id, status=JobStatus.FAILED, error=error_msg)

        # Send failure webhook
        send_webhook(
            job.callback_url,
            job.id,
            {"error": str(e), "recording_id": job.recording_id},
            success=False,
        )

        raise

    finally:
        # Clean up temp file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def send_webhook(callback_url: str, job_id: str, data: dict[str, Any], success: bool):
    """Send webhook callback to Next.js app."""
    settings = get_settings()

    payload = {
        "job_id": job_id,
        "success": success,
        "data": data,
    }

    headers = {
        "Content-Type": "application/json",
    }

    if settings.webhook_secret:
        headers["X-Webhook-Secret"] = settings.webhook_secret

    try:
        response = httpx.post(
            callback_url,
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        print(f"Webhook sent successfully for job {job_id}")
    except Exception as e:
        print(f"Failed to send webhook for job {job_id}: {e}")


class WorkerThread(threading.Thread):
    """Background worker thread for processing jobs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        """Main worker loop."""
        queue = get_queue()

        # Preload models
        print("Preloading models...")
        get_diarization_service().load_models()
        get_transcription_service().load_model()
        print("Models loaded, worker ready")

        while self.running:
            try:
                job = queue.get_next_pending()
                if job:
                    print(f"Processing job {job.id}...")
                    process_job(job)
                else:
                    # No jobs, wait a bit
                    time.sleep(1)
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1)

    def stop(self):
        """Stop the worker thread."""
        self.running = False


# Global worker instance
_worker: WorkerThread | None = None


def start_worker():
    """Start the background worker."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = WorkerThread()
        _worker.start()


def stop_worker():
    """Stop the background worker."""
    global _worker
    if _worker:
        _worker.stop()
        _worker.join(timeout=5)
        _worker = None
