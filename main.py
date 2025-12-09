"""FastAPI application for speaker diarization service."""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings
from queue import get_queue, JobStatus
from worker import start_worker, stop_worker, process_job, download_audio
from diarization import get_diarization_service
from transcription import get_transcription_service
from alignment import align_transcript_with_speakers


# Request/Response models
class DiarizeRequest(BaseModel):
    """Request to diarize an audio file."""

    recording_id: str
    audio_url: str
    callback_url: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class DiarizeResponse(BaseModel):
    """Response for async diarization request."""

    job_id: str
    status: str
    message: str


class DiarizeSyncResponse(BaseModel):
    """Response for synchronous diarization."""

    recording_id: str
    segments: list[dict[str, Any]]
    speaker_count: int
    embeddings: dict[str, list[float]]
    full_transcript: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    recording_id: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
    queue_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: start the background worker
    print("Starting background worker...")
    start_worker()
    yield
    # Shutdown: stop the worker
    print("Stopping background worker...")
    stop_worker()


# Create FastAPI app
app = FastAPI(
    title="Speaker Diarization Service",
    description="Audio transcription with speaker diarization using pyannote.audio and Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    diarization_service = get_diarization_service()

    return HealthResponse(
        status="healthy",
        models_loaded=diarization_service.pipeline is not None,
        queue_type="redis" if settings.redis_url else "in-memory",
    )


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize_async(request: DiarizeRequest):
    """
    Submit an audio file for async diarization.

    The results will be sent to the callback_url when processing is complete.
    """
    settings = get_settings()
    queue = get_queue()

    # Use default callback URL if not provided
    callback_url = request.callback_url or settings.webhook_url

    # Add job to queue
    job = queue.add_job(
        recording_id=request.recording_id,
        audio_url=request.audio_url,
        callback_url=callback_url,
    )

    return DiarizeResponse(
        job_id=job.id,
        status="pending",
        message="Job queued for processing. Results will be sent to callback URL.",
    )


@app.post("/diarize/sync", response_model=DiarizeSyncResponse)
async def diarize_sync(request: DiarizeRequest):
    """
    Synchronously diarize an audio file.

    Use this for short audio files (<5 min). For longer files, use the async endpoint.
    """
    settings = get_settings()
    audio_path = None

    try:
        # Download audio
        async with httpx.AsyncClient() as client:
            response = await client.get(
                request.audio_url, timeout=120.0, follow_redirects=True
            )
            response.raise_for_status()

        # Save to temp file
        content_type = response.headers.get("content-type", "")
        ext = ".webm"
        if "mp4" in content_type or "m4a" in content_type:
            ext = ".m4a"
        elif "wav" in content_type:
            ext = ".wav"

        fd, audio_path = tempfile.mkstemp(suffix=ext)
        try:
            os.write(fd, response.content)
        finally:
            os.close(fd)

        # Run diarization
        diarization_service = get_diarization_service()
        diarization_segments = diarization_service.diarize(
            audio_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

        # Get unique speakers
        speakers = set(seg["speaker"] for seg in diarization_segments)
        speaker_count = len(speakers)

        # Run transcription
        transcription_service = get_transcription_service()
        transcript_segments = transcription_service.transcribe(audio_path)

        # Align transcript with speakers
        aligned_segments = align_transcript_with_speakers(
            transcript_segments, diarization_segments
        )

        # Extract speaker embeddings
        embeddings = diarization_service.extract_embeddings(
            audio_path, diarization_segments
        )

        return DiarizeSyncResponse(
            recording_id=request.recording_id,
            segments=[
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                }
                for seg in aligned_segments
            ],
            speaker_count=speaker_count,
            embeddings=embeddings,
            full_transcript=" ".join(seg["text"] for seg in aligned_segments),
        )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a diarization job."""
    queue = get_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        recording_id=job.recording_id,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.post("/identify")
async def identify_speakers(
    embeddings: dict[str, list[float]],
    known_speakers: list[dict[str, Any]],
):
    """
    Match speaker embeddings against known speaker profiles.

    Args:
        embeddings: Dictionary of speaker label -> embedding vector
        known_speakers: List of known speaker profiles with embeddings

    Returns:
        Suggested speaker assignments based on similarity
    """
    import numpy as np
    from numpy.linalg import norm

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (norm(a_arr) * norm(b_arr)))

    matches = {}
    threshold = 0.75  # Similarity threshold for a match

    for speaker_label, embedding in embeddings.items():
        best_match = None
        best_similarity = 0.0

        for known in known_speakers:
            if "embedding" not in known:
                continue

            similarity = cosine_similarity(embedding, known["embedding"])

            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = {
                    "speaker_id": known.get("id"),
                    "name": known.get("name"),
                    "similarity": similarity,
                }

        matches[speaker_label] = best_match

    return {"matches": matches}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
