"""FastAPI application for speaker diarization service using pyannoteAI."""

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings
from pyannote_client import get_pyannote_client


# Request/Response models
class DiarizeRequest(BaseModel):
    """Request to diarize an audio file."""

    recording_id: str
    audio_url: str
    callback_url: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class IdentifyRequest(BaseModel):
    """Request to diarize with speaker identification."""

    recording_id: str
    audio_url: str
    voiceprints: list[dict[str, str]]  # [{"label": "John", "voiceprint": "base64..."}]
    callback_url: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class DiarizeResponse(BaseModel):
    """Response for diarization request."""

    job_id: str
    recording_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    output: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    model: str


# Create FastAPI app
app = FastAPI(
    title="Speaker Diarization Service",
    description="Audio diarization using pyannoteAI cloud API",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service="pyannoteAI",
        model=settings.pyannote_model,
    )


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(request: DiarizeRequest):
    """
    Submit an audio file for diarization.

    The results will be sent to the callback_url when processing is complete,
    or you can poll the /jobs/{job_id} endpoint.
    """
    settings = get_settings()
    client = get_pyannote_client()

    # Build webhook URL that includes recording_id for our callback
    callback_url = request.callback_url or settings.webhook_url
    # Append recording_id as query param so we know which recording this is for
    if "?" in callback_url:
        webhook_with_id = f"{callback_url}&recording_id={request.recording_id}"
    else:
        webhook_with_id = f"{callback_url}?recording_id={request.recording_id}"

    try:
        result = await client.submit_diarization(
            audio_url=request.audio_url,
            webhook_url=webhook_with_id,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

        return DiarizeResponse(
            job_id=result["jobId"],
            recording_id=request.recording_id,
            status=result["status"],
            message="Diarization job submitted to pyannoteAI",
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            raise HTTPException(status_code=402, detail="pyannoteAI payment required")
        elif e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="pyannoteAI rate limit exceeded")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")


@app.post("/identify", response_model=DiarizeResponse)
async def identify(request: IdentifyRequest):
    """
    Submit an audio file for diarization with speaker identification.

    Requires voiceprints to match speakers against known profiles.
    """
    settings = get_settings()
    client = get_pyannote_client()

    callback_url = request.callback_url or settings.webhook_url
    if "?" in callback_url:
        webhook_with_id = f"{callback_url}&recording_id={request.recording_id}"
    else:
        webhook_with_id = f"{callback_url}?recording_id={request.recording_id}"

    try:
        result = await client.submit_identification(
            audio_url=request.audio_url,
            voiceprints=request.voiceprints,
            webhook_url=webhook_with_id,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

        return DiarizeResponse(
            job_id=result["jobId"],
            recording_id=request.recording_id,
            status=result["status"],
            message="Identification job submitted to pyannoteAI",
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            raise HTTPException(status_code=402, detail="pyannoteAI payment required")
        elif e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="pyannoteAI rate limit exceeded")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status and results of a diarization job."""
    client = get_pyannote_client()

    try:
        result = await client.get_job(job_id)

        return JobStatusResponse(
            job_id=result.get("jobId", job_id),
            status=result.get("status", "unknown"),
            output=result.get("output"),
            created_at=result.get("createdAt"),
            updated_at=result.get("updatedAt"),
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")


@app.post("/voiceprint")
async def create_voiceprint(
    audio_url: str,
    callback_url: str | None = None,
):
    """
    Create a voiceprint from an audio sample.

    Audio should be max 30 seconds of a single speaker.
    """
    settings = get_settings()
    client = get_pyannote_client()

    try:
        result = await client.create_voiceprint(
            audio_url=audio_url,
            webhook_url=callback_url or settings.webhook_url,
        )

        return {
            "job_id": result["jobId"],
            "status": result["status"],
            "message": "Voiceprint job submitted to pyannoteAI",
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voiceprint creation failed: {e}")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
