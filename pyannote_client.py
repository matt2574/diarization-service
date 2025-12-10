"""Client for pyannoteAI cloud API."""

import httpx
from typing import Any

from config import get_settings

PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"


class PyannoteClient:
    """Client for interacting with pyannoteAI cloud API."""

    def __init__(self):
        self.settings = get_settings()
        self.headers = {
            "Authorization": f"Bearer {self.settings.pyannote_api_key}",
            "Content-Type": "application/json",
        }

    async def submit_diarization(
        self,
        audio_url: str,
        webhook_url: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]:
        """
        Submit an audio file for diarization.

        Args:
            audio_url: URL of the audio file to process
            webhook_url: URL to receive results when complete
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Job info with jobId and status
        """
        payload: dict[str, Any] = {
            "url": audio_url,
            "model": self.settings.pyannote_model,
        }

        if webhook_url:
            payload["webhook"] = webhook_url

        if num_speakers is not None:
            payload["numSpeakers"] = num_speakers
        if min_speakers is not None:
            payload["minSpeakers"] = min_speakers
        if max_speakers is not None:
            payload["maxSpeakers"] = max_speakers

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PYANNOTE_API_BASE}/diarize",
                headers=self.headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def submit_identification(
        self,
        audio_url: str,
        voiceprints: list[dict[str, str]],
        webhook_url: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        matching_threshold: float = 0,
    ) -> dict[str, Any]:
        """
        Submit an audio file for diarization with speaker identification.

        Args:
            audio_url: URL of the audio file to process
            voiceprints: List of {"label": "Name", "voiceprint": "base64..."}
            webhook_url: URL to receive results when complete
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            matching_threshold: Minimum confidence for speaker matching (0-100)

        Returns:
            Job info with jobId and status
        """
        payload: dict[str, Any] = {
            "url": audio_url,
            "model": self.settings.pyannote_model,
            "voiceprints": voiceprints,
            "confidence": True,  # Enable confidence scores to debug matching
            "matching": {
                "threshold": matching_threshold,
                "exclusive": False,  # Allow multiple speakers to match same voiceprint
            },
        }

        if webhook_url:
            payload["webhook"] = webhook_url

        if num_speakers is not None:
            payload["numSpeakers"] = num_speakers
        if min_speakers is not None:
            payload["minSpeakers"] = min_speakers
        if max_speakers is not None:
            payload["maxSpeakers"] = max_speakers

        # Log the payload being sent (mask the actual voiceprint data)
        debug_payload = {**payload}
        if "voiceprints" in debug_payload:
            debug_payload["voiceprints"] = [
                {"label": vp["label"], "voiceprint_length": len(vp.get("voiceprint", ""))}
                for vp in payload["voiceprints"]
            ]
        print(f"[pyannote_client] Sending to /identify: {debug_payload}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PYANNOTE_API_BASE}/identify",
                headers=self.headers,
                json=payload,
                timeout=30.0,
            )
            print(f"[pyannote_client] Response status: {response.status_code}")
            response.raise_for_status()
            return response.json()

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """
        Get the status and results of a job.

        Args:
            job_id: The job ID returned from submit_diarization

        Returns:
            Job details including status and output (if complete)
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PYANNOTE_API_BASE}/jobs/{job_id}",
                headers=self.headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def create_voiceprint(
        self,
        audio_url: str,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a voiceprint from an audio sample.

        Args:
            audio_url: URL of the audio file (max 30 seconds)
            webhook_url: URL to receive results when complete

        Returns:
            Job info with jobId and status
        """
        payload: dict[str, Any] = {
            "url": audio_url,
            "model": self.settings.pyannote_model,  # Use same model as identification
        }

        if webhook_url:
            payload["webhook"] = webhook_url

        print(f"[pyannote_client] Creating voiceprint with model: {self.settings.pyannote_model}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PYANNOTE_API_BASE}/voiceprint",
                headers=self.headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


# Global client instance
_client: PyannoteClient | None = None


def get_pyannote_client() -> PyannoteClient:
    """Get the global pyannote client instance."""
    global _client
    if _client is None:
        _client = PyannoteClient()
    return _client
