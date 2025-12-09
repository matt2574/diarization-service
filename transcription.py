"""Transcription using faster-whisper."""

from faster_whisper import WhisperModel
from typing import Any

from config import get_settings


class TranscriptionService:
    """Service for audio transcription using faster-whisper."""

    def __init__(self):
        self.settings = get_settings()
        self.model: WhisperModel | None = None

    def load_model(self):
        """Load the Whisper model. Call this once at startup."""
        if self.model is not None:
            return

        print(f"Loading Whisper model: {self.settings.whisper_model}")

        # Determine compute type based on device
        compute_type = "float16" if self.settings.device == "cuda" else "int8"

        self.model = WhisperModel(
            self.settings.whisper_model,
            device=self.settings.device,
            compute_type=compute_type,
        )

        print("Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., "en", "es", "fr")

        Returns:
            List of segments with timing and text
        """
        if self.model is None:
            self.load_model()

        # Transcribe with word timestamps for better alignment
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,  # Filter out non-speech
        )

        # Convert generator to list of segments
        segments = []
        for segment in segments_gen:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
            )

        return segments


# Global instance
_transcription_service: TranscriptionService | None = None


def get_transcription_service() -> TranscriptionService:
    """Get the global transcription service instance."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service
