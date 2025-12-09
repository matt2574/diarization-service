"""Speaker diarization using pyannote.audio."""

import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from typing import Any

from config import get_settings


class DiarizationService:
    """Service for speaker diarization using pyannote.audio."""

    def __init__(self):
        self.settings = get_settings()
        self.pipeline: Pipeline | None = None
        self.embedding_model = None

    def load_models(self):
        """Load pyannote models. Call this once at startup."""
        if self.pipeline is not None:
            return

        print(f"Loading pyannote pipeline on device: {self.settings.device}")

        # Load diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.settings.huggingface_token,
        )

        # Move to appropriate device
        if self.settings.device == "cuda" and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

        # Load embedding model for voice profiles
        from pyannote.audio import Model

        self.embedding_model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=self.settings.huggingface_token,
        )
        if self.settings.device == "cuda" and torch.cuda.is_available():
            self.embedding_model.to(torch.device("cuda"))

        print("Models loaded successfully")

    def diarize(
        self,
        audio_path: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)

        Returns:
            List of segments with speaker labels and timing
        """
        if self.pipeline is None:
            self.load_models()

        # Build parameters
        params = {}
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        # Run diarization
        with ProgressHook() as hook:
            diarization = self.pipeline(audio_path, hook=hook, **params)

        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                }
            )

        return segments

    def extract_embeddings(
        self, audio_path: str, segments: list[dict[str, Any]]
    ) -> dict[str, list[float]]:
        """
        Extract voice embeddings for each speaker.

        Args:
            audio_path: Path to the audio file
            segments: List of diarization segments

        Returns:
            Dictionary mapping speaker labels to embedding vectors
        """
        if self.embedding_model is None:
            self.load_models()

        import torchaudio
        from pyannote.audio import Inference

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Create inference object
        inference = Inference(
            self.embedding_model,
            window="whole",
            device=torch.device(self.settings.device),
        )

        # Group segments by speaker
        speaker_segments: dict[str, list[tuple[float, float]]] = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((seg["start"], seg["end"]))

        # Extract embeddings for each speaker
        embeddings: dict[str, list[float]] = {}

        for speaker, times in speaker_segments.items():
            speaker_embeddings = []

            for start, end in times:
                # Skip very short segments
                if end - start < 0.5:
                    continue

                # Extract segment
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]

                # Get embedding
                try:
                    embedding = inference(
                        {"waveform": segment_waveform, "sample_rate": sample_rate}
                    )
                    speaker_embeddings.append(embedding)
                except Exception as e:
                    print(f"Failed to extract embedding for {speaker}: {e}")
                    continue

            # Average embeddings for this speaker
            if speaker_embeddings:
                avg_embedding = np.mean(speaker_embeddings, axis=0)
                embeddings[speaker] = avg_embedding.tolist()

        return embeddings


# Global instance
_diarization_service: DiarizationService | None = None


def get_diarization_service() -> DiarizationService:
    """Get the global diarization service instance."""
    global _diarization_service
    if _diarization_service is None:
        _diarization_service = DiarizationService()
    return _diarization_service
