"""Align transcription segments with speaker diarization."""

from typing import Any


def align_transcript_with_speakers(
    transcript_segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Align transcript text with speaker diarization.

    This combines the timing/text from Whisper with speaker labels from pyannote.

    Args:
        transcript_segments: Segments from Whisper with text and timing
        diarization_segments: Segments from pyannote with speaker labels

    Returns:
        Merged segments with speaker labels and text
    """
    if not transcript_segments or not diarization_segments:
        return []

    aligned_segments = []

    for trans_seg in transcript_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_mid = (trans_start + trans_end) / 2

        # Find the speaker for this segment by checking overlap
        # Use the speaker with the most overlap
        best_speaker = None
        best_overlap = 0

        for diar_seg in diarization_segments:
            diar_start = diar_seg["start"]
            diar_end = diar_seg["end"]

            # Calculate overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        # If no overlap found, use midpoint method
        if best_speaker is None:
            for diar_seg in diarization_segments:
                if diar_seg["start"] <= trans_mid <= diar_seg["end"]:
                    best_speaker = diar_seg["speaker"]
                    break

        aligned_segments.append(
            {
                "start": trans_start,
                "end": trans_end,
                "speaker": best_speaker or "UNKNOWN",
                "text": trans_seg["text"],
                "words": trans_seg.get("words", []),
            }
        )

    # Merge consecutive segments from the same speaker
    merged_segments = merge_consecutive_speaker_segments(aligned_segments)

    return merged_segments


def merge_consecutive_speaker_segments(
    segments: list[dict[str, Any]],
    max_gap: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Merge consecutive segments from the same speaker.

    Args:
        segments: List of aligned segments
        max_gap: Maximum gap (seconds) between segments to merge

    Returns:
        Merged segments
    """
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for seg in segments[1:]:
        # Check if same speaker and close enough in time
        same_speaker = seg["speaker"] == current["speaker"]
        gap = seg["start"] - current["end"]

        if same_speaker and gap <= max_gap:
            # Merge: extend current segment
            current["end"] = seg["end"]
            current["text"] = current["text"] + " " + seg["text"]
            current["words"] = current.get("words", []) + seg.get("words", [])
        else:
            # Different speaker or too large gap: save current and start new
            merged.append(current)
            current = seg.copy()

    # Don't forget the last segment
    merged.append(current)

    return merged
