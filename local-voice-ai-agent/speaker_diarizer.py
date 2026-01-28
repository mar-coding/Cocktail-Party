"""
Speaker Diarization Module (CPU-friendly)
------------------------------------------
Uses Resemblyzer for lightweight speaker embeddings.
Works on CPU without requiring GPU acceleration.

This module is optional and disabled by default. Enable with USE_DIARIZATION=true.
"""

import os
import numpy as np
from loguru import logger

# Feature flag - disabled by default for safety
USE_DIARIZATION = os.getenv("USE_DIARIZATION", "false").lower() == "true"

_encoder = None
_speaker_embeddings = {}  # Cache: speaker_id -> embedding


def get_encoder():
    """
    Lazy load Resemblyzer encoder.

    Returns:
        VoiceEncoder instance or None if unavailable
    """
    global _encoder
    if _encoder is None and USE_DIARIZATION:
        try:
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder()
            logger.info("Resemblyzer encoder initialized (CPU-friendly)")
        except ImportError as e:
            logger.warning(f"Resemblyzer unavailable: {e}")
    return _encoder


def identify_speaker(audio_data, sample_rate: int = 16000) -> str:
    """
    Identify speaker using voice embeddings.
    Compares against known speakers, assigns new ID if novel voice.

    Args:
        audio_data: Audio samples (numpy array)
        sample_rate: Sample rate of audio (default 16kHz)

    Returns:
        Speaker ID string: "Speaker_1", "Speaker_2", etc. or "User" on failure
    """
    if not USE_DIARIZATION:
        return "User"

    encoder = get_encoder()
    if encoder is None:
        return "User"

    try:
        # Ensure audio is float32 numpy array
        audio = np.asarray(audio_data, dtype=np.float32).flatten()

        # Need minimum audio length (~1 second at 16kHz)
        if len(audio) < sample_rate:
            return "User"

        # Normalize audio to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Get embedding for this audio
        embedding = encoder.embed_utterance(audio)

        # Compare to known speakers
        best_match = None
        best_score = 0.7  # Similarity threshold

        for speaker_id, known_embedding in _speaker_embeddings.items():
            score = np.dot(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_match = speaker_id

        if best_match:
            return best_match

        # New speaker detected
        new_id = f"Speaker_{len(_speaker_embeddings) + 1}"
        _speaker_embeddings[new_id] = embedding
        logger.debug(f"New speaker detected: {new_id}")
        return new_id

    except Exception as e:
        logger.debug(f"Diarization error: {e}")
        return "User"


def reset_speakers():
    """Clear known speakers (for new conversation)."""
    global _speaker_embeddings
    _speaker_embeddings = {}
    logger.debug("Speaker embeddings cleared")
