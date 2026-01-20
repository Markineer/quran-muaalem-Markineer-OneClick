"""
Real-Time Harakat Training Module

This module provides real-time harakat detection and tracking functionality
for live Quran recitation training. It includes:
- Audio ring buffer management
- Voice activity detection (VAD)
- Start-anywhere ayah detection
- Live mistake tracking with persistence
- Session state management

The system detects which ayah the user is reciting and tracks harakat
mistakes in real-time while they recite.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Callable
import time
import numpy as np

from .harakat_mode import (
    FATIHA_AYAT,
    parse_ayah_to_slots,
    marks_to_class,
    extract_harakat_stream,
    HarakatClass,
    HARAKAT_CLASS_ARABIC,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Audio settings
SAMPLING_RATE = 16000

# Timing settings
INFER_INTERVAL = 0.8  # Seconds between inference runs
WINDOW_SEC_DETECT = 6.0  # Audio window for ayah detection
WINDOW_SEC_TRACK = 4.0  # Audio window for mistake tracking
RING_BUFFER_SEC = 12.0  # Max audio buffer size

# Detection thresholds
DETECT_THRESHOLD = 0.62  # Min score to detect ayah
SWITCH_THRESHOLD = 0.70  # Min score to switch to new ayah
UNLOCK_THRESHOLD = 0.50  # Score below which to unlock current ayah

# Stability settings
STABILITY_CYCLES = 3  # Consecutive detections before locking
MISMATCH_PERSISTENCE = 2  # Consecutive mismatches to mark as error

# VAD settings
VAD_RMS_THRESHOLD = 0.01  # Minimum RMS energy to consider as speech


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RealtimeHarakatSession:
    """
    Session state for real-time harakat training.

    Manages audio buffering, ayah detection state, and mistake tracking.
    """
    # Audio buffer
    ring_audio: deque = field(default_factory=lambda: deque(maxlen=int(RING_BUFFER_SEC * SAMPLING_RATE)))

    # Detection state
    active_ayah_idx: int | None = None
    lock_active: bool = False
    lock_since_ts: float = 0.0
    detected_history: deque = field(default_factory=lambda: deque(maxlen=STABILITY_CYCLES))

    # Tracking state
    cursor: int = 0  # Position in active ayah
    slot_error_counts: dict = field(default_factory=dict)
    wrong_slots: set = field(default_factory=set)
    uncertain_slots: set = field(default_factory=set)

    # Last inference results
    last_pred_harakat_seq: list = field(default_factory=list)
    last_score_by_ayah: list = field(default_factory=list)
    last_update_ts: float = 0.0

    # Precomputed references (set during initialization)
    expected_harakat_by_ayah: dict = field(default_factory=dict)
    phonetizer_refs: dict = field(default_factory=dict)

    # Session status
    is_detecting: bool = True
    detection_confidence: float = 0.0


# =============================================================================
# AUDIO BUFFER FUNCTIONS
# =============================================================================

def push_audio(session: RealtimeHarakatSession, chunk: np.ndarray) -> None:
    """
    Add audio samples to the ring buffer.

    Args:
        session: The realtime session
        chunk: Audio samples (float32, mono, 16kHz)
    """
    # Ensure float32
    if chunk.dtype != np.float32:
        chunk = chunk.astype(np.float32)

    # Add samples to ring buffer
    for sample in chunk:
        session.ring_audio.append(sample)


def get_audio_window(session: RealtimeHarakatSession, seconds: float) -> np.ndarray:
    """
    Get the most recent audio from the buffer.

    Args:
        session: The realtime session
        seconds: Number of seconds to retrieve

    Returns:
        Audio samples as numpy array
    """
    num_samples = min(int(seconds * SAMPLING_RATE), len(session.ring_audio))
    if num_samples == 0:
        return np.array([], dtype=np.float32)

    # Get last N samples from deque
    samples = list(session.ring_audio)[-num_samples:]
    return np.array(samples, dtype=np.float32)


def get_buffer_duration(session: RealtimeHarakatSession) -> float:
    """Get current buffer duration in seconds."""
    return len(session.ring_audio) / SAMPLING_RATE


def clear_buffer(session: RealtimeHarakatSession) -> None:
    """Clear the audio buffer."""
    session.ring_audio.clear()


# =============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# =============================================================================

def compute_rms_energy(audio: np.ndarray) -> float:
    """
    Compute RMS energy of audio signal.

    Args:
        audio: Audio samples

    Returns:
        RMS energy value
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def is_speech_present(audio: np.ndarray, threshold: float = VAD_RMS_THRESHOLD) -> bool:
    """
    Check if speech is present in audio.

    Args:
        audio: Audio samples
        threshold: RMS threshold for speech detection

    Returns:
        True if speech is likely present
    """
    rms = compute_rms_energy(audio)
    return rms > threshold


# =============================================================================
# SIMILARITY SCORING
# =============================================================================

def levenshtein_distance(seq1: list, seq2: list) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance
    """
    m, n = len(seq1), len(seq2)

    # Handle empty sequences
    if m == 0:
        return n
    if n == 0:
        return m

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


def compute_similarity_score(pred_seq: list, exp_seq: list) -> float:
    """
    Compute normalized similarity score between predicted and expected sequences.

    Uses Levenshtein distance normalized by max length.
    Score of 1.0 = perfect match, 0.0 = completely different.

    Args:
        pred_seq: Predicted harakat sequence
        exp_seq: Expected harakat sequence

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if len(pred_seq) == 0 and len(exp_seq) == 0:
        return 1.0
    if len(pred_seq) == 0 or len(exp_seq) == 0:
        return 0.0

    dist = levenshtein_distance(pred_seq, exp_seq)
    max_len = max(len(pred_seq), len(exp_seq))

    return 1.0 - (dist / max_len)


def score_against_ayah_prefix(
    pred_seq: list,
    exp_seq: list,
    prefix_sizes: list = None
) -> float:
    """
    Score predicted sequence against prefix windows of expected sequence.

    Tries multiple prefix sizes and returns the best score.
    This handles the case where user is partway through an ayah.

    Args:
        pred_seq: Predicted harakat sequence
        exp_seq: Full expected harakat sequence for ayah
        prefix_sizes: List of prefix sizes to try

    Returns:
        Best similarity score across all prefix windows
    """
    if prefix_sizes is None:
        # Default prefix sizes: 6, 10, 14, 18, and full length
        prefix_sizes = [6, 10, 14, 18, len(exp_seq)]

    best_score = 0.0

    for size in prefix_sizes:
        if size > len(exp_seq):
            size = len(exp_seq)

        exp_window = exp_seq[:size]

        # Also try matching pred against this window
        # If pred is shorter, compare full pred
        # If pred is longer, compare prefix of pred
        pred_window = pred_seq[:size] if len(pred_seq) >= size else pred_seq

        score = compute_similarity_score(pred_window, exp_window)
        best_score = max(best_score, score)

    return best_score


# =============================================================================
# AYAH DETECTION
# =============================================================================

def get_expected_harakat_sequence_for_ayah(ayah_text: str) -> list:
    """
    Get the expected harakat sequence for an ayah.

    Args:
        ayah_text: The ayah text (Uthmani with tashkil)

    Returns:
        List of harakat class labels for each letter
    """
    slots = parse_ayah_to_slots(ayah_text)
    return [s.harakat_class for s in slots if s.is_letter]


def precompute_expected_harakat(ayat: list) -> dict:
    """
    Precompute expected harakat sequences for all ayat.

    Args:
        ayat: List of ayah texts

    Returns:
        Dict mapping ayah index to expected harakat sequence
    """
    expected = {}
    for i, ayah in enumerate(ayat):
        expected[i] = get_expected_harakat_sequence_for_ayah(ayah)
    return expected


def detect_ayah(
    pred_seq: list,
    expected_by_ayah: dict,
    current_ayah: int | None = None,
) -> tuple:
    """
    Detect which ayah the user is reciting.

    Args:
        pred_seq: Predicted harakat sequence from inference
        expected_by_ayah: Dict of expected harakat sequences per ayah
        current_ayah: Currently locked ayah (if any)

    Returns:
        Tuple of (best_ayah_idx, best_score, all_scores)
    """
    if len(pred_seq) == 0:
        return (None, 0.0, [])

    scores = []
    for i, exp_seq in expected_by_ayah.items():
        score = score_against_ayah_prefix(pred_seq, exp_seq)
        scores.append((i, score))

    if not scores:
        return (None, 0.0, [])

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    best_ayah = scores[0][0]
    best_score = scores[0][1]
    all_scores = [s[1] for s in sorted(scores, key=lambda x: x[0])]

    return (best_ayah, best_score, all_scores)


def update_detection_state(
    session: RealtimeHarakatSession,
    best_ayah: int | None,
    best_score: float,
) -> None:
    """
    Update detection state with stability/hysteresis logic.

    Args:
        session: The realtime session
        best_ayah: Best detected ayah index
        best_score: Score for best ayah
    """
    now = time.time()

    # If no ayah detected, clear history
    if best_ayah is None or best_score < DETECT_THRESHOLD:
        session.detected_history.clear()
        session.detection_confidence = best_score if best_ayah is not None else 0.0
        return

    # Add to history
    session.detected_history.append(best_ayah)
    session.detection_confidence = best_score

    # Check if currently locked
    if session.lock_active:
        # Check if should unlock (score dropped too low for current ayah)
        if session.active_ayah_idx is not None:
            # Calculate score for current locked ayah
            current_score = 0.0
            if session.active_ayah_idx in session.expected_harakat_by_ayah:
                current_score = score_against_ayah_prefix(
                    session.last_pred_harakat_seq,
                    session.expected_harakat_by_ayah[session.active_ayah_idx]
                )

            # Unlock if current ayah score is too low AND new ayah is much better
            if current_score < UNLOCK_THRESHOLD and best_score > SWITCH_THRESHOLD:
                # Only unlock if different ayah detected consistently
                if len(session.detected_history) >= STABILITY_CYCLES:
                    if all(h == best_ayah for h in session.detected_history):
                        if best_ayah != session.active_ayah_idx:
                            # Switch to new ayah
                            session.lock_active = False
                            session.active_ayah_idx = best_ayah
                            session.lock_since_ts = now
                            session.lock_active = True
                            # Reset tracking state
                            session.cursor = 0
                            session.slot_error_counts.clear()
                            session.wrong_slots.clear()
                            session.uncertain_slots.clear()
                            session.is_detecting = False
    else:
        # Not locked - try to lock on stable detection
        if len(session.detected_history) >= STABILITY_CYCLES:
            # Check if all recent detections are the same ayah
            if all(h == best_ayah for h in session.detected_history):
                if best_score >= DETECT_THRESHOLD:
                    # Lock on this ayah
                    session.active_ayah_idx = best_ayah
                    session.lock_active = True
                    session.lock_since_ts = now
                    session.is_detecting = False
                    # Initialize tracking state
                    session.cursor = 0
                    session.slot_error_counts.clear()
                    session.wrong_slots.clear()
                    session.uncertain_slots.clear()


# =============================================================================
# LIVE MISTAKE TRACKING
# =============================================================================

def compare_to_active_ayah(
    session: RealtimeHarakatSession,
    pred_seq: list,
    ayah_idx: int,
) -> set:
    """
    Compare predicted harakat to active ayah and find mismatches.

    Uses prefix alignment - assumes user started from beginning of ayah.

    Args:
        session: The realtime session
        pred_seq: Predicted harakat sequence
        ayah_idx: Active ayah index

    Returns:
        Set of slot indices with mismatches
    """
    if ayah_idx not in session.expected_harakat_by_ayah:
        return set()

    expected = session.expected_harakat_by_ayah[ayah_idx]
    mismatches = set()

    # Compare element by element up to length of predicted
    compare_len = min(len(pred_seq), len(expected))

    for i in range(compare_len):
        if pred_seq[i] != expected[i]:
            mismatches.add(i)

    return mismatches


def update_slot_errors(
    session: RealtimeHarakatSession,
    mismatches: set,
    num_slots: int,
) -> None:
    """
    Update slot error counts with persistence logic.

    A slot is marked as wrong only after MISMATCH_PERSISTENCE consecutive
    mismatches to avoid false positives from noise.

    Args:
        session: The realtime session
        mismatches: Set of currently mismatched slot indices
        num_slots: Total number of slots in active ayah
    """
    # Update error counts
    for i in range(num_slots):
        if i in mismatches:
            # Increment error count
            session.slot_error_counts[i] = session.slot_error_counts.get(i, 0) + 1

            # Mark as wrong if persistent
            if session.slot_error_counts[i] >= MISMATCH_PERSISTENCE:
                session.wrong_slots.add(i)
                session.uncertain_slots.discard(i)
            elif session.slot_error_counts[i] == 1:
                # First mismatch - mark as uncertain
                session.uncertain_slots.add(i)
        else:
            # Slot matches - decrement or clear error count
            if i in session.slot_error_counts:
                session.slot_error_counts[i] = max(0, session.slot_error_counts[i] - 1)
                if session.slot_error_counts[i] == 0:
                    session.wrong_slots.discard(i)
                    session.uncertain_slots.discard(i)


# =============================================================================
# MAIN UPDATE FUNCTION
# =============================================================================

def should_infer(session: RealtimeHarakatSession, now: float = None) -> bool:
    """
    Check if enough time has passed for next inference.

    Args:
        session: The realtime session
        now: Current timestamp (uses time.time() if None)

    Returns:
        True if inference should run
    """
    if now is None:
        now = time.time()
    return (now - session.last_update_ts) >= INFER_INTERVAL


def infer_and_update(
    session: RealtimeHarakatSession,
    run_inference: Callable,
    ayat: list = None,
) -> dict:
    """
    Main update function - run inference and update session state.

    Args:
        session: The realtime session
        run_inference: Callback function that takes audio and returns predicted phonemes
        ayat: List of ayah texts (defaults to FATIHA_AYAT)

    Returns:
        Dict with current state for rendering
    """
    now = time.time()
    session.last_update_ts = now

    if ayat is None:
        ayat = FATIHA_AYAT

    # Get appropriate audio window
    window_sec = WINDOW_SEC_DETECT if session.is_detecting else WINDOW_SEC_TRACK
    audio = get_audio_window(session, window_sec)

    # Check VAD
    if not is_speech_present(audio):
        return {
            'active_ayah_idx': session.active_ayah_idx,
            'is_detecting': session.is_detecting,
            'detection_confidence': session.detection_confidence,
            'wrong_slots': session.wrong_slots.copy(),
            'uncertain_slots': session.uncertain_slots.copy(),
            'has_speech': False,
        }

    # Run inference
    try:
        pred_phonemes = run_inference(audio)
        pred_harakat_seq = extract_harakat_stream(pred_phonemes)
        session.last_pred_harakat_seq = pred_harakat_seq
    except Exception as e:
        # Inference failed - return current state
        return {
            'active_ayah_idx': session.active_ayah_idx,
            'is_detecting': session.is_detecting,
            'detection_confidence': session.detection_confidence,
            'wrong_slots': session.wrong_slots.copy(),
            'uncertain_slots': session.uncertain_slots.copy(),
            'has_speech': True,
            'error': str(e),
        }

    # Ensure expected harakat is precomputed
    if not session.expected_harakat_by_ayah:
        session.expected_harakat_by_ayah = precompute_expected_harakat(ayat)

    # Run detection
    best_ayah, best_score, all_scores = detect_ayah(
        pred_harakat_seq,
        session.expected_harakat_by_ayah,
        session.active_ayah_idx,
    )
    session.last_score_by_ayah = all_scores

    # Update detection state
    update_detection_state(session, best_ayah, best_score)

    # Run tracking if locked on an ayah
    if session.lock_active and session.active_ayah_idx is not None:
        mismatches = compare_to_active_ayah(
            session,
            pred_harakat_seq,
            session.active_ayah_idx,
        )

        # Get number of slots for active ayah
        num_slots = len(session.expected_harakat_by_ayah.get(session.active_ayah_idx, []))

        # Update error tracking
        update_slot_errors(session, mismatches, num_slots)

    return {
        'active_ayah_idx': session.active_ayah_idx,
        'is_detecting': session.is_detecting,
        'detection_confidence': session.detection_confidence,
        'wrong_slots': session.wrong_slots.copy(),
        'uncertain_slots': session.uncertain_slots.copy(),
        'has_speech': True,
        'pred_harakat_seq': pred_harakat_seq,
        'best_score': best_score,
    }


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def create_session(ayat: list = None) -> RealtimeHarakatSession:
    """
    Create a new realtime harakat session.

    Args:
        ayat: List of ayah texts (defaults to FATIHA_AYAT)

    Returns:
        New RealtimeHarakatSession instance
    """
    if ayat is None:
        ayat = FATIHA_AYAT

    session = RealtimeHarakatSession()
    session.expected_harakat_by_ayah = precompute_expected_harakat(ayat)
    session.is_detecting = True

    return session


def reset_session(session: RealtimeHarakatSession) -> None:
    """
    Reset an existing session to initial state.

    Args:
        session: The session to reset
    """
    session.ring_audio.clear()
    session.active_ayah_idx = None
    session.lock_active = False
    session.lock_since_ts = 0.0
    session.detected_history.clear()
    session.cursor = 0
    session.slot_error_counts.clear()
    session.wrong_slots.clear()
    session.uncertain_slots.clear()
    session.last_pred_harakat_seq = []
    session.last_score_by_ayah = []
    session.last_update_ts = 0.0
    session.is_detecting = True
    session.detection_confidence = 0.0


def get_hint_map_for_active(
    session: RealtimeHarakatSession,
    ayat: list = None,
) -> dict:
    """
    Generate hint map for wrong slots in active ayah.

    Args:
        session: The realtime session
        ayat: List of ayah texts

    Returns:
        Dict mapping slot index to hint text
    """
    if ayat is None:
        ayat = FATIHA_AYAT

    if session.active_ayah_idx is None:
        return {}

    if session.active_ayah_idx >= len(ayat):
        return {}

    ayah_text = ayat[session.active_ayah_idx]
    slots = parse_ayah_to_slots(ayah_text)
    letter_slots = [s for s in slots if s.is_letter]

    expected = session.expected_harakat_by_ayah.get(session.active_ayah_idx, [])
    predicted = session.last_pred_harakat_seq

    hint_map = {}

    for slot_idx in session.wrong_slots:
        if slot_idx < len(expected):
            exp_class = expected[slot_idx]
            pred_class = predicted[slot_idx] if slot_idx < len(predicted) else "غير موجود"

            exp_arabic = HARAKAT_CLASS_ARABIC.get(exp_class, exp_class)
            pred_arabic = HARAKAT_CLASS_ARABIC.get(pred_class, pred_class)

            hint_map[slot_idx] = f"المتوقع: {exp_arabic} | المقروء: {pred_arabic}"

    return hint_map
