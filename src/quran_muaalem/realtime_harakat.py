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
from enum import Enum
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MODE ENUM
# =============================================================================

class Mode(str, Enum):
    """Operating mode for the realtime harakat system."""
    DETECT = "detect"  # Searching for which ayah user is reciting
    TRACK = "track"    # Locked on an ayah, tracking mistakes

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
INFER_INTERVAL = 0.2  # Seconds between inference runs
WINDOW_SEC_DETECT = 1.5  # Audio window for ayah detection (reduced from 2.2 for faster response)
WINDOW_SEC_TRACK = 2.5  # Audio window for mistake tracking (more context for tracking)
RING_BUFFER_SEC = 4.0  # Max audio buffer size (reduced from 6.0 to prevent "old audio drag")

# Detection thresholds
DETECT_THRESHOLD = 0.40  # Min score to detect ayah
STABILITY_CYCLES = 2  # Consecutive detections before locking

# Switching logic (winner-takes-lead approach)
SWITCH_MARGIN = 0.08  # New ayah must beat current ayah by this margin to trigger switch (lowered from 0.12)
SWITCH_STREAK = 2  # Consecutive frames where new ayah wins before switching (lowered from 3)
MIN_LOCK_TIME_SEC = 1.0  # Minimum time locked before allowing switch (lowered from 1.2)

# Silence handling
SILENCE_UNLOCK_SEC = 0.8  # After this silence, unlock (back to DETECT mode) (lowered from 1.0)
SILENCE_FLUSH_SEC = 1.5  # After this silence, clear ring buffer (lowered from 1.8)

# Mismatch persistence
MISMATCH_PERSISTENCE = 5  # Consecutive mismatches to mark slot as error

# Soft-unlock: If best_score stays below this for LOW_SCORE_UNLOCK_STREAK frames, go back to DETECT
LOW_SCORE_THRESHOLD = 0.30  # Below this score, consider it "uncertain"
LOW_SCORE_UNLOCK_STREAK = 5  # N consecutive low-score frames triggers soft-unlock

# VAD settings
VAD_RMS_THRESHOLD = 0.004  # Minimum RMS energy (lowered to 0.004 to work with quieter microphones)
VAD_WINDOW_MS = 300  # Window size for VAD RMS calculation in milliseconds


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

    # Mode state (DETECT or TRACK)
    mode: Mode = Mode.DETECT

    # Detection state
    active_ayah_idx: int | None = None
    lock_active: bool = False
    lock_since_ts: float = 0.0
    detected_history: deque = field(default_factory=lambda: deque(maxlen=STABILITY_CYCLES))

    # Switching state (winner-takes-lead logic)
    switch_candidate: int | None = None
    switch_streak: int = 0

    # Soft-unlock state (for prolonged uncertainty)
    low_score_streak: int = 0  # Consecutive frames with low best_score

    # Voice activity tracking
    last_voice_ts: float = 0.0

    # Tracking state
    cursor: int = 0  # Position in active ayah
    slot_error_counts: dict = field(default_factory=dict)
    wrong_slots: set = field(default_factory=set)
    uncertain_slots: set = field(default_factory=set)

    # Last inference results
    last_pred_harakat_seq: list = field(default_factory=list)
    last_score_by_ayah: dict = field(default_factory=dict)  # Changed to dict for scores_by_ayah
    last_update_ts: float = 0.0

    # Precomputed references (set during initialization)
    expected_harakat_by_ayah: dict = field(default_factory=dict)
    phonetizer_refs: dict = field(default_factory=dict)

    # Session status
    is_detecting: bool = True
    detection_confidence: float = 0.0
    last_speech_ts: float = 0.0  # Timestamp when speech was last detected (legacy, use last_voice_ts)


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

    # Batch extend is much faster than sample-by-sample append
    session.ring_audio.extend(chunk.tolist())


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

    Computes RMS on a smaller recent window (VAD_WINDOW_MS) to avoid being
    fooled by old audio in the buffer and to detect silence more quickly.

    Args:
        audio: Audio samples
        threshold: RMS threshold for speech detection

    Returns:
        True if speech is likely present
    """
    # Use only the last VAD_WINDOW_MS of audio for more responsive silence detection
    vad_samples = int(VAD_WINDOW_MS * SAMPLING_RATE / 1000)
    if len(audio) > vad_samples:
        recent_audio = audio[-vad_samples:]
    else:
        recent_audio = audio

    rms = compute_rms_energy(recent_audio)

    # Also compute variation to detect constant noise vs actual speech
    # Speech has more variation than constant background noise
    if len(recent_audio) > 100:
        # Compute RMS of chunks to detect variation
        chunk_size = len(recent_audio) // 4
        chunk_rms = [compute_rms_energy(recent_audio[i*chunk_size:(i+1)*chunk_size]) for i in range(4)]
        rms_variation = max(chunk_rms) - min(chunk_rms)
    else:
        rms_variation = 0.0

    # Speech if: RMS above threshold AND some variation (not constant noise)
    # OR very high RMS (definitely speech even if constant)
    is_speech = (rms > threshold and rms_variation > threshold * 0.3) or (rms > threshold * 2.0)

    # Log RMS periodically for debugging
    if hasattr(is_speech_present, '_log_counter'):
        is_speech_present._log_counter += 1
    else:
        is_speech_present._log_counter = 0

    if is_speech_present._log_counter % 10 == 0:  # Log every 10th call
        logger.info(f"VAD: RMS={rms:.6f}, variation={rms_variation:.6f}, threshold={threshold}, "
                   f"window_samples={len(recent_audio)}, speech_detected={is_speech}")

    return is_speech


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


# Groups of harakat that are often confused and should be treated as similar
SIMILAR_HARAKAT_GROUPS = [
    {'NONE', 'SUKOON'},  # Silent/no vowel often confused
    {'FATHA', 'FATHATAN'},  # Similar sounds
    {'KASRA', 'KASRATAN'},
    {'DAMMA', 'DAMMATAN'},
]


def is_harakat_similar(h1: str, h2: str) -> bool:
    """
    Check if two harakat are similar enough to not count as error.

    Args:
        h1: First harakat class
        h2: Second harakat class

    Returns:
        True if they are similar/equivalent
    """
    if h1 == h2:
        return True
    for group in SIMILAR_HARAKAT_GROUPS:
        if h1 in group and h2 in group:
            return True
    return False


def align_sequences(pred: list, exp: list, gap: str = "__GAP__") -> tuple:
    """
    Align two sequences using Levenshtein (edit-distance) dynamic programming.

    This properly handles insertions and deletions without causing cascading
    mismatches. Returns aligned sequences with gap markers where needed.

    Args:
        pred: Predicted harakat sequence
        exp: Expected harakat sequence
        gap: Marker string for gaps in alignment

    Returns:
        Tuple of (aligned_pred, aligned_exp) where both lists have same length
    """
    n, m = len(pred), len(exp)

    # Handle empty sequences
    if n == 0:
        return [gap] * m, list(exp)
    if m == 0:
        return list(pred), [gap] * n

    # DP tables
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]  # backtrack

    # Initialize first row and column
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", 0, j - 1)

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost is 0 if similar, 1 if different
            cost = 0 if is_harakat_similar(pred[i - 1], exp[j - 1]) else 1

            sub = dp[i - 1][j - 1] + cost  # substitution/match
            dele = dp[i - 1][j] + 1  # deletion from pred
            ins = dp[i][j - 1] + 1  # insertion to pred

            best = min(sub, dele, ins)
            dp[i][j] = best

            # Track backpointer
            if best == sub:
                bt[i][j] = ("sub", i - 1, j - 1)
            elif best == dele:
                bt[i][j] = ("del", i - 1, j)
            else:
                bt[i][j] = ("ins", i, j - 1)

    # Backtrack to build alignment
    i, j = n, m
    aligned_pred = []
    aligned_exp = []

    while i > 0 or j > 0:
        step = bt[i][j]
        if step is None:
            break
        if step[0] == "sub":
            # Match or substitution
            aligned_pred.append(pred[step[1]])
            aligned_exp.append(exp[step[2]])
            i -= 1
            j -= 1
        elif step[0] == "del":
            # Deletion: pred has extra element
            aligned_pred.append(pred[step[1]])
            aligned_exp.append(gap)
            i -= 1
        else:  # ins
            # Insertion: exp has extra element
            aligned_pred.append(gap)
            aligned_exp.append(exp[step[2]])
            j -= 1

    # Reverse since we built backwards
    aligned_pred.reverse()
    aligned_exp.reverse()

    return aligned_pred, aligned_exp


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
    scores_by_ayah: dict,
) -> None:
    """
    Update detection state with winner-takes-lead logic.

    Uses Mode-based state machine:
    - DETECT mode: Looking for which ayah user is reciting
    - TRACK mode: Locked on an ayah, tracking mistakes

    Switching logic: If different ayah consistently beats current by SWITCH_MARGIN,
    switch to that ayah (no need for current score to drop below threshold).

    Args:
        session: The realtime session
        best_ayah: Best detected ayah index
        best_score: Score for best ayah
        scores_by_ayah: Dict mapping ayah index to its score
    """
    now = time.time()

    # Store scores for debugging
    session.last_score_by_ayah = scores_by_ayah
    session.detection_confidence = best_score if best_ayah is not None else 0.0

    # DETECT MODE: Looking for which ayah user is reciting
    if session.mode == Mode.DETECT:
        # If no ayah detected or score too low, clear history
        if best_ayah is None or best_score < DETECT_THRESHOLD:
            session.detected_history.clear()
            return

        # Add to history
        session.detected_history.append(best_ayah)

        # Check if we should lock on this ayah
        if len(session.detected_history) >= STABILITY_CYCLES:
            # Check if all recent detections are the same ayah
            if all(h == best_ayah for h in session.detected_history):
                # Lock on this ayah - transition to TRACK mode
                logger.info(f"DETECT->TRACK: Locking on ayah {best_ayah} (score={best_score:.3f})")
                session.mode = Mode.TRACK
                session.lock_active = True
                session.active_ayah_idx = best_ayah
                session.lock_since_ts = now
                session.is_detecting = False
                session.switch_candidate = None
                session.switch_streak = 0
                # Initialize tracking state
                session.cursor = 0
                session.slot_error_counts.clear()
                session.wrong_slots.clear()
                session.uncertain_slots.clear()
        return

    # TRACK MODE: Locked on an ayah, tracking mistakes
    cur = session.active_ayah_idx
    if cur is None:
        # Invalid state - go back to DETECT
        session.mode = Mode.DETECT
        return

    # Check minimum lock time
    time_locked = now - session.lock_since_ts
    if time_locked < MIN_LOCK_TIME_SEC:
        # Too early to consider switching
        return

    # Get current ayah score
    cur_score = scores_by_ayah.get(cur, 0.0)

    # Soft-unlock: If best_score stays low for too long, go back to DETECT
    # This handles the case where user switches verse without pause
    if best_score < LOW_SCORE_THRESHOLD:
        session.low_score_streak += 1
        if session.low_score_streak >= LOW_SCORE_UNLOCK_STREAK:
            logger.info(f"SOFT UNLOCK: best_score={best_score:.3f} < {LOW_SCORE_THRESHOLD} "
                       f"for {session.low_score_streak} frames, switching to DETECT mode")
            session.mode = Mode.DETECT
            session.lock_active = False
            session.active_ayah_idx = None
            session.is_detecting = True
            session.detected_history.clear()
            session.switch_candidate = None
            session.switch_streak = 0
            session.low_score_streak = 0
            # Reset tracking state
            session.wrong_slots.clear()
            session.uncertain_slots.clear()
            session.slot_error_counts.clear()
            return
    else:
        session.low_score_streak = 0  # Reset streak if score is good

    # Winner-takes-lead: If another ayah beats current by margin, consider switching
    if best_ayah is not None and best_ayah != cur:
        margin = best_score - cur_score
        if margin >= SWITCH_MARGIN and best_score >= 0.45:
            # This ayah is beating current - track streak
            if session.switch_candidate == best_ayah:
                session.switch_streak += 1
            else:
                session.switch_candidate = best_ayah
                session.switch_streak = 1

            logger.info(f"Switch candidate: ayah {best_ayah} (margin={margin:.3f}, streak={session.switch_streak}/{SWITCH_STREAK})")

            # If streak is long enough, switch
            if session.switch_streak >= SWITCH_STREAK:
                logger.info(f"SWITCHING: ayah {cur} -> {best_ayah} "
                           f"(cur_score={cur_score:.3f}, new_score={best_score:.3f})")
                session.active_ayah_idx = best_ayah
                session.lock_since_ts = now
                session.switch_candidate = None
                session.switch_streak = 0
                # Reset tracking state for new ayah
                session.cursor = 0
                session.slot_error_counts.clear()
                session.wrong_slots.clear()
                session.uncertain_slots.clear()
        else:
            # Not beating by enough margin - reset streak
            session.switch_candidate = None
            session.switch_streak = 0
    else:
        # Current ayah is still best - reset streak
        session.switch_candidate = None
        session.switch_streak = 0


# =============================================================================
# LIVE MISTAKE TRACKING
# =============================================================================

def compare_to_active_ayah(
    session: RealtimeHarakatSession,
    pred_seq: list,
    ayah_idx: int,
) -> set:
    """
    Compare predicted harakat to active ayah using alignment-based comparison.

    Uses edit-distance DP alignment to handle:
    1. Timing stretch/compress without cascading errors
    2. Insertions/deletions without false positives
    3. Similar harakat groups (FATHA/FATHATAN, etc.)

    Args:
        session: The realtime session
        pred_seq: Predicted harakat sequence
        ayah_idx: Active ayah index

    Returns:
        Set of slot indices in EXPECTED sequence that have mismatches
    """
    if ayah_idx not in session.expected_harakat_by_ayah:
        return set()

    expected = session.expected_harakat_by_ayah[ayah_idx]

    if len(pred_seq) == 0 or len(expected) == 0:
        return set()

    # Align sequences using DP
    aligned_pred, aligned_exp = align_sequences(pred_seq, expected)

    mismatches = set()
    exp_pos = -1  # Track position in original expected sequence

    for p, e in zip(aligned_pred, aligned_exp):
        # Track position in expected sequence (ignoring gaps)
        if e != "__GAP__":
            exp_pos += 1

        # Skip gaps in expected (insertions in predicted)
        if e == "__GAP__":
            continue  # Don't blame any expected slot for extra predicted elements

        # Handle gaps in predicted (deletions - missing elements)
        if p == "__GAP__":
            # Mark as uncertain rather than definitely wrong
            # The user may not have reached this position yet
            session.uncertain_slots.add(exp_pos)
            continue

        # Both have values - check if they match
        if not is_harakat_similar(p, e):
            mismatches.add(exp_pos)

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

    Includes silence-triggered unlock:
    - After SILENCE_UNLOCK_SEC: Go back to DETECT mode
    - After SILENCE_FLUSH_SEC: Clear ring buffer

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
    window_sec = WINDOW_SEC_DETECT if session.mode == Mode.DETECT else WINDOW_SEC_TRACK
    audio = get_audio_window(session, window_sec)

    # Check VAD
    vad_is_voice = is_speech_present(audio)

    if vad_is_voice:
        # Speech detected - update timestamp
        session.last_voice_ts = now
        session.last_speech_ts = now  # Legacy field
    else:
        # No speech - handle silence
        silence_duration = now - session.last_voice_ts if session.last_voice_ts > 0 else 0.0

        # Silence-triggered unlock: If tracking and user stopped speaking, unlock quickly
        if session.mode == Mode.TRACK and silence_duration >= SILENCE_UNLOCK_SEC:
            logger.info(f"SILENCE UNLOCK: {silence_duration:.2f}s silence, switching to DETECT mode")
            session.mode = Mode.DETECT
            session.lock_active = False
            session.active_ayah_idx = None
            session.is_detecting = True
            session.detected_history.clear()
            session.switch_candidate = None
            session.switch_streak = 0
            # Reset tracking state
            session.wrong_slots.clear()
            session.uncertain_slots.clear()
            session.slot_error_counts.clear()

        # If silence is long enough, flush ring buffer (prevents old audio drag)
        if silence_duration >= SILENCE_FLUSH_SEC:
            logger.info(f"SILENCE FLUSH: {silence_duration:.2f}s silence, clearing ring buffer")
            session.ring_audio.clear()
            session.last_pred_harakat_seq = []

        return {
            'active_ayah_idx': session.active_ayah_idx,
            'is_detecting': session.mode == Mode.DETECT,
            'detection_confidence': session.detection_confidence,
            'wrong_slots': session.wrong_slots.copy(),
            'uncertain_slots': session.uncertain_slots.copy(),
            'has_speech': False,
            'mode': str(session.mode.value),
            'silence_sec': round(silence_duration, 2),
        }

    # Run inference
    try:
        logger.info("About to run inference...")
        pred_phonemes = run_inference(audio)
        logger.info(f"Inference returned phonemes (len={len(pred_phonemes) if pred_phonemes else 0}): {repr(pred_phonemes[:100]) if pred_phonemes else 'EMPTY/NONE'}")
        pred_harakat_seq = extract_harakat_stream(pred_phonemes)
        session.last_pred_harakat_seq = pred_harakat_seq
        logger.info(f"Extracted harakat seq length: {len(pred_harakat_seq)}, first 10: {pred_harakat_seq[:10] if pred_harakat_seq else 'EMPTY'}")
    except Exception as e:
        # Inference failed - log and return current state
        import traceback
        logger.error(f"Inference FAILED with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'active_ayah_idx': session.active_ayah_idx,
            'is_detecting': session.mode == Mode.DETECT,
            'detection_confidence': session.detection_confidence,
            'wrong_slots': session.wrong_slots.copy(),
            'uncertain_slots': session.uncertain_slots.copy(),
            'has_speech': True,
            'error': str(e),
            'mode': str(session.mode.value),
        }

    # Ensure expected harakat is precomputed
    if not session.expected_harakat_by_ayah:
        session.expected_harakat_by_ayah = precompute_expected_harakat(ayat)

    # Run detection - get scores for all ayahs
    best_ayah, best_score, all_scores = detect_ayah(
        pred_harakat_seq,
        session.expected_harakat_by_ayah,
        session.active_ayah_idx,
    )

    # Build scores_by_ayah dict
    scores_by_ayah = {}
    for i, exp_seq in session.expected_harakat_by_ayah.items():
        scores_by_ayah[i] = score_against_ayah_prefix(pred_harakat_seq, exp_seq)

    # Update detection state with winner-takes-lead logic
    update_detection_state(session, best_ayah, best_score, scores_by_ayah)

    # Run tracking if locked on an ayah
    if session.mode == Mode.TRACK and session.active_ayah_idx is not None:
        mismatches = compare_to_active_ayah(
            session,
            pred_harakat_seq,
            session.active_ayah_idx,
        )

        # Get number of slots for active ayah
        num_slots = len(session.expected_harakat_by_ayah.get(session.active_ayah_idx, []))

        # Update error tracking
        update_slot_errors(session, mismatches, num_slots)

    # Get current ayah score for debugging
    cur_score = scores_by_ayah.get(session.active_ayah_idx, 0.0) if session.active_ayah_idx is not None else 0.0

    return {
        'active_ayah_idx': session.active_ayah_idx,
        'is_detecting': session.mode == Mode.DETECT,
        'detection_confidence': session.detection_confidence,
        'wrong_slots': session.wrong_slots.copy(),
        'uncertain_slots': session.uncertain_slots.copy(),
        'has_speech': True,
        'pred_harakat_seq': pred_harakat_seq,
        'best_score': best_score,
        # Debug fields
        'mode': str(session.mode.value),
        'best_ayah': best_ayah,
        'current_score': cur_score,
        'switch_candidate': session.switch_candidate,
        'switch_streak': session.switch_streak,
        'silence_sec': 0.0,
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
    session.mode = Mode.DETECT
    session.active_ayah_idx = None
    session.lock_active = False
    session.lock_since_ts = 0.0
    session.detected_history.clear()
    session.switch_candidate = None
    session.switch_streak = 0
    session.low_score_streak = 0  # Reset soft-unlock streak
    session.last_voice_ts = 0.0
    session.cursor = 0
    session.slot_error_counts.clear()
    session.wrong_slots.clear()
    session.uncertain_slots.clear()
    session.last_pred_harakat_seq = []
    session.last_score_by_ayah = {}
    session.last_update_ts = 0.0
    session.is_detecting = True
    session.detection_confidence = 0.0
    session.last_speech_ts = 0.0


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
