"""
Quran Muaalem - FastAPI Backend

This backend provides:
- REST API for audio analysis
- WebSocket for real-time audio streaming
- Integration with the Muaalem model
"""

import asyncio
import json
import logging
import tempfile
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quran_muaalem.harakat_mode import (
    FATIHA_AYAT,
    parse_ayah_to_slots,
    extract_harakat_stream,
    compare_harakat_to_slots,
    render_ayah_with_highlights,
    get_expected_sequences_for_all_ayat,
)
from quran_muaalem.realtime_harakat import (
    create_session,
    infer_and_update,
    push_audio,
    get_buffer_duration,
    get_hint_map_for_active,
    get_audio_window,
    score_against_ayah_prefix,
    SAMPLING_RATE,
    Mode,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Quran Muaalem API",
    description="AI-powered Quran recitation analysis",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded on first request)
muaalem_model = None
moshaf_settings = None

# Precomputed phonetizer references for each ayah (Fix 2: removes reference bias)
PHON_REF_BY_AYAH = []
FAILED_AYAHS = set()  # Track which ayahs failed phonetization

# SSE Sessions (for Server-Sent Events alternative to WebSocket)
SESSIONS: Dict[str, "RealtimeSession"] = {}
SESSION_QUEUES: Dict[str, asyncio.Queue] = {}


def init_phonetizer_refs(settings):
    """
    Precompute phonetizer references for ALL ayahs once at startup.

    This eliminates reference bias during detection - we can run inference
    against all ayah references and pick the best match.
    """
    global PHON_REF_BY_AYAH, FAILED_AYAHS
    from quran_transcript import quran_phonetizer

    PHON_REF_BY_AYAH = []
    FAILED_AYAHS = set()  # Track which ayahs can't be phonetized

    for i, txt in enumerate(FATIHA_AYAT):
        try:
            ref = quran_phonetizer(txt, settings, remove_spaces=True)

            # Validate that ref doesn't contain None values
            has_none = False
            if isinstance(ref, dict):
                for level_name, level_seq in ref.items():
                    if level_seq is None or (isinstance(level_seq, (list, tuple)) and None in level_seq):
                        has_none = True
                        logger.warning(f"Ayah {i} level '{level_name}' contains None, will skip this ayah")
                        break

            if has_none:
                PHON_REF_BY_AYAH.append("")
                FAILED_AYAHS.add(i)
                logger.warning(f"Ayah {i} phonetizer produced invalid output (contains None)")
            else:
                PHON_REF_BY_AYAH.append(ref)
                logger.info(f"Ayah {i} phonetized OK")
        except Exception as e:
            # For ayahs that fail, use empty string (will be skipped in detection)
            # This is a workaround for a bug in quran_transcript library
            PHON_REF_BY_AYAH.append("")
            FAILED_AYAHS.add(i)
            logger.warning(f"Ayah {i} phonetizer FAILED (will be skipped): {e}")

    logger.info(f"Precomputed {len(PHON_REF_BY_AYAH)} ayah phonetizer references ({len(FAILED_AYAHS)} failed)")


def get_model():
    """Lazy load the Muaalem model."""
    global muaalem_model, moshaf_settings

    # Also check PHON_REF_BY_AYAH - if model loaded but refs failed, we need to retry
    if muaalem_model is None or len(PHON_REF_BY_AYAH) == 0:
        try:
            import torch
            from quran_muaalem.inference import Muaalem
            from quran_transcript import MoshafAttributes

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Muaalem model on {device}...")

            # CPU optimizations
            if device == "cpu":
                # Use all available CPU cores
                num_threads = torch.get_num_threads()
                logger.info(f"PyTorch using {num_threads} CPU threads")
                # Use float32 on CPU - bfloat16 emulation is slower
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

            muaalem_model = Muaalem(
                model_name_or_path="obadx/muaalem-model-v3_2",
                device=device,
                dtype=dtype
            )

            moshaf_settings = MoshafAttributes(
                rewaya="hafs",
                madd_monfasel_len=4,
                madd_mottasel_len=4,
                madd_mottasel_waqf=4,
                madd_aared_len=4,
            )

            # Precompute phonetizer references for all ayahs (Fix 2)
            init_phonetizer_refs(moshaf_settings)

            logger.info("Model loaded successfully!")

        except Exception as e:
            import traceback
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Failed to load model")

    return muaalem_model, moshaf_settings


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Quran Muaalem API is running"}


@app.get("/api/fatiha")
async def get_fatiha():
    """Get Al-Fatiha ayat."""
    return {
        "ayat": FATIHA_AYAT,
        "count": len(FATIHA_AYAT),
    }


@app.post("/api/analyze-harakat")
async def analyze_harakat(
    audio: UploadFile = File(...),
    ayah_idx: int = Form(-1),
    settings: str = Form("{}"),
):
    """
    Analyze audio for harakat errors.

    Args:
        audio: Audio file (WAV, MP3, etc.)
        ayah_idx: Ayah index (-1 for full Fatiha)
        settings: JSON string of moshaf settings
    """
    try:
        # Parse settings
        user_settings = json.loads(settings)

        # Get model
        model, default_settings = get_model()

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            from librosa.core import load
            from quran_transcript import quran_phonetizer, MoshafAttributes

            # Load audio
            wave, _ = load(tmp_path, sr=SAMPLING_RATE, mono=True)

            # Get ayah text
            if ayah_idx == -1:
                ayah_text = " ".join(FATIHA_AYAT)
            else:
                ayah_text = FATIHA_AYAT[ayah_idx]

            # Get phonetic representation
            settings_obj = MoshafAttributes(
                rewaya=user_settings.get("rewaya", "hafs"),
                madd_monfasel_len=user_settings.get("madd_monfasel_len", 4),
                madd_mottasel_len=user_settings.get("madd_mottasel_len", 4),
                madd_mottasel_waqf=user_settings.get("madd_mottasel_waqf", 4),
                madd_aared_len=user_settings.get("madd_aared_len", 4),
            )

            phonetizer_out = quran_phonetizer(ayah_text, settings_obj, remove_spaces=True)

            # Run inference
            outs = model(
                [wave],
                [phonetizer_out],
                sampling_rate=SAMPLING_RATE,
            )

            # Extract harakat and compare
            slots = parse_ayah_to_slots(ayah_text)
            pred_harakat = extract_harakat_stream(outs[0].phonemes.text)
            wrong_idxs, hints = compare_harakat_to_slots(slots, pred_harakat)

            # Render HTML
            html = render_ayah_with_highlights(ayah_text, slots, wrong_idxs, hints)

            # Count stats
            letter_count = sum(1 for s in slots if s.is_letter)
            wrong_count = len(wrong_idxs)
            correct_count = letter_count - wrong_count

            return {
                "html": f'<p class="font-arabic text-2xl leading-loose">{html}</p>',
                "stats": {
                    "correct": correct_count,
                    "wrong": wrong_count,
                    "uncertain": 0,
                    "total": letter_count,
                },
            }

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SSE (Server-Sent Events) for Real-Time Streaming
# =============================================================================

@app.post("/api/realtime/session")
async def create_realtime_session():
    """
    Create a new real-time session for SSE-based audio streaming.

    Returns:
        session_id: Unique session identifier
    """
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = RealtimeSession()
    SESSION_QUEUES[session_id] = asyncio.Queue()
    logger.info(f"Created SSE session: {session_id}")
    return {"session_id": session_id}


@app.get("/api/realtime/stream")
async def sse_stream(session_id: str, request: Request):
    """
    SSE stream endpoint - server pushes JSON events to client.

    Args:
        session_id: Session identifier from create_realtime_session
        request: FastAPI request object (to detect disconnection)

    Returns:
        StreamingResponse with text/event-stream content
    """
    if session_id not in SESSION_QUEUES:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    queue = SESSION_QUEUES[session_id]
    logger.info(f"SSE client connected: {session_id}")

    async def event_generator():
        try:
            # Send initial connected event
            yield f"data: {json.dumps({'type': 'status', 'status': 'connected'})}\n\n"

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"SSE client disconnected: {session_id}")
                    break

                try:
                    # Wait for message with timeout to check disconnection periodically
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    # Send message as SSE event
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    # No message, send keepalive comment
                    yield ": keepalive\n\n"
                    continue

        except Exception as e:
            logger.error(f"SSE stream error for {session_id}: {e}")
        finally:
            logger.info(f"SSE stream closed for {session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/api/realtime/push")
async def push_audio_chunk(session_id: str, request: Request):
    """
    Receive audio chunk from client and process it.

    Args:
        session_id: Session identifier
        request: FastAPI request with raw PCM audio bytes in body

    Returns:
        {"ok": True} on success
    """
    if session_id not in SESSIONS or session_id not in SESSION_QUEUES:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    session = SESSIONS[session_id]
    queue = SESSION_QUEUES[session_id]

    # Lazy-load model
    model, settings = get_model()

    # Read raw audio bytes
    data = await request.body()
    if not data:
        return {"ok": True}

    # Convert from Int16 to float32
    audio_int16 = np.frombuffer(data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # Debug logging (periodically)
    if not hasattr(session, '_audio_log_count'):
        session._audio_log_count = 0
    session._audio_log_count += 1
    if session._audio_log_count % 10 == 1:  # Log every 10th chunk
        max_val = np.max(np.abs(audio_float32)) if len(audio_float32) > 0 else 0
        logger.info(f"[SSE {session_id[:8]}] Audio chunk #{session._audio_log_count}: len={len(audio_int16)}, max={max_val:.4f}")

    # Push audio to session buffer
    push_audio(session.session, audio_float32)

    # Check if we should run inference
    buffer_duration = get_buffer_duration(session.session)

    # Run inference only if enough audio and not already running
    if buffer_duration >= 0.5 and not session.inference_running:
        session.inference_running = True

        async def run_inference_and_publish():
            try:
                current_ayah_idx = session.session.active_ayah_idx

                # Run inference
                def inference_fn(audio):
                    return run_inference(model, audio, settings, ayah_idx=current_ayah_idx)

                result = await asyncio.to_thread(
                    infer_and_update,
                    session.session,
                    inference_fn,
                    FATIHA_AYAT,
                )

                # Publish events through SSE queue (shape matches frontend expectations)
                is_detecting = result.get("is_detecting", True)

                if not is_detecting and result.get("active_ayah_idx") is not None:
                    # Ayah detected
                    await queue.put({
                        "type": "detection",
                        "ayah_idx": result["active_ayah_idx"],
                        "confidence": result.get("detection_confidence", 0.0),
                        "mode": result.get("mode", "unknown"),
                    })

                    # Get hints
                    hints = get_hint_map_for_active(session.session, FATIHA_AYAT)

                    # Tracking info
                    await queue.put({
                        "type": "tracking",
                        "wrong_slots": list(result.get("wrong_slots", set())),
                        "uncertain_slots": list(result.get("uncertain_slots", set())),
                        "hints": hints,
                    })
                else:
                    # Still detecting or waiting
                    await queue.put({
                        "type": "status",
                        "status": "detecting" if result.get("has_speech", False) else "waiting",
                        "mode": result.get("mode", "unknown"),
                    })

            except Exception as e:
                logger.error(f"Inference error in SSE session {session_id[:8]}: {e}")
                await queue.put({"type": "error", "message": str(e)})
            finally:
                session.inference_running = False

        # Fire and forget
        asyncio.create_task(run_inference_and_publish())

    return {"ok": True}


# =============================================================================
# WebSocket for Real-Time Streaming
# =============================================================================

class RealtimeSession:
    """Manages a real-time harakat training session."""

    def __init__(self):
        self.session = create_session(FATIHA_AYAT)
        self.expected_seqs = get_expected_sequences_for_all_ayat(FATIHA_AYAT)
        self.is_active = False
        self.last_status = "waiting"
        self.inference_running = False  # Debounce flag to prevent inference queueing
        self.last_result = None  # Cache last result
        self.last_infer_start_ts = 0.0  # Track when inference started (for metrics)
        self.last_infer_done_ts = 0.0  # Track when inference finished (for metrics)
        self.infer_task = None  # Reference to current inference task

    def reset(self):
        self.session = create_session(FATIHA_AYAT)
        self.is_active = False
        self.last_status = "waiting"
        self.inference_running = False
        self.last_result = None
        self.last_infer_start_ts = 0.0
        self.last_infer_done_ts = 0.0
        self.infer_task = None


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time harakat training.

    Client sends: Raw audio chunks (Int16Array)
    Server sends: JSON messages with detection/tracking results
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    session = RealtimeSession()

    try:
        # Get model (may take time on first connection)
        model, settings = get_model()

        # Send ready status
        await websocket.send_json({
            "type": "status",
            "status": "detecting",
            "message": "Ready for audio",
        })

        # Process incoming audio
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()

            # Convert from Int16 to float32
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Debug: Log audio stats periodically
            if not hasattr(session, '_audio_log_count'):
                session._audio_log_count = 0
            session._audio_log_count += 1
            if session._audio_log_count % 50 == 1:  # Log every 50th chunk
                # Log raw bytes info
                raw_bytes_len = len(data)
                int16_max = np.max(np.abs(audio_int16)) if len(audio_int16) > 0 else 0
                max_val = np.max(np.abs(audio_float32)) if len(audio_float32) > 0 else 0
                rms_val = np.sqrt(np.mean(audio_float32**2)) if len(audio_float32) > 0 else 0
                # Show first few int16 samples
                first5_int16 = list(audio_int16[:5]) if len(audio_int16) >= 5 else list(audio_int16)
                first5_float = [f"{x:.6f}" for x in audio_float32[:5]] if len(audio_float32) >= 5 else [f"{x:.6f}" for x in audio_float32]
                logger.info(f"Audio chunk #{session._audio_log_count}: raw_bytes={raw_bytes_len}, int16_max={int16_max}, first5_int16={first5_int16}, first5_float32={first5_float}")

            # Push to session buffer
            push_audio(session.session, audio_float32)

            # Check if we should run inference
            buffer_duration = get_buffer_duration(session.session)

            if buffer_duration >= 0.5:  # At least 0.5 second of audio (reduced for lower latency)
                # CRITICAL: Skip if inference is already running to prevent backlog
                # This is the key fix for the 30+ second delay issue
                if session.inference_running:
                    # Log occasionally to show we're skipping
                    if not hasattr(session, '_skip_count'):
                        session._skip_count = 0
                    session._skip_count += 1
                    if session._skip_count % 50 == 1:
                        logger.info(f"Skipping inference (already running), skip_count={session._skip_count}")
                    continue

                # Mark inference as running BEFORE creating task
                session.inference_running = True
                session.last_infer_start_ts = time.time()

                # Create async task for inference (fire-and-forget pattern)
                # This prevents blocking the audio receive loop
                async def run_inference_task():
                    try:
                        # Get current mode and ayah for inference strategy
                        current_mode = session.session.mode
                        current_ayah_idx = session.session.active_ayah_idx

                        # Run inference with mode-aware strategy
                        def inference_with_mode_aware_logging(audio):
                            import time as _time
                            _start = _time.time()

                            # DISABLED: Multi-ayah detection due to quran_transcript library bug
                            # The library produces None values in phonetizer output which causes crashes
                            # Using simple single-ayah mode for all modes
                            phonemes = run_inference(model, audio, settings, ayah_idx=current_ayah_idx)
                            _elapsed = _time.time() - _start
                            logger.info(f"Single-ayah inference took {_elapsed*1000:.0f}ms, "
                                       f"phonemes: {phonemes[:50] if phonemes else 'EMPTY'}...")
                            return phonemes

                        inference_start = time.time()
                        result = await asyncio.to_thread(
                            infer_and_update,
                            session.session,
                            inference_with_mode_aware_logging,
                            FATIHA_AYAT,
                        )
                        total_elapsed = time.time() - inference_start

                        # Log timing metrics to detect backlog
                        time_since_last = time.time() - session.last_infer_done_ts if session.last_infer_done_ts > 0 else 0
                        logger.info(f"Inference complete: infer_ms={total_elapsed*1000:.0f}, "
                                   f"buffer_sec={buffer_duration:.2f}, "
                                   f"time_since_last={time_since_last*1000:.0f}ms")

                        session.last_result = result  # Cache result
                        session.last_infer_done_ts = time.time()

                        # Send results based on detection state
                        is_detecting = result.get("is_detecting", True)
                        has_speech = result.get("has_speech", False)
                        best_score = result.get("best_score", 0.0)
                        detection_confidence = result.get("detection_confidence", 0.0)
                        active_ayah = result.get("active_ayah_idx")

                        # Enhanced debug logging (Fix 5)
                        mode_str = result.get("mode", "unknown")
                        silence_sec = result.get("silence_sec", 0.0)
                        switch_candidate = result.get("switch_candidate")
                        switch_streak = result.get("switch_streak", 0)
                        current_score = result.get("current_score", 0.0)

                        logger.info(f"Detection state: mode={mode_str}, is_detecting={is_detecting}, "
                                   f"has_speech={has_speech}, best_score={best_score:.3f}, "
                                   f"current_score={current_score:.3f}, active_ayah={active_ayah}, "
                                   f"silence_sec={silence_sec:.1f}, switch_candidate={switch_candidate}, "
                                   f"switch_streak={switch_streak}")

                        # Send WebSocket messages (await each one)
                        ws_start = time.time()

                        if not is_detecting and result.get("active_ayah_idx") is not None:
                            # Ayah detected - send detection info with debug fields (Fix 5)
                            await websocket.send_json({
                                "type": "detection",
                                "ayah_idx": result["active_ayah_idx"],
                                "confidence": result.get("detection_confidence", 0.0),
                                # Debug fields
                                "mode": mode_str,
                                "silence_sec": round(silence_sec, 2),
                                "best_ayah": result.get("best_ayah"),
                                "best_score": round(best_score, 3),
                                "current_score": round(current_score, 3),
                                "switch_candidate": switch_candidate,
                                "switch_streak": switch_streak,
                            })

                            # Get hints for wrong slots
                            hints = get_hint_map_for_active(session.session, FATIHA_AYAT)

                            # Send tracking info
                            await websocket.send_json({
                                "type": "tracking",
                                "wrong_slots": list(result.get("wrong_slots", set())),
                                "uncertain_slots": list(result.get("uncertain_slots", set())),
                                "hints": hints,
                            })

                            new_status = "detected"
                        elif has_speech:
                            new_status = "detecting"
                        else:
                            new_status = "waiting"

                        if new_status != session.last_status:
                            await websocket.send_json({
                                "type": "status",
                                "status": new_status,
                                "mode": mode_str,
                            })

                        ws_elapsed = time.time() - ws_start
                        logger.info(f"WebSocket send took {ws_elapsed*1000:.0f}ms")

                        session.last_status = new_status

                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    finally:
                        session.inference_running = False

                # Create and store task reference (don't await, let it run independently)
                session.infer_task = asyncio.create_task(run_inference_task())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Only try to close if the WebSocket is still connected
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            # WebSocket already closed, ignore
            pass


def run_inference(model, audio, settings, phon_ref=None, ayah_idx=None):
    """Run model inference on audio.

    Args:
        model: The Muaalem model
        audio: Audio waveform
        settings: Moshaf settings
        phon_ref: Precomputed phonetizer reference (preferred, avoids recomputing)
        ayah_idx: Index of detected ayah (fallback if phon_ref not provided)

    Returns:
        Predicted phoneme text
    """
    phonetizer_out = None

    # Use precomputed reference if provided (Fix 2)
    if phon_ref is not None and phon_ref != "":
        phonetizer_out = phon_ref
    elif PHON_REF_BY_AYAH and ayah_idx is not None and 0 <= ayah_idx < len(PHON_REF_BY_AYAH):
        # Use precomputed reference for this ayah (skip if empty/failed)
        ref = PHON_REF_BY_AYAH[ayah_idx]
        if ref and ref != "":
            phonetizer_out = ref

    # If no valid ref yet, find first valid ref
    if phonetizer_out is None and PHON_REF_BY_AYAH:
        for ref in PHON_REF_BY_AYAH:
            if ref and ref != "":
                phonetizer_out = ref
                break

    # Fallback: compute on the fly (only if refs not initialized or all failed)
    if phonetizer_out is None:
        from quran_transcript import quran_phonetizer
        if ayah_idx is not None and 0 <= ayah_idx < len(FATIHA_AYAT):
            ayah_text = FATIHA_AYAT[ayah_idx]
        else:
            ayah_text = FATIHA_AYAT[0]
        phonetizer_out = quran_phonetizer(ayah_text, settings, remove_spaces=True)

    # Run model
    outs = model(
        [audio],
        [phonetizer_out],
        sampling_rate=SAMPLING_RATE,
    )

    return outs[0].phonemes.text


# =============================================================================
# Static Files (Serve Frontend)
# =============================================================================

# Serve frontend build
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend files."""
        file_path = frontend_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_path / "index.html")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
