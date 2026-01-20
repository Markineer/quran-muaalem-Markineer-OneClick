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
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    SAMPLING_RATE,
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


def get_model():
    """Lazy load the Muaalem model."""
    global muaalem_model, moshaf_settings

    if muaalem_model is None:
        try:
            import torch
            from quran_muaalem.inference import Muaalem
            from quran_transcript import MoshafAttributes

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Muaalem model on {device}...")

            muaalem_model = Muaalem(
                model_name_or_path="obadx/muaalem-model-v3_2",
                device=device
            )

            moshaf_settings = MoshafAttributes(
                rewaya="hafs",
                madd_monfasel_len=4,
                madd_mottasel_len=4,
                madd_mottasel_waqf=4,
                madd_aared_len=4,
            )

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
# WebSocket for Real-Time Streaming
# =============================================================================

class RealtimeSession:
    """Manages a real-time harakat training session."""

    def __init__(self):
        self.session = create_session(FATIHA_AYAT)
        self.expected_seqs = get_expected_sequences_for_all_ayat(FATIHA_AYAT)
        self.is_active = False
        self.last_status = "waiting"

    def reset(self):
        self.session = create_session(FATIHA_AYAT)
        self.is_active = False
        self.last_status = "waiting"


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

            # Push to session buffer
            push_audio(session.session, audio_float32)

            # Check if we should run inference
            buffer_duration = get_buffer_duration(session.session)

            if buffer_duration >= 1.0:  # At least 1 second of audio
                try:
                    # Run inference and update
                    result = await asyncio.to_thread(
                        infer_and_update,
                        session.session,
                        lambda audio: run_inference(model, audio, settings),
                        FATIHA_AYAT,
                    )

                    # Send results
                    if result["status"] == "detected":
                        await websocket.send_json({
                            "type": "detection",
                            "ayah_idx": result["active_ayah_idx"],
                            "confidence": result["confidence"],
                        })

                        # Send tracking info
                        await websocket.send_json({
                            "type": "tracking",
                            "wrong_slots": list(result["wrong_slots"]),
                            "uncertain_slots": list(result["uncertain_slots"]),
                            "hints": result["hints"],
                        })

                    elif result["status"] != session.last_status:
                        await websocket.send_json({
                            "type": "status",
                            "status": result["status"],
                        })

                    session.last_status = result["status"]

                except Exception as e:
                    logger.error(f"Inference error: {e}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)


def run_inference(model, audio, settings):
    """Run model inference on audio."""
    from quran_transcript import quran_phonetizer

    # Use first ayah as reference for phonetizer
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
