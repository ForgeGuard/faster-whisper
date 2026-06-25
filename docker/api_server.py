import logging
import os
import secrets
import tempfile

from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from faster_whisper import WhisperModel

LOGGER = logging.getLogger("faster_whisper_api")

MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or None
API_KEY = os.getenv("API_KEY") or None
ENABLE_VAD_FILTER = os.getenv("ENABLE_VAD_FILTER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="faster-whisper OpenAI-compatible API")
auth_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    LOGGER.info(
        "Loading WhisperModel(model=%s, device=%s, compute_type=%s)",
        MODEL_SIZE,
        DEVICE,
        COMPUTE_TYPE,
    )
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> None:
    if API_KEY is None:
        return

    if credentials is None or not secrets.compare_digest(
        credentials.credentials, API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None),
) -> dict:
    del model, temperature
    selected_language = language or DEFAULT_LANGUAGE
    wants_word_timestamps = (
        bool(timestamp_granularities) and "word" in timestamp_granularities
    )

    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        audio_path = tmp_file.name
        tmp_file.write(await file.read())

    try:
        whisper_model = get_model()
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=BEAM_SIZE,
            language=selected_language,
            initial_prompt=prompt,
            vad_filter=ENABLE_VAD_FILTER,
            word_timestamps=wants_word_timestamps,
        )
        segment_list = list(segments)
        transcript = "".join(segment.text for segment in segment_list).strip()

        if response_format == "text":
            return {"text": transcript}
        if response_format in {"json", "verbose_json"}:
            payload = {
                "text": transcript,
                "language": info.language,
                "duration": info.duration,
                "segments": [
                    {
                        "id": idx,
                        "seek": 0,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                    for idx, segment in enumerate(segment_list)
                ],
            }
            return payload

        raise HTTPException(status_code=400, detail="Unsupported response_format")
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            LOGGER.warning("Unable to remove temporary file: %s", audio_path)
