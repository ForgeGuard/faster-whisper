[![CI](https://github.com/SYSTRAN/faster-whisper/workflows/CI/badge.svg)](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI)
[![PyPI version](https://badge.fury.io/py/faster-whisper.svg)](https://badge.fury.io/py/faster-whisper)

# Faster Whisper transcription with CTranslate2

**faster-whisper** is a reimplementation of OpenAI's Whisper model using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), a fast inference engine for Transformer models.

This implementation is up to 4x faster than [openai/whisper](https://github.com/openai/whisper) for similar accuracy while using less memory. Efficiency can be improved further with 8-bit quantization on CPU and GPU.

## Table of contents

- [Quick start (Python)](#quick-start-python)
- [Containerized OpenAI-compatible API (recommended)](#containerized-openai-compatible-api-recommended)
- [Portainer / docker-compose stack (ready to run)](#portainer--docker-compose-stack-ready-to-run)
- [Open WebUI integration](#open-webui-integration)
- [Requirements and compatibility](#requirements-and-compatibility)
- [Usage (Python API)](#usage-python-api)
- [Benchmarks](#benchmarks)
- [Model conversion](#model-conversion)
- [Comparing performance fairly](#comparing-performance-fairly)
- [Community integrations](#community-integrations)

## Quick start (Python)

Install from [PyPI](https://pypi.org/project/faster-whisper/):

```bash
pip install faster-whisper
```

Minimal example:

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(f"Detected language: {info.language} ({info.language_probability:.3f})")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

`segments` is a generator, so transcription starts when you iterate over it:

```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)
```

<details>
<summary>Other installation methods</summary>

Install the `master` branch:

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

Install a specific commit:

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

</details>

## Containerized OpenAI-compatible API (recommended)

This repository now includes a long-running API server for OpenAI-compatible audio transcription in:

- [`docker/api_server.py`](docker/api_server.py)
- [`docker/Dockerfile`](docker/Dockerfile)

### Build locally

```bash
docker build -t faster-whisper:local -f docker/Dockerfile .
```

### Run locally (GPU)

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_SIZE=large-v3 \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  -e BEAM_SIZE=5 \
  -e ENABLE_VAD_FILTER=true \
  -e API_KEY= \
  faster-whisper:local
```

### Endpoint

- `POST /v1/audio/transcriptions` (OpenAI-compatible form upload)
- `GET /healthz`

Authentication is disabled by default. To require OpenAI-compatible bearer token
authentication for transcription requests, set `API_KEY` to a non-empty value and
send it as `Authorization: Bearer <API_KEY>`. The `/healthz` endpoint always
remains unauthenticated for container health checks.

Example request without authentication:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

Example request with `API_KEY` enabled:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Portainer / docker-compose stack (ready to run)

Reference file: [`deploy/docker-compose.portainer.yml`](deploy/docker-compose.portainer.yml)

Use this stack in Portainer or with `docker compose`:

```yaml
version: "3.9"

services:
  faster-whisper-api:
    image: ghcr.io/systran/faster-whisper:latest
    container_name: faster-whisper-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      MODEL_SIZE: large-v3
      DEVICE: cuda
      COMPUTE_TYPE: float16
      BEAM_SIZE: "5"
      ENABLE_VAD_FILTER: "true"
      DEFAULT_LANGUAGE: ""
      API_KEY: ""
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    volumes:
      - faster-whisper-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  faster-whisper-cache:
```

Notes:
- The compose file uses GHCR images produced by GitHub Actions.
- The cache volume avoids repeated model downloads.
- In vanilla Docker Compose environments, `deploy.*` may be ignored; in Portainer/Swarm it is used for GPU reservations.
- Leave `API_KEY` empty to keep the API unauthenticated, or set it to require
  `Authorization: Bearer <API_KEY>` on transcription requests. `/healthz` does
  not require authentication.

## Open WebUI integration

This API is intended to be consumed by OpenAI-compatible clients such as Open WebUI.

Typical integration values:
- Base URL: `http://<your-host>:8000/v1`
- Endpoint used for speech transcription: `/audio/transcriptions`
- Model value: `whisper-1` (accepted for compatibility; actual runtime model is selected via `MODEL_SIZE`)

## Requirements and compatibility

- Python 3.9+
- For GPU execution: NVIDIA driver + CUDA runtime compatibility on host
- `ctranslate2` current versions target CUDA 12 + cuDNN 9

### Recommended container runtime baseline

The Docker image uses:

- `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`

Why:
- CUDA 12.8 adds Blackwell tooling support (relevant for RTX 50-series).
- Ampere-class GPUs such as NVIDIA A2 remain supported on CUDA 12.x.

### Compatibility checklist

1. Install a host NVIDIA driver that supports CUDA 12.8 containers.
2. Install/configure NVIDIA Container Toolkit (`--gpus all` works).
3. Confirm GPU visibility:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

### Alternative GPU library install (non-Docker Linux)

If running outside Docker, you can install CUDA 12 user-space libraries with pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

## Usage (Python API)

### Batched transcription

```python
from faster_whisper import BatchedInferencePipeline, WhisperModel

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Distil-Whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5,
    language="en",
    condition_on_previous_text=False,
)
```

See the [distil-large-v3 model card](https://huggingface.co/distil-whisper/distil-large-v3).

### Word-level timestamps

```python
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)
for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
```

### VAD filter

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True)
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
```

VAD filter is enabled by default for batched transcription.

### Logging

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

See additional options in [`WhisperModel`](faster_whisper/transcribe.py).

## Benchmarks

Reference setup for transcribing [13 minutes](https://www.youtube.com/watch?v=0u7tTptBo9I) of audio:

- [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/tree/v20240930)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[v1.7.2](https://github.com/ggerganov/whisper.cpp/tree/v1.7.2)
- [transformers](https://github.com/huggingface/transformers)@[v4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/tree/v1.1.0)

### Large-v2 model on GPU

| Implementation | Precision | Beam size | Time | VRAM usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

### distil-whisper-large-v3 model on GPU

| Implementation | Precision | Beam size | Time | YT Commons WER |
| --- | --- | --- | --- | --- |
| transformers (SDPA) (`batch_size=16`) | fp16 | 5 | 46m12s | 14.801 |
| faster-whisper (`batch_size=16`) | fp16 | 5 | 25m50s | 13.527 |

GPU benchmarks were executed with CUDA 12.4 on an NVIDIA RTX 3070 Ti 8GB.

### Small model on CPU

| Implementation | Precision | Beam size | Time | RAM usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 6m58s | 2335MB |
| whisper.cpp | fp32 | 5 | 2m05s | 1049MB |
| whisper.cpp (OpenVINO) | fp32 | 5 | 1m45s | 1642MB |
| faster-whisper | fp32 | 5 | 2m37s | 2257MB |
| faster-whisper (`batch_size=8`) | fp32 | 5 | 1m06s | 4230MB |
| faster-whisper | int8 | 5 | 1m42s | 1477MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 51s | 3608MB |

Executed with 8 threads on an Intel Core i7-12700K.

[^1]: `transformers` OOM for any batch size > 1 in this benchmark setup.

## Model conversion

When loading a model by size (for example `WhisperModel("large-v3")`), the corresponding CTranslate2 model is automatically downloaded from the [Hugging Face Hub](https://huggingface.co/Systran).

To convert Whisper models compatible with Transformers:

```bash
pip install "transformers[torch]>=4.23"
ct2-transformers-converter \
  --model openai/whisper-large-v3 \
  --output_dir whisper-large-v3-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

- `--model` accepts a model name from the Hub or a local path.
- If you skip `--copy_files tokenizer.json`, tokenizer config is downloaded at load time.
- See also the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

### Load a converted model

```python
from faster_whisper import WhisperModel

local_model = WhisperModel("whisper-large-v3-ct2")
hub_model = WhisperModel("username/whisper-large-v3-ct2")
```

## Comparing performance fairly

When comparing against other Whisper implementations:

- Use equivalent transcription settings (especially beam size).
- Compare systems at similar WER.
- On CPU, set equivalent thread counts:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```

## Community integrations

Examples of open-source projects using faster-whisper:

- [speaches](https://github.com/speaches-ai/speaches): OpenAI-compatible server based on `faster-whisper`.
- [WhisperX](https://github.com/m-bain/whisperX): diarization + accurate word-level timestamps.
- [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2): CLI compatible with `openai/whisper`.
- [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization): diarization pipeline with NVIDIA NeMo.
- [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win): standalone binaries for Windows/Linux/macOS.
- [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline): scalable multi-speaker speech-to-text pipeline.
- [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics): transcript and lyrics generation workflow.
- [wscribe](https://github.com/geekodour/wscribe): transcript generation and editor workflow.
- [aTrain](https://github.com/BANDAS-Center/aTrain): GUI-based transcription and diarization app.
- [Whisper-Streaming](https://github.com/ufal/whisper_streaming): near real-time streaming mode.
- [WhisperLive](https://github.com/collabora/WhisperLive): nearly-live transcription backend.

## Future work

Helm chart packaging and a dedicated Helm-focused GitHub Actions workflow are intentionally left for a later phase.
