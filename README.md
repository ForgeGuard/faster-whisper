[![CI](https://github.com/SYSTRAN/faster-whisper/workflows/CI/badge.svg)](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI)
[![PyPI version](https://badge.fury.io/py/faster-whisper.svg)](https://badge.fury.io/py/faster-whisper)

# Faster Whisper transcription with CTranslate2

**faster-whisper** is a fast, memory-efficient reimplementation of OpenAI's Whisper speech-to-text models using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), an optimized inference engine for Transformer models.

Compared with [openai/whisper](https://github.com/openai/whisper), faster-whisper can be up to 4x faster for similar accuracy while using less memory. It also supports 8-bit quantization on CPU and GPU for further memory and throughput improvements.

This fork keeps the upstream Python package behavior and adds a containerized, OpenAI-compatible transcription API for deployment with Docker, Portainer, Open WebUI, and other OpenAI-style clients.

## Contents

- [What this fork adds](#what-this-fork-adds)
- [Quick start: Python library](#quick-start-python-library)
- [Quick start: containerized API](#quick-start-containerized-api)
- [Installation](#installation)
- [Requirements and compatibility](#requirements-and-compatibility)
- [Model reference](#model-reference)
- [Python API usage](#python-api-usage)
- [OpenAI-compatible HTTP API](#openai-compatible-http-api)
- [Docker, Portainer, and Open WebUI deployment](#docker-portainer-and-open-webui-deployment)
- [Benchmarks](#benchmarks)
- [Model conversion](#model-conversion)
- [Development and validation](#development-and-validation)
- [Community integrations](#community-integrations)

## What this fork adds

This branch differs from upstream [`SYSTRAN/faster-whisper`](https://github.com/SYSTRAN/faster-whisper) by adding deployment assets around the existing library:

| Area | Files | Summary |
| --- | --- | --- |
| OpenAI-compatible API server | [`docker/api_server.py`](docker/api_server.py) | FastAPI service exposing `POST /v1/audio/transcriptions` and `GET /healthz`. |
| CUDA container image | [`docker/Dockerfile`](docker/Dockerfile) | CUDA 12.8 + cuDNN runtime image that installs faster-whisper, FastAPI, Uvicorn, and multipart upload support. |
| Docker build optimization | [`.dockerignore`](.dockerignore) | Keeps local Git, caches, tests, and other unnecessary files out of Docker build contexts. |
| GHCR publishing workflow | [`.github/workflows/docker-image.yml`](.github/workflows/docker-image.yml) | Builds, smoke-tests, and publishes Docker images to GitHub Container Registry. |
| Portainer / Compose example | [`deploy/docker-compose.portainer.yml`](deploy/docker-compose.portainer.yml) | Ready-to-adapt GPU deployment stack with persistent Hugging Face model cache. |
| README overhaul | [`README.md`](README.md) | Documents both the upstream Python library and this fork's container/API deployment path. |

No core transcription package code is changed in this fork; the additional functionality is packaging and service-layer deployment around the existing `faster_whisper` package.

## Quick start: Python library

Install the package from PyPI:

```bash
pip install faster-whisper
```

Transcribe an audio file:

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(f"Detected language: {info.language} ({info.language_probability:.3f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

`segments` is a generator. Transcription starts when the generator is consumed:

```python
segments, info = model.transcribe("audio.mp3")
segments = list(segments)
```

## Quick start: containerized API

Build the API image from this repository:

```bash
docker build -t faster-whisper-api:local -f docker/Dockerfile .
```

Run with an NVIDIA GPU:

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_SIZE=large-v3 \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  -e BEAM_SIZE=5 \
  -e ENABLE_VAD_FILTER=true \
  faster-whisper-api:local
```

Run in CPU mode for smoke tests or small workloads:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -e MODEL_SIZE=tiny \
  -e DEVICE=cpu \
  -e COMPUTE_TYPE=int8 \
  faster-whisper-api:local
```

Check health and submit a transcription request:

```bash
curl -fsS http://127.0.0.1:8000/healthz

curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Installation

### PyPI

```bash
pip install faster-whisper
```

### Development checkout

```bash
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper
pip install -e .[dev]
```

For this fork, replace the clone URL with the fork URL when working on the container/API additions.

### Install from Git

Install the upstream master branch:

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

Install a specific commit:

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

### Optional extras

```bash
pip install "faster-whisper[conversion]"
```

The `conversion` extra installs dependencies needed for model conversion workflows.

## Requirements and compatibility

### Python and audio decoding

- Python 3.9 or newer.
- The package depends on [PyAV](https://github.com/PyAV-Org/PyAV), which bundles FFmpeg libraries. A system FFmpeg installation is not required for the library.
- Common audio containers supported by FFmpeg/PyAV, such as WAV, MP3, M4A/AAC, FLAC, and OGG/Opus, can be decoded.

### GPU requirements

GPU execution requires NVIDIA libraries compatible with the installed CTranslate2 version:

- cuBLAS for CUDA 12
- cuDNN 9 for CUDA 12

Current `ctranslate2` releases target CUDA 12 and cuDNN 9. If you must run older CUDA/cuDNN stacks, pin `ctranslate2` accordingly:

| CUDA / cuDNN stack | Suggested CTranslate2 workaround |
| --- | --- |
| CUDA 12 + cuDNN 9 | Current `ctranslate2>=4.0,<5` from this project. |
| CUDA 12 + cuDNN 8 | `pip install --force-reinstall ctranslate2==4.4.0` |
| CUDA 11 + cuDNN 8 | `pip install --force-reinstall ctranslate2==3.24.0` |

### Container runtime baseline

This fork's Docker image uses:

```text
nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
```

This gives a modern CUDA 12 runtime, cuDNN support, and compatibility with recent NVIDIA GPUs while continuing to work with CUDA 12-capable Ampere-class devices.

Before running the GPU container, verify the host driver and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

### Non-Docker Linux GPU library install

On Linux, CUDA user-space libraries can also be installed with pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

### Compute type guidance

| Device | Common `compute_type` values | Notes |
| --- | --- | --- |
| NVIDIA GPU | `float16`, `int8_float16`, `int8` | `float16` is a strong default for quality and speed; quantized modes reduce VRAM. |
| CPU | `int8`, `float32` | `int8` usually gives the best CPU memory/speed tradeoff. |

## Model reference

Model aliases are resolved by `faster_whisper.utils.available_models()` and map to CTranslate2-converted models on the Hugging Face Hub.

| Alias | Hugging Face repository | Notes |
| --- | --- | --- |
| `tiny.en` | `Systran/faster-whisper-tiny.en` | English-only, smallest footprint. |
| `tiny` | `Systran/faster-whisper-tiny` | Multilingual, smallest footprint. |
| `base.en` | `Systran/faster-whisper-base.en` | English-only base model. |
| `base` | `Systran/faster-whisper-base` | Multilingual base model. |
| `small.en` | `Systran/faster-whisper-small.en` | English-only small model. |
| `small` | `Systran/faster-whisper-small` | Multilingual small model. |
| `medium.en` | `Systran/faster-whisper-medium.en` | English-only medium model. |
| `medium` | `Systran/faster-whisper-medium` | Multilingual medium model. |
| `large-v1` | `Systran/faster-whisper-large-v1` | Large Whisper v1. |
| `large-v2` | `Systran/faster-whisper-large-v2` | Large Whisper v2. |
| `large-v3` | `Systran/faster-whisper-large-v3` | Large Whisper v3. |
| `large` | `Systran/faster-whisper-large-v3` | Alias for `large-v3`. |
| `distil-small.en` | `Systran/faster-distil-whisper-small.en` | Distil-Whisper English small. |
| `distil-medium.en` | `Systran/faster-distil-whisper-medium.en` | Distil-Whisper English medium. |
| `distil-large-v2` | `Systran/faster-distil-whisper-large-v2` | Distil-Whisper large v2. |
| `distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | Distil-Whisper large v3. |
| `distil-large-v3.5` | `distil-whisper/distil-large-v3.5-ct2` | Distil-Whisper large v3.5 CT2 model. |
| `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | Turbo model alias. |
| `turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | Alias for `large-v3-turbo`. |

You can also pass any local CTranslate2 model directory or Hugging Face model ID directly:

```python
model = WhisperModel("username/my-whisper-ct2-model")
model = WhisperModel("/models/whisper-large-v3-ct2")
```

Models are cached by Hugging Face tooling, typically under `~/.cache/huggingface`. The Docker Compose example mounts that cache as a named volume to avoid downloading models on every container restart.

## Python API usage

### Standard transcription

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(info.language, info.language_probability, info.duration)
for segment in segments:
    print(segment.start, segment.end, segment.text)
```

### Batched transcription

`BatchedInferencePipeline.transcribe` is a drop-in replacement for `WhisperModel.transcribe` that can improve throughput when memory allows larger batches.

```python
from faster_whisper import BatchedInferencePipeline, WhisperModel

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

Increase `batch_size` for throughput if you have enough VRAM/RAM. Lower it if you see out-of-memory errors or if latency for single short files matters more than throughput.

### Distil-Whisper

Distil-Whisper checkpoints are compatible with faster-whisper. For `distil-large-v3`, disabling conditioning on previous text is commonly recommended:

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

See the [distil-large-v3 model card](https://huggingface.co/distil-whisper/distil-large-v3) for model-specific guidance.

### Word-level timestamps

```python
segments, info = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
```

### Voice activity detection (VAD)

The library integrates [Silero VAD](https://github.com/snakers4/silero-vad) to remove non-speech regions before transcription:

```python
segments, info = model.transcribe("audio.mp3", vad_filter=True)
```

Customize VAD with `vad_parameters`:

```python
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters={
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 300,
    },
)
```

Important defaults from `VadOptions`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.5` | Speech probability threshold. |
| `neg_threshold` | `None` | Silence threshold; defaults to `max(threshold - 0.15, 0.01)`. |
| `min_speech_duration_ms` | `0` | Drop speech chunks shorter than this. |
| `max_speech_duration_s` | `inf` | Split very long speech chunks when needed. |
| `min_silence_duration_ms` | `2000` | Silence duration needed to separate chunks. |
| `speech_pad_ms` | `400` | Padding added around detected speech chunks. |
| `min_silence_at_max_speech` | `98` | Silence used when splitting at max speech duration. |
| `use_max_poss_sil_at_max_speech` | `True` | Prefer the longest available silence when splitting long speech. |

VAD is enabled by default in batched transcription. In standard `WhisperModel.transcribe`, it is disabled unless `vad_filter=True` is passed.

### Frequently used transcription parameters

| Parameter | Purpose |
| --- | --- |
| `language` | ISO language code such as `en` or `fr`; auto-detected when omitted. |
| `task` | `transcribe` or `translate`. |
| `beam_size` | Beam-search width. Larger values may improve quality but cost time. |
| `temperature` | Sampling temperature or fallback schedule. |
| `condition_on_previous_text` | Use previous decoded text as context for the next window. Disable to reduce repetition loops. |
| `initial_prompt` | Text or token IDs to seed the first decoding window. |
| `prefix` | Prefix text for the first window. |
| `word_timestamps` | Include word timing information in each segment. |
| `clip_timestamps` | Process only selected time ranges; disables VAD for those ranges. |
| `hotwords` | Hint phrases for the model when no prefix is set. |
| `language_detection_threshold` | Minimum language probability for automatic language detection. |

See [`faster_whisper/transcribe.py`](faster_whisper/transcribe.py) for the full parameter list and return types.

### Logging

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

## OpenAI-compatible HTTP API

The fork's API server is intentionally small and deployment-focused. It loads one `WhisperModel` per process and exposes transcription over HTTP.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Lightweight readiness/liveness check. |
| `POST` | `/v1/audio/transcriptions` | OpenAI-style multipart transcription endpoint. |

### Runtime configuration

Environment variables are read at process startup:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_SIZE` | `large-v3` | Model alias, Hugging Face model ID, or local model path. |
| `DEVICE` | `cuda` | Device passed to `WhisperModel`, usually `cuda` or `cpu`. |
| `COMPUTE_TYPE` | `float16` | Compute precision, such as `float16`, `int8_float16`, `int8`, or `float32`. |
| `BEAM_SIZE` | `5` | Beam size used for every API transcription request. |
| `DEFAULT_LANGUAGE` | unset | Optional default language when the request omits `language`. |
| `ENABLE_VAD_FILTER` | `true` | Enables VAD for API requests when set to `1`, `true`, `yes`, or `on`. |

### Request form fields

| Field | Default | Status | Description |
| --- | --- | --- | --- |
| `file` | required | supported | Audio file upload. |
| `model` | `whisper-1` | accepted for compatibility | Ignored by the server; use `MODEL_SIZE` to select the actual runtime model. |
| `language` | `DEFAULT_LANGUAGE` or auto-detect | supported | Optional language code. |
| `prompt` | unset | supported | Passed as `initial_prompt`. |
| `response_format` | `json` | supported values: `json`, `verbose_json`, `text` | Controls response shape. `text` currently returns a JSON object containing `text`. |
| `temperature` | `0.0` | accepted, currently ignored | Present for OpenAI client compatibility. |
| `timestamp_granularities` | unset | partially supported | `word` enables word timestamp computation internally, but the current API response includes segment timings only. Use the Python API for word arrays. |

### Response examples

`response_format=json` and `response_format=verbose_json` return JSON like:

```json
{
  "text": "Hello world.",
  "language": "en",
  "duration": 3.2,
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello world."
    }
  ]
}
```

`response_format=text` returns:

```json
{"text": "Hello world."}
```

Unsupported response formats return HTTP 400.

### OpenAI Python client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

with open("sample.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="json",
    )

print(transcript.text)
```

## Docker, Portainer, and Open WebUI deployment

### Local build

```bash
docker build -t faster-whisper-api:local -f docker/Dockerfile .
```

### GHCR image publishing

[`docker-image.yml`](.github/workflows/docker-image.yml) builds and smoke-tests the image on `master`, tags, and manual workflow dispatches. The publish job uses:

```text
ghcr.io/${{ github.repository }}
```

For this fork, that resolves to the repository's GHCR package, typically lowercased by Docker tooling, such as:

```text
ghcr.io/forgeguard/faster-whisper:latest
```

Published tags include:

- `latest` on the default branch
- branch names
- version tags such as `v1.2.3`
- short commit SHA tags

### Docker Compose / Portainer stack

Reference stack: [`deploy/docker-compose.portainer.yml`](deploy/docker-compose.portainer.yml)

```yaml
version: "3.9"

services:
  faster-whisper-api:
    image: ghcr.io/forgeguard/faster-whisper:latest
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

- Adjust `image` to the registry path produced by your fork or local build.
- The cache volume avoids repeated model downloads.
- `deploy.resources.reservations.devices` is used by Swarm/Portainer-style deployments. Some plain Docker Compose setups may ignore `deploy.*`; use `docker run --gpus all` or Compose GPU support appropriate for your environment.
- For CPU deployments, set `DEVICE=cpu`, choose `COMPUTE_TYPE=int8`, and use a smaller model such as `tiny`, `base`, or `small` unless you have substantial CPU/RAM capacity.

### Open WebUI integration

Use the API as an OpenAI-compatible audio transcription backend:

| Setting | Value |
| --- | --- |
| Base URL | `http://<host>:8000/v1` |
| Transcription endpoint | `/audio/transcriptions` |
| Model value | `whisper-1` |
| Actual model selection | `MODEL_SIZE` environment variable |
| API key | Any placeholder if the client requires one; this server does not validate API keys. |

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

[^1]: `transformers` was out of memory for any batch size greater than 1 in this benchmark setup.

### Running benchmark scripts

Benchmark helpers live in [`benchmark/`](benchmark/):

- [`speed_benchmark.py`](benchmark/speed_benchmark.py)
- [`memory_benchmark.py`](benchmark/memory_benchmark.py)
- [`wer_benchmark.py`](benchmark/wer_benchmark.py)
- [`evaluate_yt_commons.py`](benchmark/evaluate_yt_commons.py)

Install benchmark-specific dependencies first:

```bash
pip install -r benchmark/requirements.benchmark.txt
```

When comparing implementations, keep the settings equivalent:

- Use the same beam size and decoding options.
- Compare systems at similar WER, not only raw speed.
- On CPU, set comparable thread counts:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```

## Model conversion

When loading a built-in model alias such as `WhisperModel("large-v3")`, the corresponding CTranslate2 model is automatically downloaded from the [Hugging Face Hub](https://huggingface.co/Systran).

To convert a Transformers-compatible Whisper model:

```bash
pip install "faster-whisper[conversion]"
ct2-transformers-converter \
  --model openai/whisper-large-v3 \
  --output_dir whisper-large-v3-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

- `--model` accepts a Hugging Face model name or local model directory.
- `--copy_files tokenizer.json preprocessor_config.json` copies tokenizer and preprocessing metadata into the converted model directory.
- If tokenizer files are not copied, the tokenizer configuration is downloaded when the model is loaded later.
- See the [CTranslate2 conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html) for programmatic conversion.

Load a converted model from disk or the Hub:

```python
from faster_whisper import WhisperModel

local_model = WhisperModel("whisper-large-v3-ct2")
hub_model = WhisperModel("username/whisper-large-v3-ct2")
```

## Development and validation

Install the development dependencies:

```bash
pip install -e .[dev]
```

Run tests and formatting checks:

```bash
pytest tests/
black .
isort .
flake8 .
```

The upstream contribution guide is in [`CONTRIBUTING.md`](CONTRIBUTING.md). This fork also validates the Docker image with a CPU-mode smoke test in [`.github/workflows/docker-image.yml`](.github/workflows/docker-image.yml).

## Community integrations

Examples of open-source projects using faster-whisper:

- [speaches](https://github.com/speaches-ai/speaches): OpenAI-compatible server based on faster-whisper.
- [WhisperX](https://github.com/m-bain/whisperX): diarization and accurate word-level timestamps.
- [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2): command-line client compatible with `openai/whisper`.
- [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization): diarization pipeline with NVIDIA NeMo.
- [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win): standalone binaries for Windows, Linux, and macOS.
- [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline): scalable multi-speaker speech-to-text pipeline.
- [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics): transcript and lyrics generation workflow.
- [wscribe](https://github.com/geekodour/wscribe): transcript generation and editor workflow.
- [aTrain](https://github.com/BANDAS-Center/aTrain): GUI transcription and diarization app.
- [Whisper-Streaming](https://github.com/ufal/whisper_streaming): near real-time streaming mode.
- [WhisperLive](https://github.com/collabora/WhisperLive): nearly-live transcription backend.
- [Faster-Whisper-Transcriber](https://github.com/BBC-Esq/ctranslate2-faster-whisper-transcriber): GUI transcription tool.
- [Open-dubbing](https://github.com/softcatala/open-dubbing): AI dubbing workflow.
- [Whisper-FastAPI](https://github.com/heimoshuiyu/whisper-fastapi): simple OpenAI/HomeAssistant/Konele-compatible API backend.

## Roadmap note

Helm chart packaging and a dedicated Helm-focused GitHub Actions workflow are intentionally left for a later phase.
