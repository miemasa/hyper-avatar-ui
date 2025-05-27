#!/usr/bin/env python
# ===============================================================
#  Seed-VC + Faster-Whisper FastAPI server  (API-KEY / resample fix)
# ===============================================================
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pathlib import Path
import tempfile, io, os, sys, struct
import numpy as np, soundfile as sf, librosa, torch

from seed_vc_wrapper import SeedVCWrapper
from faster_whisper   import WhisperModel
from config import SEEDVC_API_KEY

# ─────────── 設定 ──────────────────────────────────────────
API_KEY = (SEEDVC_API_KEY or "").strip()
if not API_KEY:
    print("[WARN] SEEDVC_API_KEY が未設定です。認証せずに受け付けます。",
          file=sys.stderr)

app    = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vc  = SeedVCWrapper(device=device)
asr = WhisperModel("small", device=device.type, compute_type="float32")

MODEL2REF = {
    "sakaguchi_model_v1": "ref/ref_sakaguchi.wav",
    "aoki_model_v1":      "ref/ref_aoki.wav",
    "anton_model_v1":     "ref/ref_anton.wav",
}

# Streamable WAV header with unspecified length
def _wav_header(sample_rate: int, bits: int = 16, channels: int = 1) -> bytes:
    byte_rate   = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size   = 0xFFFFFFFF
    return b"".join([
        b"RIFF",
        struct.pack("<I", data_size + 36),
        b"WAVE",
        b"fmt ",
        struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits),
        b"data",
        struct.pack("<I", data_size),
    ])

# ─────────── 共通：API-KEY チェック ─────────────────────
def _check_key(x_api_key: str | None):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-KEY")

# =========================== VC API ============================
@app.post("/convert")
async def convert(
        src: UploadFile,
        model: str = Form(...),
        x_api_key: str | None = Header(None, alias="X-API-KEY")):
    """音声を Seed-VC で変換し WAV を返す"""
    _check_key(x_api_key)

    # ① 受信 WAV を一時保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(await src.read())
        src_path = Path(f.name)

    # ② Seed-VC 推論（bytes + Tensor チャンク）
    gen = vc.convert_voice(
        source=str(src_path),
        target=MODEL2REF[model],
        diffusion_steps=10,
        length_adjust=0.7,
        inference_cfg_rate=0.7,
        f0_condition=False,
        stream_output=True,
    )

    sr_target = 24_000

    def iter_wav():
        yield _wav_header(sr_target)
        for i, chunk in enumerate(gen, 1):
            raw = chunk[0]

            if torch.is_tensor(raw):
                wav = raw.squeeze(0).cpu().numpy()
            elif isinstance(raw, (bytes, bytearray)):
                try:
                    wav, sr_in = sf.read(io.BytesIO(raw), dtype="float32")
                    if wav.ndim == 2:
                        wav = wav.mean(axis=1)
                    if sr_in != sr_target:
                        try:
                            wav = librosa.resample(wav, orig_sr=sr_in, target_sr=sr_target)
                        except TypeError:
                            wav = librosa.resample(wav, sr_in, sr_target)
                except Exception as e:
                    print(f"[{i:02d}] decode err: {e}", file=sys.stderr)
                    continue
            else:
                print(f"[{i:02d}] skip {type(raw)}", file=sys.stderr)
                continue

            pcm_bytes = (wav * 32768.0).astype("<i2").tobytes()
            yield pcm_bytes

    return StreamingResponse(iter_wav(), media_type="audio/wav")

# =========================== ASR API ===========================
@app.post("/transcribe")
async def transcribe(
        src: UploadFile,
        x_api_key: str | None = Header(None, alias="X-API-KEY")):
    """音声を Faster-Whisper で文字起こし"""
    _check_key(x_api_key)

    wav = await src.read()
    tmp = Path(tempfile.gettempdir()) / "asr_tmp.wav"
    tmp.write_bytes(wav)

    segs, _ = asr.transcribe(tmp.as_posix(),
                             beam_size=2,
                             vad_filter=True)
    return {"text": "".join(s.text for s in segs).strip()}

# ---------------------------------------------------------------
if __name__ == "__main__":
    print(f"★ Seed-VC/Whisper API on :8000  (device={device})",
          file=sys.stderr)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
