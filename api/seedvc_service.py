#!/usr/bin/env python
# ===============================================================
#  Seed-VC + Faster-Whisper FastAPI server  (API-KEY / resample fix)
# ===============================================================
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException, Response
from pathlib import Path
import tempfile, io, os, sys
import numpy as np, soundfile as sf, librosa, torch

from seed_vc_wrapper import SeedVCWrapper
from faster_whisper   import WhisperModel

# ─────────── 設定 ──────────────────────────────────────────
API_KEY_ENV = "SEEDVC_API_KEY"               # 環境変数名
API_KEY     = (os.getenv(API_KEY_ENV) or "").strip()
if not API_KEY:
    print(f"[WARN] {API_KEY_ENV} が未設定です。認証せずに受け付けます。",
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
    pcm_list  = []

    for i, chunk in enumerate(gen, 1):
        raw = chunk[0]

        # --- A. Tensor で来た場合 -----------------------------------
        if torch.is_tensor(raw):                         # Tensor [1, N]
            wav = raw.squeeze(0).cpu().numpy()

        # --- B. bytes (= 完全な WAV) で来た場合 ----------------------
        elif isinstance(raw, (bytes, bytearray)):
            try:
                wav, sr_in = sf.read(io.BytesIO(raw), dtype="float32")
                if wav.ndim == 2:                        # stereo→mono
                    wav = wav.mean(axis=1)

                if sr_in != sr_target:                   # リサンプル
                    # librosa 0.9 と 0.10 で API が微妙に異なるので両対応
                    try:  # 0.10 以降 (キーワード専用)
                        wav = librosa.resample(wav,
                                               orig_sr=sr_in,
                                               target_sr=sr_target)
                    except TypeError:  # 0.9 以前 (位置引数可)
                        wav = librosa.resample(wav, sr_in, sr_target)

            except Exception as e:
                print(f"[{i:02d}] decode err: {e}", file=sys.stderr)
                continue

        # --- C. それ以外の型はスキップ ------------------------------
        else:
            print(f"[{i:02d}] skip {type(raw)}", file=sys.stderr)
            continue

        pcm_list.append(wav)

    if not pcm_list:
        raise HTTPException(500, "VC 変換に失敗（波形が得られません）")

    pcm = np.concatenate(pcm_list)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr_target, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")

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
