#!/usr/bin/env python
# ==============================================================
#  voice_bot_app5.py
#    Browser mic → Whisper(API) → ChatGPT → Edge-TTS → Seed-VC(API)
#    （X-API-KEY ヘッダ付きリクエスト対応）
# ==============================================================

from __future__ import annotations

import asyncio, contextlib, io, os, subprocess, tempfile, threading, time, wave
from pathlib import Path
from time import perf_counter

import edge_tts
import openai
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------------- 基本設定 --------------------------------
st.set_page_config(page_title="ハイパーアバター", page_icon="🎤", layout="centered")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ▼ 追加：Seed-VC 用 API キー（環境変数から取得）
SEEDVC_API_KEY = os.getenv("SEEDVC_API_KEY", "")       # 必須なら空チェックを

# ---------------------- RAG ベクターストア -----------------------
@st.cache_resource(show_spinner="📚 知識ベースをロード中…")
def load_vectorstore():
    return FAISS.load_local(
        "hyponet_db",
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True,
    )

VSTORE = load_vectorstore()

# ---------------------- キャラ設定 -------------------------------
PROMPT_MAP = {
    "ja": {
        "sakaguchi_model_v1": "あなたは22歳のハイパーネットワーク社会研究所の職員 坂口萌々子として300文字以内で答えてください。",
        "aoki_model_v1":      "あなたは62歳のハイパーネットワーク社会研究所の所長 青木栄二として大分弁で300文字以内。",
        "anton_model_v1":     "あなたはRIIZEの21歳メンバー アントンとして400文字以内。",
        "_default":           "音声入力なので校正したうえで回答してください。",
    },
    "en": {"_default": "Answer within 200 words as a staff member of the Hyper Network Society Research Institute."},
}
GIF_TALK   = {m: f"{m.split('_')[0]}_talk.gif"   for m in PROMPT_MAP["ja"] if m != "_default"}
IMG_IDLE   = {m: f"{m.split('_')[0]}_idle.png"   for m in GIF_TALK}
AVATAR_IMG = {m: f"{m.split('_')[0]}_jiburi.png" for m in GIF_TALK}

RAW_WAV    = "input_tmp.wav"
MODEL_NAME = "gpt-4o"
API_HOST   = "http://127.0.0.1:8000"   # FastAPI サーバ

# ---------------------- ヘルパー --------------------------------
def build_system_prompt(model_name: str, lang: str, user_q: str) -> str:
    base = (
        PROMPT_MAP.get(lang, {}).get(model_name)
        or PROMPT_MAP.get(lang, {}).get("_default")
        or PROMPT_MAP["ja"]["_default"]
    )
    docs    = VSTORE.max_marginal_relevance_search(user_q, k=8, lambda_mult=0.5)
    context = "\n\n".join(d.page_content for d in docs)
    return f"{base}\n\nハイパーネットワーク社会研究所の職員や所長は以下の参考情報を《》で引用しながら答えてください。\n\n参考情報:\n{context}"

# ◇ API キーヘッダをまとめて用意
auth_headers = {"X-API-KEY": SEEDVC_API_KEY} if SEEDVC_API_KEY else {}

# ---------------------- UI --------------------------------------
st.sidebar.header("設定")
model_name  = st.sidebar.selectbox("🧑‍💼 声モデル", list(GIF_TALK.keys()))
lang_option = st.sidebar.selectbox("🌐 言語 (auto)", ["auto", "ja", "en", "ko", "zh"])
st.sidebar.image(AVATAR_IMG[model_name], width=140)

st.markdown("<h1 style='text-align:center'>🎤 ハイパーアバター🧠</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

for k in ("processing", "idle_ready"):
    st.session_state.setdefault(k, False)

log_area = st.container()
log_area.subheader("📡 進捗ログ")

avatar_slot = st.empty()
audio_slot  = st.empty()          # ← audio 用プレースホルダ

audio_data = st.audio_input("🎙 ブラウザ録音 (押して話す → Stop)")

# ============================================================== #
#                       メイン処理                               #
# ============================================================== #
if audio_data and not st.session_state.processing:
    st.session_state.processing = True

    # ① 録音ファイル保存
    t0 = perf_counter()
    Path(RAW_WAV).write_bytes(audio_data.getbuffer())
    t1 = perf_counter()
    log_area.info("🎙 録音完了")

    # ② Whisper ASR
    r = requests.post(
        f"{API_HOST}/transcribe",
        headers=auth_headers,                        # ★ 追加
        files={"src": ("in.wav", open(RAW_WAV, "rb"), "audio/wav")},
        timeout=60,
    )
    try:
        user_text     = r.json()["text"]
        detected_lang = "ja"
    except ValueError:
        st.error(r.text[:300])
        st.session_state.processing = False
        st.stop()

    t2 = perf_counter()
    st.text_area("🎤 あなたの発言", user_text, height=80)

    # ③ ChatGPT
    target_lang = lang_option if lang_option != "auto" else detected_lang
    system_msg  = build_system_prompt(model_name, target_lang, user_text)
    reply = openai.OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user",   "content": user_text}],
    ).choices[0].message.content
    t3 = perf_counter()
    st.text_area("🤖 ChatGPT の返答", reply, height=150)

    # ④ Edge-TTS
    voice_map = {"ja": "ja-JP-NanamiNeural", "en": "en-US-JennyNeural",
                 "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural"}
    voice   = voice_map.get(target_lang, "en-US-JennyNeural")
    tmp_ogg = Path(tempfile.gettempdir()) / "edge_tts.ogg"
    asyncio.run(edge_tts.Communicate(reply, voice).save(tmp_ogg))
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_ogg, "-ac", "1", "-ar", "24000", RAW_WAV],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    t4 = perf_counter()
    
    # ⑤ Seed-VC
    r = requests.post(
        f"{API_HOST}/convert",
        headers=auth_headers,                        # ★ 追加
        files={"src": ("voice.wav", open(RAW_WAV, "rb"), "audio/wav")},
        data={"model": model_name},
        timeout=120,
    )
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("audio"):
        st.error(f"Seed-VC エラー ({r.status_code})\n{r.text[:300]}")
        st.session_state.processing = False
        st.stop()

    wav_bytes = r.content
    t5 = perf_counter()

    # ⑥ 処理時間表示
    lat = {
        "録音→保存": round(t1 - t0, 3),
        "音声をテキストへ" : round(t2 - t1, 3),
        "LLM" : round(t3 - t2, 3),
        "テキストを音声": round(t4 - t3, 3),
        "音声変換" : round(t5 - t4, 3),
        "total"   : round(t5 - t0, 3),
    }
    log_area.info(f"⏱️ 処理時間 (秒): {lat}")

    # ⑦ 再生
    with contextlib.closing(wave.open(io.BytesIO(wav_bytes))) as wf:
        duration = wf.getnframes() / wf.getframerate() + 0.2

    avatar_slot.image(GIF_TALK[model_name], use_container_width=True)
    audio_slot.audio(io.BytesIO(wav_bytes), format="audio/wav", autoplay=True)

    # idle 画像に戻す
    def mark_idle():
        time.sleep(duration)
        st.session_state.idle_ready = True
    threading.Thread(target=mark_idle, daemon=True).start()

    st.session_state.processing = False

# ---------------------- idle 描画 -------------------------------
if st.session_state.idle_ready:
    avatar_slot.image(IMG_IDLE[model_name], use_container_width=True)
    st.session_state.idle_ready = False
