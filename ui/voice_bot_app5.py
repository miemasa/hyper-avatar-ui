#!/usr/bin/env python
# ==============================================================
#  voice_bot_app5.py
#    Browser mic → Whisper(API) → ChatGPT → Edge-TTS → Seed-VC(API)
#    （X-API-KEY ヘッダ付きリクエスト対応）
# ==============================================================

from __future__ import annotations

import asyncio, contextlib, io, os, subprocess, tempfile, threading, time, wave
import base64, mimetypes
from pathlib import Path
from time import perf_counter

import edge_tts
import openai
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, SEEDVC_API_KEY, API_HOST
import uuid

# ---------------------- 基本設定 --------------------------------
st.set_page_config(page_title="HYPER AVATAR", page_icon="🎤", layout="centered")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ――― どこか最上部（import の直後など）に 1 回書く -------------
def _rerun() -> None:
    """Streamlit 1.45 ～ 1.48 と 1.49 以降の差異を吸収するラッパー"""
    if hasattr(st, "rerun"):          # 1.49+
        st.rerun()
    else:                             # ≤1.48
        st.experimental_rerun()

# ---------------------- ログイン認証 ------------------------------
# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

# if not st.session_state.authenticated:
#     st.title("ログイン")
#     user_id = st.text_input("ID")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if user_id == "hyper" and password == "hyper":
#             st.session_state.authenticated = True
#             _rerun()
#         else:
#             st.error("ID またはパスワードが違います")
#     st.stop()

# ---------------------- RAG ベクターストア -----------------------
@st.cache_resource
def load_vectorstore(path: str):
    """Load FAISS vector store with a visible spinner and cache."""
    with st.spinner(f"📚 {path} ロード中…"):
        return FAISS.load_local(
            path,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True,
        )

VSTORE_PATH = {
    "aoki_model_v1": "hyponet_db",
    "sakaguchi_model_v1": "hyponet_db2",
}

def get_vectorstore(model_name: str) -> FAISS:
    path = VSTORE_PATH.get(model_name, "hyponet_db")
    return load_vectorstore(path)

# ---------------------- キャラ設定 -------------------------------
PROMPT_MAP = {
    "ja": {
        "sakaguchi_model_v1": "あなたは18歳の情報科学高校の生徒 坂口萌々子として300文字以内で答えてください。",
        "aoki_model_v1":      "あなたは62歳のハイパーネットワーク社会研究所の所長 青木栄二として大分弁で300文字以内。",
        "anton_model_v1":     "あなたはRIIZEの21歳メンバー アントンとして400文字以内。",
        "_default":           "音声入力なので校正したうえで回答してください。",
    },
    "en": {"_default": "Answer within 200 words as a staff member of the Hyper Network Society Research Institute."},
}
GIF_TALK   = {m: f"{m.split('_')[0]}_talk.gif"   for m in PROMPT_MAP["ja"] if m != "_default"}
IMG_IDLE   = {m: f"{m.split('_')[0]}_idle.png"   for m in GIF_TALK}
AVATAR_IMG = {m: f"{m.split('_')[0]}_jiburi.png" for m in GIF_TALK}

# UI 表示用のモデル名ラベル
DISPLAY_LABELS = {
    "sakaguchi_model_v1": "情報科学高校の坂口さん",
    "aoki_model_v1": "ハイパー研の青木所長",
    "anton_model_v1": "ライズのアントン",
}

RAW_WAV    = "input_tmp.wav"
MODEL_NAME = "gpt-4o"

# ---------------------- ヘルパー --------------------------------
def build_system_prompt(model_name: str, lang: str, user_q: str) -> str:
    base = (
        PROMPT_MAP.get(lang, {}).get(model_name)
        or PROMPT_MAP.get(lang, {}).get("_default")
        or PROMPT_MAP["ja"]["_default"]
    )
    vect = get_vectorstore(model_name)
    docs = vect.max_marginal_relevance_search(user_q, k=8, lambda_mult=0.5)
    context = "\n\n".join(d.page_content for d in docs)

    # ★ 研究所の固定文は削除し、プレーンな参照指示だけにする
    return f"""{base}

以下の参考情報を参照したうえで答えてください。

参考情報:
{context}"""




# ◇ API キーヘッダをまとめて用意
auth_headers = {"X-API-KEY": SEEDVC_API_KEY} if SEEDVC_API_KEY else {}

# ---------------------- UI --------------------------------------
st.sidebar.header("設定")
model_options = list(GIF_TALK.keys())
model_name  = st.sidebar.selectbox(
    "🧑‍💼 誰と話したいですか？",
    model_options,
    index=model_options.index("aoki_model_v1"),
    format_func=lambda x: DISPLAY_LABELS.get(x, x),
    key="model_name",
)
lang_option = st.sidebar.selectbox(
    "🌐 言語 (auto)", ["auto", "ja", "en", "ko", "zh"], key="lang_option"
)
st.sidebar.image(AVATAR_IMG[model_name], width=140)

st.markdown(
    """
    <h1 style='text-align:center; font-family:"Orbitron", sans-serif; color:#03a9f4; text-shadow:0 0 10px #039be5;'>🚀 HYPER AVATAR 🚀</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP&family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*='st-'] {font-family: 'Noto Sans JP', sans-serif;}
      .stApp {background: linear-gradient(135deg, #141E30 0%, #243B55 100%); color: #fff;}
      .stChatInput {
        position: fixed;
        bottom: 1rem;
        left: 0;
        right: 0;
        margin: 0 1rem;
      }
      div[data-testid="stRadio"] > label {display: none;}
      div[data-testid="stRadio"] div[role="radiogroup"] {display: flex; gap: 0.5rem; justify-content: center;}
      div[data-testid="stRadio"] [data-baseweb="radio"] {background: #1e1e1e; border-radius: 20px; padding: 0.2rem 0.8rem; border: 2px solid #03a9f4; color: #03a9f4; font-weight: bold;}
      div[data-testid="stRadio"] [data-baseweb="radio"] input:checked + div {background: #03a9f4; color: #fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

for k in ("processing", "idle_ready", "messages", "input_mode"):
    if k == "messages":
        st.session_state.setdefault(k, [])
    elif k == "input_mode":
        st.session_state.setdefault(k, "text")
    else:
        st.session_state.setdefault(k, False)

st.session_state.setdefault("pending_voice", None)
st.session_state.setdefault("clear_mic", False)
st.session_state.setdefault("mic_key", f"mic-{uuid.uuid4()}")
if st.session_state.clear_mic:
    st.session_state.clear_mic = False
    st.session_state.pop("mic", None)
    st.session_state.mic_key = f"mic-{uuid.uuid4()}"

if "prev_model_name" not in st.session_state:
    st.session_state.prev_model_name = model_name
elif st.session_state.prev_model_name != model_name:
    st.session_state.messages = []
    st.session_state.prev_model_name = model_name

mode = st.radio(
    "入力モード",
    ["⌨️ テキスト", "🎙️ 音声"],
    key="mode_switch",
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.input_mode = "voice" if "音声" in mode else "text"

log_area = st.container()

avatar_slot = st.empty()
audio_slot  = st.empty()          # ← audio 用プレースホルダ

# ============================================================== #
#                       メイン処理                               #
# ============================================================== #

for m in st.session_state.messages:
    avatar = AVATAR_IMG[model_name] if m["role"] == "assistant" else None
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

if st.session_state.pending_voice:
    pv = st.session_state.pending_voice
    reply = pv["reply"]
    target_lang = pv["target_lang"]
    pv_model = pv.get("model_name", model_name)
    with st.spinner("🔊 音声合成中…"):
        voice_map = {
            "ja": "ja-JP-NanamiNeural",
            "en": "en-US-JennyNeural",
            "ko": "ko-KR-SunHiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
        }
        voice = voice_map.get(target_lang, "en-US-JennyNeural")
        tmp_ogg = Path(tempfile.gettempdir()) / "edge_tts.ogg"
        asyncio.run(edge_tts.Communicate(reply, voice).save(tmp_ogg))
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_ogg, "-ac", "1", "-ar", "24000", RAW_WAV],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        r = requests.post(
            f"{API_HOST}/convert",
            headers=auth_headers,
            files={"src": ("voice.wav", open(RAW_WAV, "rb"), "audio/wav")},
            data={"model": pv_model},
            timeout=120,
        )
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("audio"):
            st.error(f"Seed-VC エラー ({r.status_code})\n{r.text[:300]}")
            st.session_state.processing = False
            st.session_state.pending_voice = None
            st.stop()
        wav_bytes = r.content

    with contextlib.closing(wave.open(io.BytesIO(wav_bytes))) as wf:
        duration = wf.getnframes() / wf.getframerate() + 0.2

    avatar_slot.image(GIF_TALK[pv_model], use_container_width=True)
    audio_slot.audio(io.BytesIO(wav_bytes), format="audio/wav", autoplay=True)
    st.components.v1.html(
        "<script>window.scrollTo({top:0,behavior:'smooth'});</script>",
        height=0,
    )

    def mark_idle():
        time.sleep(duration)
        st.session_state.idle_ready = True
    threading.Thread(target=mark_idle, daemon=True).start()

    st.session_state.pending_voice = None
    st.session_state.processing = False

user_text = None
#uploaded_image = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"], key="image")
if not st.session_state.processing:
    if st.session_state.input_mode == "text":
        user_text = st.chat_input("メッセージを入力")
    else:
        if "mic_key" not in st.session_state:
            st.session_state.mic_key = f"mic-{uuid.uuid4()}"

        audio_data = st.audio_input(
            "🎤 ①マイクボタンで録音開始　②もう一度おして録音終了)", key=st.session_state.mic_key
        )

        def clear_recording():
            st.session_state.pop("mic", None)
            st.session_state.mic_key = f"mic-{uuid.uuid4()}"
            st.session_state.processing = False
            st.experimental_rerun()

        if st.session_state.get("mic") is not None:
            st.audio(st.session_state.mic)
            st.button("録音クリア", on_click=clear_recording)
        if audio_data:
            st.session_state.processing = True
            t0 = perf_counter()
            Path(RAW_WAV).write_bytes(audio_data.getbuffer())
            t1 = perf_counter()
            r = requests.post(
                f"{API_HOST}/transcribe",
                headers=auth_headers,
                files={"src": ("in.wav", open(RAW_WAV, "rb"), "audio/wav")},
                timeout=60,
            )
            try:
                user_text = r.json()["text"]
            except ValueError:
                st.error(r.text[:300])
                st.session_state.processing = False
                st.stop()
            log_area.info("🎙 録音完了")
            # Flag to reset audio widget on the next run
            st.session_state.clear_mic = True
            t2 = perf_counter()
            st.session_state.processing = False

if user_text and not st.session_state.processing:
    st.session_state.processing = True
    image_b64 = None
    mime_type = None
    st.session_state.messages.append({"role": "user", "content": user_text})

    target_lang = lang_option if lang_option != "auto" else "ja"
    system_msg  = build_system_prompt(model_name, target_lang, user_text)

    with st.spinner("💭 考え中…"):
        t0 = perf_counter()
        openai_messages = [{"role": "system", "content": system_msg}]
        openai_messages.extend(st.session_state.messages)
        if image_b64 and mime_type:
            openai_messages[-1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        reply = (
            openai.OpenAI(api_key=OPENAI_API_KEY)
            .chat.completions.create(model=MODEL_NAME, messages=openai_messages)
            .choices[0]
            .message.content
        )
        t1 = perf_counter()

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.pending_voice = {
            "reply": reply,
            "target_lang": target_lang,
            "model_name": model_name,
        }
        _rerun()
    st.stop()

# ---------------------- idle 描画 -------------------------------
if st.session_state.idle_ready:
    avatar_slot.image(IMG_IDLE[model_name], use_container_width=True)
    st.session_state.idle_ready = False
