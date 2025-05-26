# hyper-avatar-ui

リポジトリのルートには短い README のみが含まれています。

# hyper-avatar-ui
主な機能はapi/、Seed-VC音声変換コードをバンドルしたフォルダにあります。READMEには、プロジェクトの目標とインストール手順が記載されています。

Currently released model supports *zero-shot voice conversion* ...  
## Installation📥
Suggested python 3.10 on Windows, Mac M Series (Apple Silicon) or Linux.
Windows and Linux:
```bash
pip install -r requirements.txt

seedvc_service.py/convertエンド/transcribeポイントとオプションの API キー認証を備えた FastAPI サーバーを公開します。

#  Seed-VC + Faster-Whisper FastAPI server  (API-KEY / resample fix)
API_KEY_ENV = "SEEDVC_API_KEY"               # 環境変数名
...
@app.post("/convert")
async def convert(
        src: UploadFile,
        model: str = Form(...),
        x_api_key: str | None = Header(None, alias="X-API-KEY")):
    """音声を Seed-VC で変換し WAV を返す"""

音声変換ロジックは に実装されていますseed_vc_wrapper.py。このconvert_voiceメソッドは、オーディオの読み込み、特徴抽出、推論を処理します。

def convert_voice(self, source, target, diffusion_steps=10, length_adjust=1.0,
                  inference_cfg_rate=0.7, f0_condition=False, auto_f0_adjust=True,
                  pitch_shift=0, stream_output=True):
    """Convert both timbre and voice from source to target."""
    ...
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]
    ...

ディレクトリには、マイク入力 → Whisper → ChatGPT → Edge-TTS → Seed-VC を連鎖させるui/Streamlit アプリケーション ( ) が含まれています。voice_bot_app5.py

#  voice_bot_app5.py
#    Browser mic → Whisper(API) → ChatGPT → Edge-TTS → Seed-VC(API)
...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "...")
...
VSTORE = load_vectorstore()

同じファイルの後半では、アプリはオーディオを Seed-VC API に投稿し、変換された音声を再生します。

# ⑤ Seed-VC
r = requests.post(
    f"{API_HOST}/convert",
    headers=auth_headers,
    files={"src": ("voice.wav", open(RAW_WAV, "rb"), "audio/wav")},
    data={"model": model_name},
    timeout=120,
)
...
avatar_slot.image(GIF_TALK[model_name], use_container_width=True)
audio_slot.audio(io.BytesIO(wav_bytes), format="audio/wav", autoplay=True)

環境ファイルはPythonの依存関係を記述します。Streamlit UIの場合：

name: voice-ui
channels:
  - conda-forge        # conda-forge 優先
dependencies:
  - python=3.10
  - streamlit=1.45
  - openai>=1.30
  - langchain-community>=0.3
  - langchain-openai>=0.3
  - edge-tts>=6.1,<7
  - faiss-cpu
  - pypdf
  - ffmpeg
  - pip:
    - edge-tts~=6.1

さらに、リポジトリのルートには、よりシンプルな pip 要件ファイルが存在します。

streamlit==1.45.1
edge-tts>=6.1,<7
openai>=1.30
requests>=2.31
langchain-community
langchain-openai
faiss-cpu
pypdf>=3.9

補助スクリプトにはmk-faiss.py、docs/PDF を FAISS ベクトル ストアにインデックス付けして、検索強化型生成を行う が含まれます。

docs = []
for pdf_path in glob.glob("docs/*.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())           # 1 ページ＝1 Document
...
vect = FAISS.from_documents(chunks, embed)
vect.save_local("hyponet_db")
print(f"✅ {len(chunks)} chunks indexed → hyponet_db/")

全体として、リポジトリは次のものを提供します。

api/seedvc_service.pySeed-VC と Faster-Whisper を実行するFastAPI サーバー ( )。

ui/voice_bot_app5.pyAPI および ChatGPT と対話するStreamlit インターフェース ( )。

でのモデルのトレーニングと評価のためのスクリプトと構成をサポートしますapi/。

Python 依存関係を設定するための環境仕様。

これは、新しい貢献者にとってコードベースを探索するための出発点となるはずです。にapi/README.mdは詳細な使用方法とトレーニング手順が含まれており、UIフォルダには音声変換機能をWebアプリに統合する方法が示されています。
