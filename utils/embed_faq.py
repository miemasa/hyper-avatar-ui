# utils/embed_faq.py
from __future__ import annotations

from dotenv import load_dotenv          # ← .env を読む
load_dotenv()                           # ← .env を環境変数に展開

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

import argparse, sys

# ---------- CLI 引数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="Markdown/TXT file path")
parser.add_argument("--dst", required=True, help="Destination directory for FAISS index")
args = parser.parse_args()

# ---------- API キーチェック ----------
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-") is False:
    sys.exit("❌ OPENAI_API_KEY が設定されていません。 .env または config.py を確認してください。")

# ---------- 埋め込み生成 ----------
docs = TextLoader(args.src).load()
emb  = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vs   = FAISS.from_documents(docs, emb)
vs.save_local(args.dst)

print(f"✅ Saved FAISS index → {args.dst}")
