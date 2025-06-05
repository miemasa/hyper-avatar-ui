# utils/add_to_faiss.py
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True)          # 追加したい PDF
parser.add_argument("--dst", default="hyponet_db")   # 既存ストア
args = parser.parse_args()

emb  = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vs   = FAISS.load_local(args.dst, emb, allow_dangerous_deserialization=True)
docs = PyPDFLoader(args.pdf).load()                  # PDF をページ単位でロード
vs.add_documents(docs)                              # 追加
vs.save_local(args.dst)                             # 上書き保存
print(f"✅ {args.pdf} を {args.dst} に追加しました")

