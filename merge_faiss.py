from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings()

base = FAISS.load_local("hyponet_db", emb, allow_dangerous_deserialization=True)
faq  = FAISS.load_local("faiss_index/faq_form", emb, allow_dangerous_deserialization=True)

base.merge_from(faq)          # ベクトルとメタデータを結合
base.save_local("hyponet_db") # 既存ストアを上書き

print("✅ merged & saved   -> hyponet_db")

