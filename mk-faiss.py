# mk-faiss.py 例（LangChain の PDFLoader を利用）
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pathlib, os, glob

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

docs = []
for pdf_path in glob.glob("docs/*.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())           # 1 ページ＝1 Document

# チャンク分割（3 – 4 k tokens 程度で）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "、", " "],
)
chunks = splitter.split_documents(docs)

vect = FAISS.from_documents(chunks, embed)
vect.save_local("hyponet_db")
print(f"✅ {len(chunks)} chunks indexed → hyponet_db/")

