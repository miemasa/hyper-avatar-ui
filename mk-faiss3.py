#!/usr/bin/env python
"""Build Chroma index with OCR."""
from __future__ import annotations

import glob
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from unstructured.partition.pdf import partition_pdf

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=256,
    openai_api_key=OPENAI_API_KEY,
)

docs = []
for pdf_path in glob.glob("docs/*.pdf"):
    elements = partition_pdf(pdf_path, strategy="hi_res")
    for el in elements:
        text = getattr(el, "text", "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": pdf_path}))

vect = Chroma.from_documents(
    docs,
    embed,
    persist_directory="hyponet_chroma",
    collection_metadata={"hnsw:space": "cosine"},
)

vect.persist()
try:
    vect._collection.create_index("bm25")
except Exception:
    pass

print(f"✅ {len(docs)} elements indexed → hyponet_chroma/")
