from __future__ import annotations

from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from config import OPENAI_API_KEY

# Path mapping for vector stores (single store used for now)
VSTORE_PATH = {
    "aoki_model_v1": "hyponet_chroma",
    "sakaguchi_model_v1": "hyponet_chroma",
    "anton_model_v1": "hyponet_chroma",
}


@lru_cache(maxsize=None)
def _load_vectorstore(path: str) -> Chroma:
    """Load persistent Chroma vector store."""
    embed = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=256,
        openai_api_key=OPENAI_API_KEY,
    )
    vect = Chroma(
        persist_directory=path,
        embedding_function=embed,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Ensure BM25 index exists for hybrid search
    try:
        vect._collection.create_index("bm25")
    except Exception:
        pass
    return vect


def get_retriever(model_name: str):
    """Return retriever with MMR search and Cohere rerank."""
    path = VSTORE_PATH.get(model_name, "hyponet_chroma")
    vect = _load_vectorstore(path)
    base = vect.as_retriever(search_type="mmr", k=30)
    rerank = CohereRerank(top_n=8)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=rerank)
