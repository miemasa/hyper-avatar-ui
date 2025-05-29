from __future__ import annotations

from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from config import OPENAI_API_KEY

# Vector store path per model
VSTORE_PATH = {
    # 青木所長は docs/ 由来のインデックス
    "aoki_model_v1": "hyponet_chroma",
    # 坂口さんは docs2/ 由来のインデックス
    "sakaguchi_model_v1": "hyponet_chroma2",
    # アントンは参照データなし
    "anton_model_v1": None,
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


class _EmptyRetriever:
    """Retriever that returns no documents."""

    def invoke(self, query: str):  # noqa: D401
        """Return an empty list regardless of query."""
        return []


def get_retriever(model_name: str):
    """Return retriever with MMR search and Cohere rerank."""
    path = VSTORE_PATH.get(model_name)
    if not path:
        return _EmptyRetriever()
    vect = _load_vectorstore(path)
    base = vect.as_retriever(search_type="mmr", k=30)
    rerank = CohereRerank(top_n=8)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=rerank)
