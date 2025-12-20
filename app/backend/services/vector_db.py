"""
Vector Store Manager with Hybrid Search (BM25 + Vector)

Uses Reciprocal Rank Fusion to combine keyword and semantic search results
for more robust retrieval. Discovered need for hybrid search when
"multi agent systems" failed to match "multi-agent systems" in vector-only search.
"""

import chromadb
import os
import uuid
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rank_bm25 import BM25Okapi


class VectorStoreManager:
    def __init__(self):
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))

        # Connect to Chroma Service
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection("pdf_rag_collection")

        # Initialize Embedders
        self.embedder_ingest = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
        )
        self.embedder_query = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY"
        )

        # BM25 index (rebuilt on each query from ChromaDB data)
        self._bm25_index = None
        self._bm25_docs = []

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def _embed_with_retry(self, texts):
        return self.embedder_ingest.embed_documents(texts)

    def add_documents(self, documents: List[Document]) -> bool:
        """
        Embeds and stores documents. Clears existing collection for single-document mode.
        """
        if not documents:
            return False

        # Clear existing collection (single-document mode)
        try:
            self.client.delete_collection("pdf_rag_collection")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection("pdf_rag_collection")

        # Process in Batches
        batch_size = 10
        all_embeddings = []
        all_ids = []
        all_metadatas = []
        all_texts = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch]

            try:
                batch_embeddings = self._embed_with_retry(batch_texts)
                all_embeddings.extend(batch_embeddings)
                all_texts.extend(batch_texts)
                all_ids.extend([str(uuid.uuid4()) for _ in batch])
                all_metadatas.extend([doc.metadata for doc in batch])
                time.sleep(2.0)  # Rate limit
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")
                raise e

        if all_embeddings:
            self.collection.add(
                documents=all_texts,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids,
            )
            # Reset BM25 index so it rebuilds on next query
            self._bm25_index = None
            return True

        return False

    def _build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB."""
        # Get all documents from collection
        all_data = self.collection.get(include=["documents", "metadatas"])

        if not all_data["documents"]:
            self._bm25_docs = []
            self._bm25_index = None
            return

        self._bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]

        # Tokenize documents for BM25
        tokenized_docs = [doc.page_content.lower().split() for doc in self._bm25_docs]
        self._bm25_index = BM25Okapi(tokenized_docs)

    def _bm25_search(self, query: str, k: int = 4) -> List[tuple]:
        """Keyword search using BM25. Returns list of (doc, score)."""
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]

        return [(self._bm25_docs[i], scores[i]) for i in top_indices if scores[i] > 0]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Pure vector similarity search."""
        query_vector = self.embedder_query.embed_query(query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=k)

        if not results or not results["documents"]:
            return []

        found_docs = []
        batch_docs = results["documents"][0]
        batch_metas = results["metadatas"][0]

        for text, meta in zip(batch_docs, batch_metas):
            found_docs.append(Document(page_content=text, metadata=meta))

        return found_docs

    def hybrid_search(
        self, query: str, k: int = 6, vector_weight: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining vector similarity and BM25 keyword search.
        Uses Reciprocal Rank Fusion (RRF) to merge results.

        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector results (0-1), BM25 gets (1-vector_weight)
        """
        # Get results from both methods
        vector_docs = self.similarity_search(query, k=k)
        bm25_results = self._bm25_search(query, k=k)
        bm25_docs = [doc for doc, _ in bm25_results]

        # Reciprocal Rank Fusion
        # Score = sum(1 / (rank + 60)) for each ranking
        doc_scores: Dict[str, tuple] = {}  # content_hash -> (doc, score)

        # Score vector results
        for rank, doc in enumerate(vector_docs):
            content_hash = hash(doc.page_content[:200])
            rrf_score = vector_weight * (1 / (rank + 60))
            if content_hash in doc_scores:
                doc_scores[content_hash] = (
                    doc,
                    doc_scores[content_hash][1] + rrf_score,
                )
            else:
                doc_scores[content_hash] = (doc, rrf_score)

        # Score BM25 results
        bm25_weight = 1 - vector_weight
        for rank, doc in enumerate(bm25_docs):
            content_hash = hash(doc.page_content[:200])
            rrf_score = bm25_weight * (1 / (rank + 60))
            if content_hash in doc_scores:
                doc_scores[content_hash] = (
                    doc,
                    doc_scores[content_hash][1] + rrf_score,
                )
            else:
                doc_scores[content_hash] = (doc, rrf_score)

        # Sort by combined score and return top k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:k]]
