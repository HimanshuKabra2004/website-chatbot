from typing import List, Dict
from processing.embeddings import EmbeddingStore


class Retriever:
    """
    Retrieves relevant chunks from FAISS vector store
    based on user query.
    """

    def __init__(self, top_k: int = 5, score_threshold: float = 1.2):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.store = EmbeddingStore()
        self.store.load()

    def retrieve(self, query: str) -> List[Dict]:
        """
        Returns relevant chunks' metadata.
        If nothing relevant is found, returns empty list.
        """
        indices, distances = self.store.search(query, self.top_k)

        relevant_chunks = []

        for idx, distance in zip(indices, distances):
            # Lower distance = more similarity (L2)
            if distance <= self.score_threshold:
                relevant_chunks.append(
                    {
                        "metadata": self.store.metadata[idx],
                        "score": float(distance)
                    }
                )

        return relevant_chunks
