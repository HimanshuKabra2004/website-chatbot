import os
import pickle
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """
    Responsible for:
    - Generating embeddings from text chunks
    - Storing embeddings in FAISS
    - Persisting and loading vector index
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "vectorstore/faiss.index",
        metadata_path: str = "vectorstore/metadata.pkl"
    ):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def create_and_store(self, chunks: List[Dict]):
        """
        Creates embeddings from chunks and stores them in FAISS.
        """

        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        texts = [chunk["text"] for chunk in chunks]
        self.metadata = [chunk["metadata"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self._persist()

    def _persist(self):
        """
        Saves FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """
        Loads FAISS index and metadata from disk.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("FAISS index or metadata not found.")

        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 5):
        """
        Searches FAISS index for relevant chunks.
        Returns metadata indices.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        return indices[0], distances[0]
