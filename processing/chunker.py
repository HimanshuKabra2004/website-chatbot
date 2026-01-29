from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Splits cleaned website text into semantic chunks
    with configurable size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def create_chunks(
        self,
        text: str,
        source_url: str,
        page_title: str
    ) -> List[Dict]:
        """
        Returns a list of chunks with metadata.

        Each chunk format:
        {
            "text": "...",
            "metadata": {
                "source": "...",
                "title": "..."
            }
        }
        """

        if not text or len(text.strip()) < 200:
            raise ValueError("Text is too short for chunking.")

        split_texts = self.splitter.split_text(text)

        chunks = []
        for idx, chunk in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk.strip(),
                    "metadata": {
                        "source": source_url,
                        "title": page_title,
                        "chunk_id": idx
                    }
                }
            )

        if not chunks:
            raise RuntimeError("No chunks were created from the text.")

        return chunks
