from qa.retriever import Retriever
from qa.promt import build_prompt
from qa.llm import LLMClient


class QAPipeline:
    """
    End-to-end QA pipeline:
    Question → Retrieval → Prompt → LLM → Answer
    """

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMClient()

    def answer(self, question: str) -> str:
        """
        Returns final grounded answer for the user question.
        """

        retrieved_chunks = self.retriever.retrieve(question)

        # If nothing relevant is found
        if not retrieved_chunks:
            return "The answer is not available on the provided website."

        # Build context from retrieved chunks (IMPORTANT)
        context_parts = []
        for item in retrieved_chunks:
            text = item["text"]
            meta = item["metadata"]

            context_parts.append(
                f"""
Source: {meta.get('source')}
Title: {meta.get('title')}

{text}
"""
            )

        context = "\n\n".join(context_parts)

        # Call LLM safely
        answer = self.llm.generate_answer(
            context=context,
            question=question
        )

        return answer
