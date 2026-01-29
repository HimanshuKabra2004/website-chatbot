import os
import openai


class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        openai.api_key = api_key

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"""
You are an AI assistant that answers questions strictly based on the provided context.

Rules:
- Use ONLY the information from the context
- If the answer is not present, say:
  "The answer is not available on the provided website."

Context:
{context}

Question:
{question}

Answer:
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message["content"].strip()
