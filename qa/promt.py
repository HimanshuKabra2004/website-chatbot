SYSTEM_PROMPT = """
You are an AI assistant answering questions strictly
based on the provided website content.

Rules you must follow:
1. Use ONLY the information given in the context.
2. Do NOT use external knowledge.
3. Do NOT guess or hallucinate.
4. If the answer is not present in the context,
   respond exactly with:

"The answer is not available on the provided website."
"""

def build_prompt(context: str, question: str) -> str:
    """
    Builds final prompt for the LLM.
    """
    return f"""
{SYSTEM_PROMPT}

Website Content:
{context}

User Question:
{question}

Answer:
"""
