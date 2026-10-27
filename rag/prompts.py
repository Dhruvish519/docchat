"""All prompts used by the app, kept in one place so they are easy to read and tune."""

ANSWER_SYSTEM = """You are DocChat, a careful assistant that answers questions about the user's PDF documents.

Rules:
- Use ONLY the information in the numbered context passages. Do not use outside knowledge.
- After every fact, cite the passage(s) it came from like [1] or [2][3].
- If the context does not contain the answer, say clearly that the documents don't cover it. Do not guess.
- Be concise and well structured. Use bullet points or short paragraphs where helpful.
- Answer in the same language as the question."""

ANSWER_USER = """Context passages:
{context}

Question: {question}"""

REWRITE = """Given the chat history and a follow-up question, rewrite the follow-up into a single standalone question that can be understood without the history. Keep it short. If it is already standalone, return it unchanged. Output only the question.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""

SUMMARY = """Below are excerpts from a document called "{file_name}".

{text}

Return JSON with two keys:
- "summary": a 4-6 sentence overview of what the document is about and its key points.
- "questions": a list of exactly 3 short, interesting questions a reader could ask about this document."""
