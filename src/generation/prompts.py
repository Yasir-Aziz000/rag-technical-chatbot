from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = """You are a precise technical documentation assistant.
Your answers are grounded strictly in the provided context excerpts.

Rules:
- Answer using ONLY information present in the context.
- If the answer is not in the context, respond exactly: "I don't have enough information in the provided documents to answer this."
- Always cite your sources using [Source: <filename>, page <N>] notation at the end of relevant sentences.
- Be concise and technical. Avoid filler phrases.
- For code or commands, use markdown code blocks."""

RAG_HUMAN_PROMPT = """Context excerpts from uploaded documents:
{context}

---

Question: {question}"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", RAG_HUMAN_PROMPT),
    ]
)
