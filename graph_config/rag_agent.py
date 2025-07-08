from retriever.load_vector_store import load_vector_store
from langchain_groq import ChatGroq

RAG_PROMPT = """
You are a medical assistant. Use the provided context from a medical textbook to answer the user's question. Only use the information in the context. Always cite the source and page if available. If unsure, say you don't know and recommend consulting a healthcare professional.

Context:
{context}

User question: {query}

Answer:
"""

def answer_with_rag(query: str) -> dict:
    retriever = load_vector_store()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = RAG_PROMPT.format(context=context, query=query)
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content
    return {"answer": answer.strip()} 