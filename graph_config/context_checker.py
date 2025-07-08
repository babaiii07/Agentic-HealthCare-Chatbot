from langchain_groq import ChatGroq

CONTEXT_CHECK_PROMPT = """
You are a context checker for a medical assistant. Decide if the user's question can be answered using a standard medical textbook (in-context) or if it requires web search or external sources (out-of-context).

Respond with only 'in-context' or 'out-of-context'.

User question: {query}
Context:
"""

def check_context(query: str) -> bool:
    llm = ChatGroq(model="gemma2-9b-it")
    prompt = CONTEXT_CHECK_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    answer = response.strip().lower()
    return answer.startswith("in-context") 