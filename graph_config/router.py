from langchain_groq import ChatGroq

ROUTER_PROMPT = """
You are a medical query router. Classify the user's query into one of the following categories:
- symptom_analysis: If the user describes symptoms or asks about possible conditions.
- context_check: If the user asks a general medical question that may be answered from a medical book.
- home_remedy: If the user asks for natural or home remedies.
- medication: If the user asks for medication recommendations, dosages, or side effects.
- web_search: If the query is out-of-context or not covered above.

Respond with only the category name.
Query: {query}
Category:
"""

def route_query(query: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ROUTER_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    category = response.strip().split("\n")[0].lower()
    valid = ["symptom_analysis", "context_check", "home_remedy", "medication", "web_search"]
    if category not in valid:
        return "web_search"
    return category 