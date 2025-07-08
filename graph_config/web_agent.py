import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEB_PROMPT = """
You are a medical assistant. Use the following web search results to answer the user's question. Only use information from the results. Always cite the source URLs. If unsure, say you don't know .

Web results:
{results}

User question: {query}

Answer:
"""

def fetch_web_results(query: str) -> list:
    search = TavilySearchResults()
    results = search.invoke(query)
    # TavilySearchAPIWrapper returns a list of dicts with 'title', 'content', 'url'
    # Map 'content' to 'snippet' for compatibility
    formatted = []
    for r in results:
        formatted.append({
            'title': r.get('title', ''),
            'snippet': r.get('content', ''),
            'url': r.get('url', '')
        })
    return formatted

def answer_with_web(query: str) -> dict:
    results = fetch_web_results(query)
    if not results:
        return {
            "answer": "Sorry, I couldn't find relevant information online. Please consult a healthcare professional."
        }
    formatted = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')} (URL: {r.get('url', r.get('link', ''))})" for r in results])
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = WEB_PROMPT.format(results=formatted, query=query)
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content
    return {"answer": answer.strip()} 