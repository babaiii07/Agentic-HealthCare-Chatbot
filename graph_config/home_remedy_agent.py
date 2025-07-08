from langchain_groq import ChatGroq

REMEDY_PROMPT = """
You are a medical assistant. Provide safe, evidence-based home remedies for the user's condition. Also provide a brief list of common symptoms and safe over-the-counter medications for the condition. Only suggest remedies that are widely accepted and safe. Always include a disclaimer: 'Consult a healthcare professional before trying any remedy.' If unsure, say you don't know. Cite sources if possible.

User request: {query}

Symptoms:
Remedy:
Medication:
Source(s):
"""

def get_home_remedy(query: str) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = REMEDY_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    result = {"symptoms": "", "remedy": "", "medication": "", "sources": ""}
    lines = response.split("\n")
    section = None
    for line in lines:
        if line.lower().startswith("symptoms"):
            section = "symptoms"
            continue
        elif line.lower().startswith("remedy"):
            section = "remedy"
            continue
        elif line.lower().startswith("medication"):
            section = "medication"
            continue
        elif line.lower().startswith("source"):
            section = "sources"
            continue
        if section and line.strip():
            result[section] += line.strip() + " "
    for k in result:
        result[k] = result[k].strip()
    return result 