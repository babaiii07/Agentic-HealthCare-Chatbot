from langchain_groq import ChatGroq

SYMPTOM_PROMPT = """
You are a cautious medical assistant. Given the user's symptoms, suggest possible common conditions (not a diagnosis), general advice, relevant over-the-counter medications, and safe home remedies.

User symptoms: {query}

Possible conditions:
Advice:
Medication:
Home remedies:
"""

def analyze_symptoms(query: str) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = SYMPTOM_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    # Parse response into sections
    result = {"conditions": "", "advice": "", "medication": "", "home_remedies": ""}
    lines = response.split("\n")
    section = None
    for line in lines:
        if line.lower().startswith("possible conditions"):
            section = "conditions"
            continue
        elif line.lower().startswith("advice"):
            section = "advice"
            continue
        elif line.lower().startswith("medication"):
            section = "medication"
            continue
        elif line.lower().startswith("home remedies"):
            section = "home_remedies"
            continue
        if section and line.strip():
            result[section] += line.strip() + " "
    for k in result:
        result[k] = result[k].strip()
    return result 