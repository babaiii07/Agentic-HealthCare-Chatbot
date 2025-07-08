from langchain_groq import ChatGroq
from graph_config.web_agent import fetch_web_results
import re

MEDICATION_PROMPT = """
You are a medical assistant. Use the information from the authentic sources below (FDA, NIH, RxList, Mayo Clinic, WebMD, DailyMed) to answer the user's medication question. If no authentic source is present, you may use reputable general medical sources or your own knowledge, If unsure, say you don't know.

User request: {query}

Web results:
{web_results}

Answer in this format (if possible):
- Medication name(s):
- Dosage:
- Usage/Indication:
- Side effects:
"""

def extract_authentic_results(results):
    authentic_domains = [
        r"fda\\.gov", r"nih\\.gov", r"rxlist\\.com", r"mayoclinic\\.org", r"webmd\\.com", r"dailymed\\.nlm\\.nih\\.gov"
    ]
    authentic = []
    for r in results:
        url = r.get('url') or r.get('link') or ''
        if any(re.search(domain, url) for domain in authentic_domains):
            title = r.get('title', '')
            snippet = r.get('snippet', '')
            authentic.append(f"- {title}: {snippet} (URL: {url})")
    return "\n".join(authentic)

def get_medication(query: str) -> dict:
    web_results_list = fetch_web_results(query)
    authentic_results = extract_authentic_results(web_results_list)
    if authentic_results:
        web_results = authentic_results
    else:
        # Fallback: include all web results for LLM context
        web_results = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')} (URL: {r.get('url', r.get('link', ''))})" for r in web_results_list
        ]) or "No relevant web results found."
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = MEDICATION_PROMPT.format(query=query, web_results=web_results)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    # Try to parse the LLM output for structured info
    result = {"symptoms": "", "medication": "", "dosage": "", "side_effects": "", "home_remedies": "", "sources": "", "disclaimer": ""}
    lines = response.split("\n")
    section = None
    for line in lines:
        l = line.lower()
        if l.startswith("medication name") or l.startswith("medication(s)"):
            section = "medication"
            continue
        elif l.startswith("dosage"):
            section = "dosage"
            continue
        elif l.startswith("usage") or l.startswith("indication"):
            section = "symptoms"
            continue
        elif l.startswith("side effects"):
            section = "side_effects"
            continue
        elif l.startswith("source"):
            section = "sources"
            continue
        elif l.startswith("disclaimer"):
            section = "disclaimer"
            continue
        if section and line.strip():
            result[section] += line.strip() + " "
    for k in result:
        result[k] = result[k].strip()
    return result 