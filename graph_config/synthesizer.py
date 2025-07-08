from langchain_groq import ChatGroq

SYNTH_PROMPT = """
You are a medical assistant. Given the outputs from various expert agents (symptom analysis, RAG, web, home remedy, medication), synthesize a clear, user-friendly response. Only answer medical questions. If the question is not medical, respond with: 'I am a medical healthcare chatbot. I don't know.'

User query: {user_query}

Agent outputs:
{agent_outputs}

Final response:
"""

def synthesize_response(agent_outputs: dict, user_query: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = SYNTH_PROMPT.format(agent_outputs=agent_outputs, user_query=user_query)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    return response.strip() 