import streamlit as st
from graph_config.graph import run_graph

st.set_page_config(page_title="Agentic Healthcare Chatbot", page_icon="🩺", layout="centered")
st.title("🩺 Agentic Healthcare Chatbot")
st.markdown("""
This advanced AI assistant can:
- Answer general medical questions from a medical textbook
- Analyze symptoms and suggest possible conditions
- Provide safe home remedies
- Recommend over-the-counter medications
- Search the web for out-of-context queries

**Disclaimer:** This is not a substitute for professional medical advice. Always consult a licensed healthcare provider.
""")

sample_queries = [
    "I have chest pain and fever",
    "Home remedy for headache",
    "What is hypertension?",
    "Best OTC medicine for allergies",
    "Is COVID-19 airborne?"
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.expander("💡 Sample queries"):
    for q in sample_queries:
        if st.button(q):
            st.session_state["input"] = q

user_input = st.text_input("Enter your health question:", key="input")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        try:
            data = run_graph(user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": data.get("final_response", "Sorry, something went wrong."), "details": data.get("agent_outputs", {})})
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.subheader("Chat History")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    with st.expander("Show agent details"):
        st.json(chat["details"]) 