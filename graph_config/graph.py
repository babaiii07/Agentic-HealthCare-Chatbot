from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from graph_config.router import route_query
from graph_config.symptom_analyzer import analyze_symptoms
from graph_config.context_checker import check_context
from graph_config.rag_agent import answer_with_rag
from graph_config.web_agent import answer_with_web
from graph_config.home_remedy_agent import get_home_remedy
from graph_config.medication_agent import get_medication
from graph_config.synthesizer import synthesize_response

# Define the workflow state
ChatState = dict

def debug_merge_state(fn, key):
    def wrapped(state):
        print(f"[DEBUG] Before {key}: {state}")
        result = fn(state["query"])
        new_state = {**state, key: result, "query": state["query"]}
        print(f"[DEBUG] After {key}: {new_state}")
        return new_state
    return wrapped

def synthesizer_debug_node(state):
    print(f"[DEBUG] Before synthesizer: {state}")
    new_state = {**state, "final_response": synthesize_response(state, state["query"])}
    print(f"[DEBUG] After synthesizer: {new_state}")
    return new_state

router_node = RunnableLambda(debug_merge_state(route_query, "route"))
symptom_node = RunnableLambda(debug_merge_state(analyze_symptoms, "symptom_analysis"))
context_check_node = RunnableLambda(debug_merge_state(check_context, "in_context"))
rag_node = RunnableLambda(debug_merge_state(answer_with_rag, "rag"))
web_node = RunnableLambda(debug_merge_state(answer_with_web, "web"))
home_remedy_node = RunnableLambda(debug_merge_state(get_home_remedy, "home_remedy"))
medication_node = RunnableLambda(debug_merge_state(get_medication, "medication"))
synthesizer_node = RunnableLambda(synthesizer_debug_node)

def router_branch(state):
    route = state["route"]
    if route == "symptom_analysis":
        return "symptom_analysis"
    elif route == "context_check":
        return "context_check"
    elif route == "home_remedy":
        return "home_remedy"
    elif route == "medication":
        return "medication"
    else:
        return "web"

def context_branch(state):
    return "rag" if state["in_context"] else "web"

def build_workflow():
    g = StateGraph(ChatState)
    g.add_node("router", router_node)
    g.add_node("symptom_analysis", symptom_node)
    g.add_node("context_check", context_check_node)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("home_remedy", home_remedy_node)
    g.add_node("medication", medication_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        router_branch,
        {
            "symptom_analysis": "symptom_analysis",
            "context_check": "context_check",
            "home_remedy": "home_remedy",
            "medication": "medication",
            "web": "web"
        }
    )
    g.add_conditional_edges(
        "context_check",
        context_branch,
        {
            "rag": "rag",
            "web": "web"
        }
    )
    # All agent nodes go to synthesizer
    g.add_edge("symptom_analysis", "synthesizer")
    g.add_edge("rag", "synthesizer")
    g.add_edge("web", "synthesizer")
    g.add_edge("home_remedy", "synthesizer")
    g.add_edge("medication", "synthesizer")
    g.set_finish_point("synthesizer")
    return g.compile()

workflow = build_workflow()

def run_graph(query: str) -> dict:
    # Defensive: ensure query is a non-empty string
    if not isinstance(query, str) or not query.strip():
        print(f"[ERROR] run_graph received invalid input: {query} ({type(query)})")
        return {"error": "Input to run_graph must be a non-empty string."}
    state = ChatState({"query": query})
    print(f"[DEBUG] Initial state: {state}")
    result = workflow.invoke(state)
    # Remove the query from the output for clarity
    result.pop("query", None)
    return {"agent_outputs": {k: v for k, v in result.items() if k != "final_response"}, "final_response": result.get("final_response", "")} 