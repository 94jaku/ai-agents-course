"""Two-agent LangGraph: Tech News Researcher → Summary Reviewer with feedback loop."""

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

MAX_REVISIONS = 3

# ── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    news: str
    summary: str
    review: str
    revision: int
    approved: bool


# ── Shared clients (initialized lazily after env is loaded) ──────────────────

llm = None
tavily = None


def _init_clients():
    global llm, tavily
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        tavily = TavilyClient()


# ── Node: Research Agent ─────────────────────────────────────────────────────

def research_agent(state: State) -> State:
    """Search for recent tech news and write a summary, or revise based on feedback."""
    _init_clients()
    revision = state.get("revision", 0)

    if revision == 0:
        results = tavily.search(
            query="new technology news last week",
            search_depth="basic",
            max_results=5,
            topic="news",
        )
        articles = "\n".join(
            f"- {r['title']}: {r['content'][:200]}" for r in results["results"]
        )
        response = llm.invoke(
            f"You are a tech news researcher. Based on these recent articles, "
            f"write a concise summary (3-5 bullet points) of the most important "
            f"new technology developments from the past week.\n\n{articles}"
        )
        return {"news": articles, "summary": response.content, "revision": 1}
    else:
        response = llm.invoke(
            f"You are a tech news researcher. Revise the following summary "
            f"based on the reviewer's feedback.\n\n"
            f"Original articles:\n{state['news']}\n\n"
            f"Current summary:\n{state['summary']}\n\n"
            f"Reviewer feedback:\n{state['review']}\n\n"
            f"Write an improved summary addressing the feedback."
        )
        return {"summary": response.content, "revision": revision + 1}


# ── Node: Reviewer Agent ────────────────────────────────────────────────────

def reviewer_agent(state: State) -> State:
    """Review the summary — approve or request revisions."""
    _init_clients()
    response = llm.invoke(
        f"You are an editor reviewing a tech news summary.\n\n"
        f"Summary:\n{state['summary']}\n\n"
        f"If the summary is clear, accurate, and complete, respond with "
        f"exactly 'APPROVED' on the first line.\n"
        f"Otherwise, provide 2-3 brief, actionable suggestions to improve it."
    )
    approved = response.content.strip().startswith("APPROVED")
    return {"review": response.content, "approved": approved}


# ── Routing ──────────────────────────────────────────────────────────────────

def should_continue(state: State) -> str:
    if state.get("revision", 0) < 2:
        return "revise"
    if state.get("approved") or state.get("revision", 0) >= MAX_REVISIONS:
        return "end"
    return "revise"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)
    graph.add_node("researcher", research_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "reviewer")
    graph.add_conditional_edges("reviewer", should_continue, {
        "revise": "researcher",
        "end": END,
    })
    return graph.compile()


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    app = build_graph()
    result = app.invoke({"news": "", "summary": "", "review": "", "revision": 0, "approved": False})

    revisions = result["revision"]
    approved = result["approved"]

    print("=" * 60)
    print(f"TECH NEWS SUMMARY (revisions: {revisions}, approved: {approved})")
    print("=" * 60)
    print(result["summary"])
    print("\n" + "=" * 60)
    print("FINAL REVIEWER FEEDBACK")
    print("=" * 60)
    print(result["review"])
