"""
LangGraph Agent with RAG, Citations, and Guardrails.
"""

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from app.backend.services.vector_db import VectorStoreManager
import operator


# ============================================================
# SYSTEM PROMPT - Enforces grounding and citations
# ============================================================
SYSTEM_PROMPT = """You are a document-based AI assistant. You MUST follow these rules strictly:

**CRITICAL: You MUST use the retrieve_documents tool for EVERY question about the document, even if you think you already know the answer from the chat history. Always search first, then answer.**

1. **ONLY answer using the retrieved document content.** Do not use any external knowledge or previous chat responses.

2. **CITATIONS - Add a Sources section at the end:**
   - ONLY cite pages where you actually got information for your answer
   - Do NOT list all retrieved pages - only the ones you used
   - After your answer, add a "**Sources:**" section on a new line
   - List each used page on its own line: `- [DocumentName, Page X]`

3. **If the retrieved documents do NOT contain the answer**, respond with:
   "The provided document does not contain this information."

4. **NEVER answer questions outside the scope of the document.**
   If asked about topics not in the document (e.g., general knowledge), respond:
   "I can only answer questions about the uploaded document. This topic is not covered."

5. **Format your responses using Markdown** for better readability:
   - Use bullet points or numbered lists for multiple items
   - Use **bold** for emphasis
   - Use headings (##) for sections when appropriate
   - For tabular data, use proper markdown tables with:
     * A header row
     * A separator row with dashes (|---|---|)
     * Data rows (each on a new line)
     Example:
     | Model | Capacity |
     |-------|----------|
     | RD205 | 180L     |

6. **Be concise and accurate.** Only include information directly from the document.

7. **At the end of EVERY response**, suggest 3 brief follow-up questions the user might ask.
   Format them EXACTLY like this (on separate lines):
   
   ---
   FOLLOW_UP: What is the energy rating?
   FOLLOW_UP: How much does it weigh?
   FOLLOW_UP: What colors are available?"""


# ============================================================
# STATE
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ============================================================
# TOOLS
# ============================================================
@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve relevant chunks from the loaded documents based on the query.
    Uses hybrid search (BM25 + vector) for robust retrieval.
    """
    manager = VectorStoreManager()

    # Use hybrid search for better recall (combines keyword + semantic)
    docs = manager.hybrid_search(query, k=6, vector_weight=0.5)

    if not docs:
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    context_str = ""
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "Unknown")
        context_str += (
            f"\n--- [Page {page}] (Source: {source}) ---\n{doc.page_content}\n"
        )

    return context_str


# ============================================================
# NODES
# ============================================================
def agent_node(state: AgentState):
    messages = state["messages"]

    # Prepend system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Initialize LLM (8B model for lower token usage)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, streaming=True)

    # Bind tools
    tools = [retrieve_documents]
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ============================================================
# ROUTER
# ============================================================
def decide_next_node(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ============================================================
# GRAPH BUILDER
# ============================================================
def create_graph(model_name: str = "llama-3.1-8b-instant"):
    """Create the agent graph with specified model."""

    def agent_node_with_model(state: AgentState):
        """Agent node that uses the specified model."""
        messages = state["messages"]

        # Prepend system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Initialize LLM with selected model
        llm = ChatGroq(model=model_name, temperature=0, streaming=True)

        # Bind tools
        tools = [retrieve_documents]
        llm_with_tools = llm.bind_tools(tools)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node_with_model)
    workflow.add_node("tools", ToolNode([retrieve_documents]))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", decide_next_node, {"tools": "tools", "__end__": END}
    )

    workflow.add_edge("tools", "agent")  # Loop back to generate answer

    return workflow.compile()
