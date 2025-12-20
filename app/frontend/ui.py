"""
Agentic PDF Assistant - Streamlit UI
Clean implementation with proper state management and streaming.
"""

import streamlit as st
import requests
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="PDF AI Assistant", page_icon="🤖", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Thinking box */
    [data-testid="stExpander"] {
        background: rgba(100, 255, 218, 0.05);
        border: 1px solid rgba(100, 255, 218, 0.2);
        border-radius: 8px;
    }
    
    /* Starter chips */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Scrollable tables */
    .stMarkdown table {
        display: block;
        max-width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 400px;
        white-space: nowrap;
    }
    .stMarkdown thead {
        position: sticky;
        top: 0;
        background: #1a1a2e;
        z-index: 1;
    }
    
    /* Add padding at bottom */
    .main .block-container {
        padding-bottom: 80px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "starter_questions" not in st.session_state:
    st.session_state.starter_questions = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "document_filename" not in st.session_state:
    st.session_state.document_filename = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"

# ============================================================
# CHECK FOR EXISTING DOCUMENT ON PAGE LOAD
# ============================================================
if st.session_state.document_filename is None:
    try:
        resp = requests.get(f"{BACKEND_URL}/document-info", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("has_document"):
                st.session_state.document_filename = data.get("filename")
                st.session_state.file_uploaded = True
                # Also get starter questions
                st.session_state.starter_questions = data.get("starter_questions", [])
    except Exception:
        pass

# ============================================================
# SIDEBAR: FILE UPLOAD
# ============================================================
with st.sidebar:
    st.header("📂 Upload Document")
    st.info("⚠️ Uploading a new file will replace the current document.", icon="ℹ️")

    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], help="Upload a PDF document to analyze"
    )

    # Check if a NEW file is being uploaded (different from current)
    new_upload = uploaded_file and (
        not st.session_state.file_uploaded
        or uploaded_file.name != st.session_state.document_filename
    )

    if new_upload:
        with st.spinner("🔍 Analyzing document..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ingest", files={"file": uploaded_file}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.starter_questions = data.get(
                        "starter_questions", []
                    )
                    st.session_state.document_filename = data.get("filename")
                    st.session_state.file_uploaded = True
                    st.session_state.messages = []  # Clear chat for new doc
                    st.success("✅ Document ready!")
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if st.session_state.file_uploaded and st.session_state.document_filename:
        st.success(f"📄 **{st.session_state.document_filename}**")
        if st.button("🔄 Clear Document"):
            # Delete from ChromaDB
            try:
                requests.delete(f"{BACKEND_URL}/clear-document")
            except Exception:
                pass
            st.session_state.file_uploaded = False
            st.session_state.messages = []
            st.session_state.starter_questions = []
            st.session_state.document_filename = None
            st.rerun()

    # --- Model Selection ---
    st.divider()
    MODEL_OPTIONS = {
        "⚡ Fast (8B)": "llama-3.1-8b-instant",
        "🧠 Smart (70B)": "llama-3.3-70b-versatile",
    }
    st.session_state.selected_model = MODEL_OPTIONS[
        st.selectbox(
            "Select Model",
            options=list(MODEL_OPTIONS.keys()),
            index=0 if st.session_state.selected_model == "llama-3.1-8b-instant" else 1,
        )
    ]


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.title("🤖 PDF AI Assistant")
st.caption("Ask questions about your uploaded documents")


# --- Helper: Parse Follow-up Questions ---
def parse_followups(content):
    """Extract FOLLOW_UP: lines and return (clean_content, followups_list)"""
    lines = content.split("\n")
    clean_lines = []
    followups = []

    for line in lines:
        if line.strip().startswith("FOLLOW_UP:"):
            question = line.strip().replace("FOLLOW_UP:", "").strip()
            if question:
                followups.append(question)
        else:
            clean_lines.append(line)

    # Remove trailing separator if present
    clean_content = "\n".join(clean_lines).rstrip("-").rstrip()
    return clean_content, followups


# --- Display Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Show thinking process if available
        if msg.get("thinking"):
            with st.expander("🧠 Thinking Process", expanded=False):
                for log in msg["thinking"]:
                    st.write(log)

        # Parse and display content (without follow-up lines)
        if msg["role"] == "assistant":
            clean_content, followups = parse_followups(msg["content"])
            st.markdown(clean_content)

            # Show follow-up buttons for the LAST assistant message only
            if followups and idx == len(st.session_state.messages) - 1:
                st.caption("**Follow-up questions:**")
                for i, q in enumerate(followups):
                    if st.button(q, key=f"followup_{idx}_{i}", type="secondary"):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()
        else:
            st.markdown(msg["content"])

# --- Starter Question Chips ---
if st.session_state.starter_questions and not st.session_state.messages:
    st.markdown("### 💡 Suggested Questions")
    st.caption("Click any question to get started:")

    for i, question in enumerate(st.session_state.starter_questions):
        # Use a container for each question for better styling
        if st.button(f"→ {question}", key=f"starter_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# --- User Input ---
user_input = st.chat_input("Ask about your document...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# --- Generate Response (if last message is from user) ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        # Thinking container FIRST (so it's on top)
        thinking_container = st.empty()
        thinking_logs = []

        # Response container SECOND
        response_container = st.empty()
        full_response = ""

        try:
            # Build clean chat history (exclude thinking logs)
            clean_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            with requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "question": st.session_state.messages[-1]["content"],
                    "chat_history": clean_history,
                    "model_name": st.session_state.selected_model,
                },
                stream=True,
            ) as response:
                if response.status_code != 200:
                    st.error(f"Backend error: {response.status_code}")
                    st.stop()

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            event_type = data.get("type")
                            content = data.get("content", "")

                            if event_type == "status":
                                thinking_logs.append(content)
                                thinking_container.info(f"🧠 {content}")

                            elif event_type == "tool_log":
                                thinking_logs.append(f"🛠️ {content}")
                                thinking_container.info(f"🛠️ Building query: {content}")

                            elif event_type == "token":
                                full_response += content
                                response_container.markdown(full_response + "▌")

                            elif event_type == "error":
                                st.error(content)
                                st.stop()

                        except json.JSONDecodeError:
                            pass

                # Final update
                thinking_container.empty()
                response_container.markdown(full_response)

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "thinking": thinking_logs if thinking_logs else None,
                    }
                )
                st.rerun()

        except Exception as e:
            st.error(f"Connection failed: {e}")
