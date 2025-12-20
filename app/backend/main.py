from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import shutil

from app.backend.services.ingest import DocumentProcessor
from app.backend.services.vector_db import VectorStoreManager
from app.backend.graph import create_graph
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Create temp directory if not exists
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process PDF
        processor = DocumentProcessor()
        text, docs = processor.process(temp_path)

        # Add to Chroma
        v_mgr = VectorStoreManager()
        added = v_mgr.add_documents(docs)

        # Generate Onboarding Questions
        starter_qs = processor.generate_starter_questions()

        return {
            "status": "success",
            "message": "File processed successfully",
            "filename": file.filename,
            "starter_questions": starter_qs,
            "cached": not added,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/document-info")
async def get_document_info():
    """Get info about the currently loaded document, including starter questions."""
    try:
        v_mgr = VectorStoreManager()
        # Get first document's metadata to find the source filename
        result = v_mgr.collection.peek(limit=3)  # Get 3 chunks for starter Qs

        if result and result["metadatas"] and len(result["metadatas"]) > 0:
            source = result["metadatas"][0].get("source", None)
            count = v_mgr.collection.count()

            # Generate starter questions from existing chunks
            starter_questions = []
            if result["documents"]:
                from app.backend.services.ingest import DocumentProcessor

                processor = DocumentProcessor()
                # Manually set chunks for question generation
                from langchain_core.documents import Document

                processor.chunks = [
                    Document(page_content=doc) for doc in result["documents"]
                ]
                starter_questions = processor.generate_starter_questions()

            return {
                "has_document": True,
                "filename": source,
                "chunk_count": count,
                "starter_questions": starter_questions,
            }
        return {
            "has_document": False,
            "filename": None,
            "chunk_count": 0,
            "starter_questions": [],
        }
    except Exception:
        return {
            "has_document": False,
            "filename": None,
            "chunk_count": 0,
            "starter_questions": [],
        }


@app.post("/ask")
async def ask_question(req: ChatRequest):
    graph = create_graph()

    # Reconstruct history
    messages = []
    for msg in req.chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=req.question))

    async def event_generator():
        # Use astream_events to get granular updates (tokens & tool status)
        try:
            async for event in graph.astream_events(
                {"messages": messages}, version="v1"
            ):
                event_type = event["event"]

                # 1. Tool Call Start (Glass Box: "Searching...")
                if event_type == "on_tool_start":
                    tool_input = event["data"].get("input", "")
                    yield (
                        json.dumps(
                            {
                                "type": "status",
                                "content": f"🔍 Searching: {event['name']} ({tool_input})...",
                            }
                        )
                        + "\n"
                    )

                # 2. Chat Model Stream (Tokens)
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]

                    # A. Content Tokens (Text)
                    content = ""
                    if hasattr(chunk, "content"):
                        if isinstance(chunk.content, str):
                            content = chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and "text" in part:
                                    content += part["text"]

                    if content:
                        yield (json.dumps({"type": "token", "content": content}) + "\n")

                    # B. Tool Call Chunks (Streaming Args)
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            if tc_chunk.get("args"):
                                yield (
                                    json.dumps(
                                        {
                                            "type": "tool_log",
                                            "content": tc_chunk["args"],
                                        }
                                    )
                                    + "\n"
                                )  # else:
                #     print(f"DEBUG: Event {event_type}")

                # 3. Tool End (Glass Box: "Found info")
                elif event_type == "on_tool_end":
                    yield (
                        json.dumps(
                            {
                                "type": "status",
                                "content": "✅ Found relevant documents.",
                            }
                        )
                        + "\n"
                    )

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
