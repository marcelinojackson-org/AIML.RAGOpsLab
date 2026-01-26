from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict
import textwrap

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph


class GraphState(TypedDict, total=False):
    query: str
    k: int
    k_max: int
    attempts: int
    docs: list
    context: str
    citations: list[dict[str, Any]]
    answer: str
    response_metadata: dict[str, Any] | None


@dataclass
class GraphChatResult:
    answer: str
    citations: list[dict[str, Any]]
    used_k: int
    attempts: int
    response_metadata: dict[str, Any] | None = None
    context: str | None = None


def _build_context(docs: list) -> tuple[str, list[dict[str, Any]]]:
    context_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        context_lines.append(f"[{idx}] {doc.page_content}")
        metadata = doc.metadata or {}
        citations.append(
            {
                "index": idx,
                "source": metadata.get("source", ""),
                "file_name": metadata.get("file_name", ""),
                "page": metadata.get("page", ""),
            }
        )
    return "\n\n".join(context_lines), citations


def answer_question_graph(
    query: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chat_model: str,
    k_default: int,
    k_max: int,
    retry_on_no_answer: bool,
    trace: bool = False,
    trace_preview_width: int = 120,
) -> GraphChatResult:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_default})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use only the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite sources with [#] matching the context numbers.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        ]
    )
    llm = ChatOllama(model=chat_model)
    chain = prompt | llm

    def _log(message: str) -> None:
        if trace:
            print(message)

    def retrieve(state: GraphState) -> GraphState:
        _log(f"[graph] retrieve: k={state['k']}")
        k = state["k"]
        retriever.search_kwargs["k"] = k
        docs = retriever.invoke(state["query"])
        if not docs:
            _log("[graph] retrieve: no documents returned")
            return {"docs": [], "context": "", "citations": []}
        context, citations = _build_context(docs)
        _log(f"[graph] retrieve: docs={len(docs)} context_chars={len(context)}")
        for idx, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source", "")
            page = metadata.get("page", "")
            preview = textwrap.shorten(
                doc.page_content.replace("\n", " "),
                width=trace_preview_width,
                placeholder="…",
            )
            _log(f"[graph] doc {idx}: page={page} source={source}")
            _log(f"[graph] doc {idx} preview: {preview}")
        return {"docs": docs, "context": context, "citations": citations}

    def answer(state: GraphState) -> GraphState:
        _log("[graph] answer: generating response")
        if not state.get("context"):
            return {"answer": "No relevant documents found.", "response_metadata": {}}
        response = chain.invoke({"context": state["context"], "question": state["query"]})
        metadata = getattr(response, "response_metadata", {}) or {}
        return {"answer": response.content, "response_metadata": metadata}

    def assess(state: GraphState) -> str:
        if not retry_on_no_answer:
            return "end"
        answer_text = (state.get("answer") or "").lower()
        no_answer = (
            "no relevant documents" in answer_text
            or "don't know" in answer_text
            or "do not know" in answer_text
            or "not stated" in answer_text
            or "not in the context" in answer_text
        )
        if no_answer and state["k"] < state["k_max"]:
            _log("[graph] assess: no answer, retrying with higher k")
            return "retry"
        _log("[graph] assess: done (no retry)")
        return "end"

    def retry(state: GraphState) -> GraphState:
        next_k = min(state["k"] * 2, state["k_max"])
        _log(f"[graph] retry: k {state['k']} -> {next_k}")
        return {"k": next_k, "attempts": state.get("attempts", 0) + 1}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("retry", retry)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges("answer", assess, {"retry": "retry", "end": END})
    graph.add_edge("retry", "retrieve")

    app = graph.compile()
    final_state = app.invoke(
        {
            "query": query,
            "k": k_default,
            "k_max": k_max,
            "attempts": 0,
        }
    )

    return GraphChatResult(
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []) or [],
        used_k=final_state.get("k", k_default),
        attempts=final_state.get("attempts", 0),
        response_metadata=final_state.get("response_metadata"),
        context=final_state.get("context", ""),
    )
