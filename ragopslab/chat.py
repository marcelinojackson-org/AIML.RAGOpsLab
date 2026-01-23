from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


@dataclass
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]


def answer_question(
    query: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chat_model: str,
    k: int,
) -> ChatResult:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        return ChatResult(answer="No relevant documents found.", citations=[])

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
    response = chain.invoke({"context": "\n\n".join(context_lines), "question": query})

    return ChatResult(answer=response.content, citations=citations)
