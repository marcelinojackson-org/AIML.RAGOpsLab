import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

from ragopslab.chat import answer_question
from ragopslab.config import load_config
from ragopslab.graph_chat import answer_question_graph
from ragopslab.ingest import ingest_directory
from ragopslab.inspect import list_sources, summarize_collection
from ragopslab.usage import build_usage_summary


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    extensions = args.extensions
    if extensions is None:
        extensions = config["files"]["extensions"]
    if isinstance(extensions, str):
        extensions = [ext.strip() for ext in extensions.split(",") if ext.strip()]
    try:
        stats = ingest_directory(
            data_dir=Path(args.data_dir or config["paths"]["data_dir"]),
            persist_dir=Path(args.persist_dir or config["paths"]["persist_dir"]),
            collection_name=args.collection or config["chroma"]["collection"],
            embedding_model=args.embedding_model or config["models"]["embedding_model"],
            chunk_size=args.chunk_size or config["chunking"]["chunk_size"],
            chunk_overlap=args.chunk_overlap or config["chunking"]["chunk_overlap"],
            extensions=extensions,
            reset=args.reset,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Ingestion complete.")
    print(f"- files_seen: {stats.files_seen}")
    print(f"- files_loaded: {stats.files_loaded}")
    print(f"- docs_loaded: {stats.docs_loaded}")
    print(f"- chunks_created: {stats.chunks_created}")
    print(f"- skipped: {stats.skipped}")
    print(f"- duplicates: {stats.duplicates}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    query = args.query or ""
    if not query:
        print("Error: --query is required.")
        return 1

    chat_model = args.chat_model or config["models"]["chat_model"]
    embedding_model = args.embedding_model or config["models"]["embedding_model"]
    k = args.k if args.k is not None else config["retrieval"]["k"]
    k_default = config["retrieval"].get("k_default", k)
    k_max = config["retrieval"].get("k_max", k)
    retry_on_no_answer = config["retrieval"].get("retry_on_no_answer", True)

    use_graph = bool(args.graph)
    if use_graph:
        result = answer_question_graph(
            query=query,
            persist_dir=Path(args.persist_dir or config["paths"]["persist_dir"]),
            collection_name=args.collection or config["chroma"]["collection"],
            embedding_model=embedding_model,
            chat_model=chat_model,
            k_default=k_default,
            k_max=k_max,
            retry_on_no_answer=retry_on_no_answer,
            trace=bool(args.trace),
            trace_preview_width=args.trace_preview_width,
        )
        response_metadata = result.response_metadata
        context = result.context or ""
        used_k = result.used_k
        attempts = result.attempts
    else:
        result = answer_question(
            query=query,
            persist_dir=Path(args.persist_dir or config["paths"]["persist_dir"]),
            collection_name=args.collection or config["chroma"]["collection"],
            embedding_model=embedding_model,
            chat_model=chat_model,
            k=k,
        )
        response_metadata = result.response_metadata
        context = result.context or ""
        used_k = k
        attempts = 0

    cost_cfg = config.get("cost", {})
    pricing = config.get("pricing", {})
    show_usage = bool(args.show_usage or cost_cfg.get("show_usage", False))
    usage = build_usage_summary(
        response_metadata=response_metadata,
        estimator=cost_cfg.get("estimator", "ollama"),
        prompt_text=context + "\n\nQuestion: " + query,
        completion_text=result.answer,
        model=chat_model,
        pricing=pricing,
        default_prompt_per_1k=cost_cfg.get("default_prompt_per_1k", 0.0),
        default_completion_per_1k=cost_cfg.get("default_completion_per_1k", 0.0),
        enabled=cost_cfg.get("enabled", False),
    )

    if args.output_format == "markdown":
        print("\nAnswer:")
        print("```")
        print(result.answer)
        print("```")
        print("\nCitations:")
        if not result.citations:
            print("1. None")
        else:
            for item in result.citations:
                file_name = item.get("file_name", "") or "unknown"
                page = item.get("page", "")
                source = item.get("source", "")
                page_part = f" (page {page})" if page != "" else ""
                print(f"{item['index']}. {file_name}{page_part} — {source}")
        if use_graph:
            print(f"\nRetrieval:")
            print(f"- used_k: {used_k}")
            print(f"- attempts: {attempts}")
            print("")
        if show_usage and usage:
            print("\nUsage:")
            print(f"- prompt_tokens: {usage.prompt_tokens}")
            print(f"- completion_tokens: {usage.completion_tokens}")
            print(f"- total_tokens: {usage.total_tokens}")
            if usage.estimated_cost is not None:
                print(f"- estimated_cost: ${usage.estimated_cost:.4f}")
        return 0

    if args.output_format == "json":
        payload = {
            "question": query,
            "answer": result.answer,
            "citations": result.citations,
        }
        if use_graph:
            payload["retrieval"] = {"used_k": used_k, "attempts": attempts}
        if show_usage and usage:
            payload["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost": usage.estimated_cost,
            }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    print("Answer:")
    print(result.answer)
    print("\nCitations:")
    if not result.citations:
        print("1) None")
        return 0
    for item in result.citations:
        print(
            f"{item['index']}) file={item.get('file_name','')} | "
            f"page={item.get('page','')} | source={item.get('source','')}"
        )
    if use_graph:
        print(f"\nRetrieval: used_k={used_k} attempts={attempts}")
    if show_usage and usage:
        print(
            f"\nUsage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"total={usage.total_tokens} cost=${usage.estimated_cost:.4f}"
        )
    return 0


def _render_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(sep: str = "-") -> str:
        parts = [sep * (w + 2) for w in widths]
        return f"+{'+'.join(parts)}+"

    def _row(values: list[str]) -> str:
        padded = [f" {val.ljust(widths[idx])} " for idx, val in enumerate(values)]
        return f"|{'|'.join(padded)}|"

    print(_line("-"))
    print(_row(headers))
    print(_line("="))
    for row in rows:
        print(_row(row))
        print(_line("-"))


def _render_delimited(headers: list[str], rows: list[list[str]], delimiter: str) -> None:
    writer = csv.writer(sys.stdout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    writer.writerows(rows)


def _write_delimited(path: Path, headers: list[str], rows: list[list[str]], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)


def _cmd_list(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    limit = args.limit if args.limit is not None else config["list"]["limit"]
    fmt = args.format or config["list"]["format"]
    preview_width = (
        args.preview_width if args.preview_width is not None else config["list"]["preview_width"]
    )
    try:
        summary = summarize_collection(
            persist_dir=Path(args.persist_dir or config["paths"]["persist_dir"]),
            collection_name=args.collection or config["chroma"]["collection"],
            limit=limit,
            include_embeddings=args.include_vectors,
            page=args.page,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if summary.count == 0:
        if fmt in {"csv", "tsv"}:
            headers = ["#", "id", "file", "page", "ext", "preview"]
            if args.full_meta:
                headers.append("metadata")
            if args.include_vectors:
                headers.append("vector")
            delimiter = "," if fmt == "csv" else "\t"
            if args.output:
                _write_delimited(Path(args.output), headers=headers, rows=[], delimiter=delimiter)
            else:
                _render_delimited(headers=headers, rows=[], delimiter=delimiter)
        else:
            print(f"Collection: {args.collection}")
            print("- count: 0")
        return 0

    rows: list[list[str]] = []
    for idx, doc_id in enumerate(summary.ids, start=1):
        metadata = summary.metadatas[idx - 1] if idx - 1 < len(summary.metadatas) else {}
        document = summary.documents[idx - 1] if idx - 1 < len(summary.documents) else ""
        content = document.replace("\n", " ")
        if args.chunk_text:
            preview = content
        else:
            preview = textwrap.shorten(
                content,
                width=preview_width,
                placeholder="…",
            )

        source = metadata.get("source", "")
        file_name = metadata.get("file_name") or Path(source).name if source else ""
        page = str(metadata.get("page", "")) if metadata else ""
        ext = metadata.get("file_ext", "")

        row = [str(idx), doc_id, file_name, page, ext, preview]
        if args.full_meta:
            row.append(json.dumps(metadata, ensure_ascii=True))
        if args.include_vectors:
            vector = summary.embeddings[idx - 1] if summary.embeddings else None
            if vector is not None and hasattr(vector, "tolist"):
                vector = vector.tolist()
            if vector is not None and args.vector_dims:
                vector = vector[: args.vector_dims]
            row.append(json.dumps(vector, ensure_ascii=True))
        rows.append(row)

    headers = ["#", "id", "file", "page", "ext", "preview"]
    if args.full_meta:
        headers.append("metadata")
    if args.include_vectors:
        headers.append("vector")

    if fmt == "table":
        if args.output:
            print("Error: --output requires --format csv or tsv.")
            return 1
        print(f"Collection: {args.collection}")
        print(f"- count: {summary.count}")
        print("- sample:")
        _render_table(headers=headers, rows=rows)
        if args.full_meta:
            print("\n- full metadata:")
            for idx, metadata in enumerate(summary.metadatas, start=1):
                print(f"  {idx}) {metadata}")
    elif fmt == "csv":
        if args.output:
            _write_delimited(Path(args.output), headers=headers, rows=rows, delimiter=",")
        else:
            _render_delimited(headers=headers, rows=rows, delimiter=",")
    elif fmt == "tsv":
        if args.output:
            _write_delimited(Path(args.output), headers=headers, rows=rows, delimiter="\t")
        else:
            _render_delimited(headers=headers, rows=rows, delimiter="\t")
    if args.output:
        print(f"Wrote output to {args.output}")
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    try:
        sources = list_sources(
            persist_dir=Path(args.persist_dir or config["paths"]["persist_dir"]),
            collection_name=args.collection or config["chroma"]["collection"],
            source_type=args.source_type,
            file_name=args.file_name,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.format in {"csv", "tsv"}:
        headers = ["source_type", "file_name", "source", "count"]
        rows = [[s.source_type, s.file_name, s.source, str(s.count)] for s in sources]
        delimiter = "," if args.format == "csv" else "\t"
        if args.output:
            _write_delimited(Path(args.output), headers=headers, rows=rows, delimiter=delimiter)
        else:
            _render_delimited(headers=headers, rows=rows, delimiter=delimiter)
        return 0

    if not sources:
        print("No sources found.")
        return 0

    rows = [[s.source_type, s.file_name, s.source, str(s.count)] for s in sources]
    _render_table(headers=["type", "file", "source", "count"], rows=rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Ops Lab (LangChain)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a directory into Chroma")
    ingest.add_argument("--config", default="config.yaml")
    ingest.add_argument("--data-dir")
    ingest.add_argument("--persist-dir")
    ingest.add_argument("--collection")
    ingest.add_argument("--embedding-model")
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)
    ingest.add_argument("--extensions")
    ingest.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma index before re-ingesting (full re-index).",
    )
    ingest.set_defaults(func=_cmd_ingest)

    chat = subparsers.add_parser("chat", help="Chat over the indexed data")
    chat.add_argument("--config", default="config.yaml")
    chat.add_argument("--query")
    chat.add_argument("--persist-dir")
    chat.add_argument("--collection")
    chat.add_argument("--embedding-model")
    chat.add_argument("--chat-model")
    chat.add_argument("--k", type=int)
    chat.add_argument("--output-format", choices=["markdown", "json", "plain"], default="markdown")
    chat.add_argument("--graph", action="store_true", help="Use LangGraph adaptive flow.")
    chat.add_argument("--show-usage", action="store_true", help="Print token/cost usage.")
    chat.add_argument("--trace", action="store_true", help="Print step-by-step graph logs.")
    chat.add_argument("--trace-preview-width", type=int, default=120)
    chat.set_defaults(func=_cmd_chat)

    list_cmd = subparsers.add_parser("list", help="List documents in Chroma")
    list_cmd.add_argument("--config", default="config.yaml")
    list_cmd.add_argument("--persist-dir")
    list_cmd.add_argument("--collection")
    list_cmd.add_argument("--limit", type=int)
    list_cmd.add_argument("--full-meta", action="store_true")
    list_cmd.add_argument("--format", choices=["table", "csv", "tsv"])
    list_cmd.add_argument("--preview-width", type=int)
    list_cmd.add_argument("--chunk-text", action="store_true", help="Show full chunk text.")
    list_cmd.add_argument("--include-vectors", action="store_true", help="Include embedding vectors.")
    list_cmd.add_argument("--page", type=int, help="Filter results to a PDF page number.")
    list_cmd.add_argument("--vector-dims", type=int, help="Limit vector dimensions in output.")
    list_cmd.add_argument("--output", help="Write CSV/TSV output to a file (relative paths allowed).")
    list_cmd.set_defaults(func=_cmd_list)

    sources_cmd = subparsers.add_parser("sources", help="List indexed sources")
    sources_cmd.add_argument("--config", default="config.yaml")
    sources_cmd.add_argument("--persist-dir")
    sources_cmd.add_argument("--collection")
    sources_cmd.add_argument("--format", choices=["table", "csv", "tsv"], default="table")
    sources_cmd.add_argument("--output", help="Write CSV/TSV output to a file.")
    sources_cmd.add_argument("--source-type", help="Filter by source_type (csv/json/pdf/txt/md).")
    sources_cmd.add_argument("--file-name", help="Filter by file name.")
    sources_cmd.set_defaults(func=_cmd_sources)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
