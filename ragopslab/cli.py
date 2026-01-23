import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

from ragopslab.config import load_config
from ragopslab.ingest import ingest_directory
from ragopslab.inspect import summarize_collection


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


def _cmd_chat(_args: argparse.Namespace) -> int:
    print("chat: not implemented yet")
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
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if summary.count == 0:
        if fmt in {"csv", "tsv"}:
            headers = ["#", "id", "file", "page", "ext", "preview"]
            if args.full_meta:
                headers.append("metadata")
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
        preview = textwrap.shorten(
            document.replace("\n", " "),
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
        rows.append(row)

    headers = ["#", "id", "file", "page", "ext", "preview"]
    if args.full_meta:
        headers.append("metadata")

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
    ingest.add_argument("--reset", action="store_true")
    ingest.set_defaults(func=_cmd_ingest)

    chat = subparsers.add_parser("chat", help="Chat over the indexed data")
    chat.set_defaults(func=_cmd_chat)

    list_cmd = subparsers.add_parser("list", help="List documents in Chroma")
    list_cmd.add_argument("--config", default="config.yaml")
    list_cmd.add_argument("--persist-dir")
    list_cmd.add_argument("--collection")
    list_cmd.add_argument("--limit", type=int)
    list_cmd.add_argument("--full-meta", action="store_true")
    list_cmd.add_argument("--format", choices=["table", "csv", "tsv"])
    list_cmd.add_argument("--preview-width", type=int)
    list_cmd.add_argument("--output", help="Write CSV/TSV output to a file (relative paths allowed).")
    list_cmd.set_defaults(func=_cmd_list)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
