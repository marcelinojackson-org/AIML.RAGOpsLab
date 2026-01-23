# AIML.RAGOpsLab (LangChain)

A clean, incremental build of a local RAG system using LangChain + Ollama + Chroma.

## What this repo is

- A LangChain-first RAG lab that grows in small, demoable increments.
- A local-first stack that uses Ollama for models and Chroma for storage.
- A CLI-driven workflow for ingesting data and chatting with it.

## Repo layout

```
AIML.RAGOpsLab
├─ ragopslab/          # Python package + CLI
├─ config.yaml         # Default configuration (auto-loaded)
├─ data/               # Sample data (optional)
├─ storage/            # Local vector store persistence
├─ requirements.txt    # Dependencies (managed via uv)
├─ README.md           # Repo overview (this file)
├─ README_DAYS.md      # Day-by-day plan + demos
```

## Requirements

- Python 3.10+
- uv
- Ollama running locally

## Current status

- Foundation in place (package + CLI stubs).
- Next milestone: ingestion pipeline.

## Chroma data behavior

- Re-running `ingest` now **skips duplicates** when a file has already been indexed.
- Duplicate files are reported as `Duplicate: <path>` and are not re-loaded.
- To start fresh, use `--reset` to delete the Chroma persistence directory before indexing.

## PDF ingestion notes

- PDF parsing warnings from `pypdf` are suppressed to keep ingest output clean.

## Inspecting Chroma data

- Use `python -m ragopslab list --limit 0` to return **all rows** in the collection.

## Configuration

- `config.yaml` is auto-loaded by default.
- CLI flags override values from `config.yaml`.

## See the day-by-day plan

For daily goals, demos, and standup-friendly status notes, see `README_DAYS.md`.
