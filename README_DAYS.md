# AIML.RAGOpsLab - Day-by-day plan

This file is the running mini-sprint plan. Each day ends with a runnable demo and a short standup-ready status.

## Day 1 - Foundation (done)

**Goal**
- Establish a clean project skeleton with a CLI entrypoint.

**What was built**
- Minimal Python package scaffold (`ragopslab/`).
- CLI stubs for `ingest` and `chat`.
- Dependency list in `requirements.txt`.

**End-of-day demo**
```bash
cd /Users/marc/dev/opensource/AIML/AIML.RAGOpsLab
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

python -m ragopslab --help
python -m ragopslab ingest
python -m ragopslab chat
```

**Standup notes (readable status)**
- Yesterday I set up the LangChain project skeleton with a CLI entrypoint.
- I can run the CLI and see the ingest/chat commands wired up (stubs).
- Next, I will implement the ingestion pipeline to load files and build a Chroma index.

## Day 2 - Ingestion pipeline (done)

**Goal**
- Load files from a directory, chunk them, embed with Ollama, and store in Chroma.

**What was built**
- File discovery for common extensions (txt, md, pdf), including PDFs dropped into `data/sample_docs`.
- Chunking with `RecursiveCharacterTextSplitter`.
- Local embeddings via `OllamaEmbeddings`.
- Chroma persistence in a local directory (`storage/chroma`).
- CLI options for chunk sizing, extensions, and reset.
- A `list` command to inspect Chroma contents (table/CSV/TSV).
- A `config.yaml` file that is auto-loaded; CLI flags override config values.
- Output-to-file support for CSV/TSV lists.

**End-of-day demo**
```bash
cd /Users/marc/dev/opensource/AIML/AIML.RAGOpsLab
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

ollama pull nomic-embed-text

# First-time ingest (or incremental ingest).
python -m ragopslab ingest \
  --data-dir data/sample_docs \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --embedding-model nomic-embed-text

# Full re-index: reset clears the local Chroma store first.
python -m ragopslab ingest \
  --data-dir data/sample_docs \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --embedding-model nomic-embed-text \
  --reset

python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 5

# CSV/TSV output for easy scrolling or export.
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 5 \
  --format csv

# CSV/TSV output with full metadata column.
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 5 \
  --format tsv \
  --full-meta

# Use --limit 0 to return all rows in the collection.
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 0 \
  --format csv

# Save CSV/TSV output to a file (relative to the repo root).
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 5 \
  --format csv \
  --output reports/chroma_list.csv

# Include full metadata for each sample row.
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --limit 5 \
  --full-meta

# Filter by page and include full chunk text + vectors (large output).
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --page 0 \
  --chunk-text \
  --include-vectors \
  --format csv \
  --output temp/page0_chunks_vectors.csv

# Limit vector output to the first N dimensions for readability.
python -m ragopslab list \
  --persist-dir storage/chroma \
  --collection ragopslab \
  --page 0 \
  --chunk-text \
  --include-vectors \
  --vector-dims 10 \
  --format csv \
  --output temp/page0_chunks_vectors_10dims.csv
```

**Standup notes (readable status)**
- I implemented the ingestion pipeline that discovers files (including PDFs), chunks them, and embeds locally with Ollama.
- Vectors are persisted in a local Chroma store so the chat step can reuse the index.
- The CLI now supports chunk sizing, file extensions, reset for clean re-indexing, a list command to inspect stored docs, and duplicate skipping (files already indexed are not reloaded).
- The list output supports table/CSV/TSV, optional full metadata, `--limit 0` for all rows, and file output via `--output`.
- PDF parsing warnings from `pypdf` are suppressed to keep ingest output clean.
- Next, I will add the chat retrieval path and citations.

**Notes**
- Re-running ingest without `--reset` will now skip duplicate files already indexed.

## Day 3 - Chat retrieval (done)

**Goal**
- Ask questions against the indexed data and return citations.

**What was built**
- Chat command that retrieves top‑k chunks from Chroma and answers with Ollama.
- Citations printed alongside the answer.
- Config support for chat model and retrieval k.
- New `ragopslab/chat.py` module for retrieval + answer logic.

**Chat parameters (CLI overrides)**
- `--query`: the question to ask (required).
- `--chat-model`: Ollama chat model (default from config).
- `--embedding-model`: embedding model for retrieval (default from config).
- `--k`: number of chunks retrieved from Chroma (default from config).
- `--persist-dir`: Chroma storage path (default from config).
- `--collection`: Chroma collection name (default from config).

**Config additions (`config.yaml`)**
- `models.chat_model`: default chat model (e.g., `llama3.1:8b`).
- `retrieval.k`: default top‑k for retrieval.

**End-of-day demo**
```bash
cd /Users/marc/dev/opensource/AIML/AIML.RAGOpsLab
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

ollama pull llama3.1:8b

python -m ragopslab chat \
  --query "Summarize the resume in 3 bullet points." \
  --chat-model llama3.1:8b \
  --k 4
```

**Standup notes (readable status)**
- I implemented the chat flow: retrieve top‑k chunks from Chroma and answer with Ollama.
- Answers include citations that map back to source files and pages.
- Chat config is now driven by `config.yaml` with CLI overrides.
- Chat parameters are exposed via CLI for quick experiments (model/k/collection/persist dir).
- Chat output supports `--output-format markdown|json|plain` (default is markdown).

**Limitations & how to ask questions (Day 3)**
- The model only sees the **top‑k retrieved chunks**, not the whole PDF at once.
- `k` controls how many chunks are fed into the LLM; higher `k` increases recall but also context size and noise.
- If the answer is not explicitly in the retrieved chunks, the model should say **“I don’t know.”**
- You don’t need to know the exact “top‑k” ahead of time; ask normal questions and increase `k` if answers are incomplete.
- Example: If “years of Python experience” isn’t explicitly stated in the resume, the correct output is “not stated.”

**What’s coming to improve this**
- Day 4 (LangGraph): automatic fallback — re‑run retrieval with higher `k` or refined queries if the answer is missing.
- Day 5 (more formats + metadata): structured data + file filters to target specific sources (e.g., resume only).
- Day 6 (advanced tooling): reranking/compression to improve quality without huge `k`.

## Day 4 - LangGraph orchestration

**Goal**
- Model the pipeline (ingest → retrieve → answer) as a LangGraph.

**Planned demo**
- Run the graph version and confirm identical output.

## Day 5 - More data formats + config

**Goal**
- Add loaders for more formats and a config-driven pipeline.

**Planned demo**
- Ingest a mixed-format directory and show it in chat.

## Day 6 - Advanced LangChain tooling

**Goal**
- Optional LangSmith tracing + evals and optional LangServe API.

**Planned demo**
- Show tracing or an eval report (if enabled).
