# Legal RAG - Vietnamese Labor Law Legal RAG System

A Retrieval-Augmented Generation (RAG) system for Vietnamese Labor Law (Bộ luật Lao động 2019).

## Project Structure

```
LegalRAG/
├── configs/                 # Configuration files (YAML)
│   ├── base.yaml           # Main configuration
│   ├── chunking/           # Chunking strategy configs
│   ├── retrieval/          # Retrieval configs
│   ├── generation.yaml     # Generation config
│   ├── evaluation.yaml     # Evaluation config
│   └── experiment.yaml     # Experiment tracking config
├── data/                    # Data directories (run outputs are git-ignored)
│   ├── raw/                # Source PDFs (read-only)
│   ├── ocr/                # OCR evidence + page patches
│   │   ├── patches/        # JSON page patches (e.g. page-86 transcription fix)
│   │   └── renders/        # Verification renders (local only, ignored)
│   ├── processed/          # Corpus per run: labor_code_2019_<run_id>_corpus.jsonl
│   ├── chunks/             # legal_aware_<run_id>/ from pipeline;
│   │                       # fixed_300/500/800, paragraph, article, clause on demand
│   ├── indexes/            # Retrieval indexes (created by `legal-rag index ...`)
│   └── qa/                 # Ground truth QA pairs (planned, frozen)
├── src/                     # Source code (installable package)
│   └── legal_rag/
│       ├── __init__.py
│       ├── config.py       # Pydantic config management
│       ├── parsing/        # PDF parsing & legal structure
│       ├── chunking/       # Legal-aware + baseline chunkers
│       ├── corpus/         # Registry, audit, OCR adapters, batch build, page patches
│       ├── qa/             # QA dataset schema (tolerant loader)
│       ├── evaluation/     # Retrieval metrics + gold coverage
│       ├── retrieval/      # BM25, Dense, Hybrid, Reranker
│       ├── pipeline.py     # Standard end-to-end build (parse -> patch -> chunk -> validate)
│       └── cli/            # CLI entry points (Typer)
├── tests/                   # Unit & integration tests
│   ├── unit/
│   └── integration/
├── notebooks/               # EDA & analysis notebooks
├── reports/                 # Decision logs / generated reports
├── experiments/             # Experiment outputs (auto-generated, planned)
├── pyproject.toml           # Package configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── README.md
├── ARCHITECTURE.md
└── AGENTS.md               # Project rules
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"

# Or install only production dependencies
pip install -r requirements.txt
```

### Configuration

Copy and modify the base config:

```bash
cp configs/base.yaml configs/my_experiment.yaml
```

### Pipeline Commands

```bash
# 0. Standard build (recommended): PDF -> page patches -> corpus -> legal-aware chunks -> validation
#    Pick a new --run-id for every build; reusing an existing id is rejected (results are never overwritten).
legal-rag pipeline data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf --run-id run07

# 1. Parse PDF into a structured corpus (page patches in data/ocr/patches are applied automatically)
legal-rag parse pdf data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf

# 2. Run all baseline chunking strategies into a new run-specific directory
legal-rag chunk all-baselines data/processed/labor_code_2019_run07_corpus.jsonl --output-base data/chunks/run07

# 3. Build BM25 / dense indexes from the run's chunks
legal-rag index bm25 data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag index dense data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
# Optional BGE-M3 dense index (separate output, max sequence length 256)
legal-rag index dense data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl --model BAAI/bge-m3 --max-length 256 --output data/indexes/dense/bge_m3/dense_index.pkl

# 4. Retrieval (--chunks is required, so a stale run is never picked silently)
legal-rag retrieve bm25 "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag retrieve hybrid "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag retrieve rerank "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl

# 5. Validate / clean chunks
legal-rag chunk validate data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag chunk clean data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
```

### Whole-corpus build (all PDFs in `data/raw/`)

Document identity lives in `configs/corpus/documents.yaml`; parser and chunker
never fall back to hard-coded Labour Code metadata.

```bash
# 0. Audit text-layer coverage per PDF/page (read-only; never edits data/raw)
legal-rag corpus audit --output data/processed/corpus_audit.json

# 1. Build every registry document into a fresh run directory
legal-rag corpus build --run-id run10_ocr --ocr-engine tesseract --dpi 300 --ocr-workers 6

# 2. Inspect statuses (ok / partial_text / ocr_required / appendix_only / ...)
cat data/chunks/run10_ocr/corpus_status.json

# 3. Gold coverage: can every test reference be found in the chunks?
legal-rag evaluate coverage data/qa/test.jsonl \
  --chunks data/chunks/run10_ocr/all_legal_aware.jsonl

# 4. Retrieval evaluation (BM25 only works without a GPU/model download)
legal-rag evaluate retrieval data/qa/test.jsonl \
  --chunks data/chunks/run10_ocr/all_legal_aware.jsonl \
  --method bm25 --k 1,3,5,10,20
```

Build output contract (`data/chunks/<run_id>/`):

| File | Content |
|------|---------|
| `all_legal_aware.jsonl` | merged chunks from every document that produced text |
| `by_document/<id>.jsonl` | per-document chunks |
| `chunk_stats.json` | counts by chunk type and document |
| `corpus_status.json` | per-document status and OCR page counts |
| `parse_reports/<id>.json` | dropped lines, articles seen/missing, signature/appendix counts |
| `run_manifest.json` | git commit, PDF hashes, OCR engine, validation counts |

Documents whose PDF has no text layer are marked `ocr_required`, never silently
skipped. Appendix/table chunks get `retrieval_eligible: false` and are excluded
from retrieval metrics by default.

### OCR setup

OCR is optional. `easyocr` (supports `vi`) or Tesseract (`vie`) can be used:

```bash
pip install -e ".[ocr]"            # easyocr + pytesseract

# Tesseract on Windows (UB Mannheim build), then Vietnamese model:
winget install --id UB-Mannheim.TesseractOCR -e
# Download vie.traineddata into data/ocr/tessdata/
#   https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata
```

`corpus build` caches each OCR page in `data/ocr/<document_id>/page-XXX.json`
(keyed by engine + language + DPI), so re-runs do not repeat OCR work. Use
`--force-ocr` to ignore the cache and `--ocr-workers N` to OCR pages in
parallel (each page is an independent Tesseract subprocess; 4-6 workers is a
good default on a laptop).

OCR quality matters for article numbers: check
`data/chunks/<run_id>/parse_reports/<id>.json` for missing or duplicate
articles after a build. A page that OCRs badly should be fixed with a page
patch in `data/ocr/patches/` rather than by editing `data/raw/`.

### QA dataset schema

The frozen evaluation set is a JSONL file (`data/qa/*.jsonl` is git-ignored).
Each line is one question with explicit gold references; see
[`docs/qa_schema.md`](docs/qa_schema.md) for the full contract and an example.
Gold references use the same `document_id` as `configs/corpus/documents.yaml`
and bare article/clause numbers (`"Điều 113"` and `113` are both accepted).

`evaluate coverage` must be run before any retrieval comparison: if a gold
reference has no retrieval-eligible chunk, no retriever can return it and the
run is invalid for that question.

### Using Config File

```bash
# Run with custom config
legal-rag --config configs/my_experiment.yaml parse pdf ...
```

## Chunking Strategies

| Strategy | Description | Output Dir |
|----------|-------------|------------|
| `legal_aware` | Preserves legal hierarchy (Chapter→Section→Article→Clause→Point); standard output of `legal-rag pipeline` | `data/chunks/legal_aware_<run_id>/` |
| `fixed_300/500/800` | Fixed-size with overlap | `<output-base>/fixed_{size}/` |
| `paragraph` | Paragraph-based | `<output-base>/paragraph/` |
| `article` | One chunk per article | `<output-base>/article/` |
| `clause` | Clause chunks plus article introductions | `<output-base>/clause/` |

Baseline chunks retain document name/number, page range, and `source_refs` with the source legal hierarchy. Fixed-size and paragraph chunks can reference multiple Articles; clause chunks also retain article-introduction text. Use a new `--output-base` for each run.

The default dense model remains `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. BGE-M3 is available as an explicit alternative via `--model BAAI/bge-m3` or the [BGE-M3 dense config](configs/retrieval/dense_bge_m3.yaml). The current dense index uses BGE-M3's sentence-transformer dense embeddings only; its sparse and multi-vector retrieval modes are not implemented here.

For a same-length GPU comparison of both dense models across the current chunk sets, upload `data/chunks/run07_colab_inputs.zip` to the Colab runtime and clone this repository. Install the project and `sentence-transformers`, extract the archive to `/content/input`, then run:

```bash
python scripts/build_colab_embeddings.py \
  --input-root /content/input/chunks \
  --output-root /content/drive/MyDrive/LegalRAG/indexes/run07_maxlen256 \
  --device cuda \
  --max-length 256
```

The script saves separate indexes per model and chunk strategy plus an `embedding_manifest.json`; it requires a CUDA runtime by default and refuses to overwrite existing indexes. Mount Google Drive first if using the example output path.

## Retrieval Methods

| Method | Description |
|--------|-------------|
| `bm25` | Lexical search with Vietnamese tokenization |
| `dense` | Semantic search with multilingual embeddings |
| `hybrid` | RRF or weighted fusion of BM25 + Dense |
| `rerank` | Cross-encoder reranking of hybrid results |

## Evaluation

```bash
# Gold coverage: every gold reference must exist in the chunk corpus
legal-rag evaluate coverage data/qa/test.jsonl --chunks data/chunks/<run_id>/all_legal_aware.jsonl

# Retrieval evaluation (fixed hit definitions; see evaluation/retrieval.py)
legal-rag evaluate retrieval data/qa/test.jsonl \
  --chunks data/chunks/<run_id>/all_legal_aware.jsonl \
  --method bm25 --k 1,3,5,10,20 --level article

# Generation evaluation
legal-rag evaluate generation output/predictions.jsonl

# End-to-end evaluation
legal-rag evaluate end_to_end data/qa/test.jsonl
```

Metric definitions are written into the output JSON (`metrics.definitions`) and
must not change between runs: `hit_rate_at_k` is the fraction of queries with at
least one relevant chunk in the top-k, while `ref_recall_at_k` counts distinct
gold references. `--level clause` additionally requires a matching clause.

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run with coverage
pytest --cov=src/legal_rag tests/
```

## Code Quality

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/
```

## Experiment Tracking

Each run creates a directory under `experiments/runs/{run_id}/` with:
- `config_snapshot.yaml` - Full config used
- `metrics.json` - Evaluation metrics
- `predictions.jsonl` - Model predictions
- `logs/` - Execution logs

## Data Versioning

DVC is planned but not enabled yet. Source PDFs and run outputs (`data/processed/`, `data/chunks/`, `data/indexes/`) are git-ignored and reproducible via `legal-rag pipeline`.

## Disclaimer

⚠️ **This system is for research purposes only. It does not replace professional legal advice.**

## License

MIT License
