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
│       ├── corpus/         # Corpus extraction + page patches
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

# 2. Run all baseline chunking strategies on the run's corpus
legal-rag chunk all_baselines data/processed/labor_code_2019_run07_corpus.jsonl

# 3. Build BM25 / dense indexes from the run's chunks
legal-rag index bm25 data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag index dense data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl

# 4. Retrieval (--chunks is required, so a stale run is never picked silently)
legal-rag retrieve bm25 "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag retrieve hybrid "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag retrieve rerank "nghỉ phép năm bao nhiêu ngày" --chunks data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl

# 5. Validate / clean chunks
legal-rag chunk validate data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
legal-rag chunk clean data/chunks/legal_aware_run07/labor_code_2019_legal_aware.jsonl
```

### Using Config File

```bash
# Run with custom config
legal-rag --config configs/my_experiment.yaml parse pdf ...
```

## Chunking Strategies

| Strategy | Description | Output Dir |
|----------|-------------|------------|
| `legal_aware` | Preserves legal hierarchy (Chapter→Section→Article→Clause→Point); standard output of `legal-rag pipeline` | `data/chunks/legal_aware_<run_id>/` |
| `fixed_300/500/800` | Fixed-size with overlap | `data/chunks/fixed_{size}/` |
| `paragraph` | Paragraph-based | `data/chunks/paragraph/` |
| `article` | One chunk per article | `data/chunks/article/` |
| `clause` | One chunk per clause | `data/chunks/clause/` |

## Retrieval Methods

| Method | Description |
|--------|-------------|
| `bm25` | Lexical search with Vietnamese tokenization |
| `dense` | Semantic search with multilingual embeddings |
| `hybrid` | RRF or weighted fusion of BM25 + Dense |
| `rerank` | Cross-encoder reranking of hybrid results |

## Evaluation

```bash
# Retrieval evaluation
legal-rag evaluate retrieval data/qa/test.jsonl

# Generation evaluation
legal-rag evaluate generation output/predictions.jsonl

# End-to-end evaluation
legal-rag evaluate end_to_end data/qa/test.jsonl
```

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