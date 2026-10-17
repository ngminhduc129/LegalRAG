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
├── data/                    # Data directories (versioned with DVC)
│   ├── raw/                # Original PDF (read-only)
│   ├── processed/          # Parsed corpus (JSONL)
│   ├── chunks/             # Chunked data by strategy
│   │   ├── legal_aware/    # Legal-aware chunking (proposed)
│   │   ├── fixed_300/      # Fixed-size 300 tokens
│   │   ├── fixed_500/      # Fixed-size 500 tokens
│   │   ├── fixed_800/      # Fixed-size 800 tokens
│   │   ├── paragraph/      # Paragraph-based
│   │   ├── article/        # Article-level
│   │   └── clause/         # Clause-level
│   ├── indexes/            # Retrieval indexes
│   │   ├── bm25/           # BM25 indexes
│   │   └── dense/          # Dense vector indexes
│   └── qa/                 # Ground truth QA pairs (frozen)
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── src/                     # Source code (installable package)
│   └── legal_rag/
│       ├── __init__.py
│       ├── config.py       # Pydantic config management
│       ├── parsing/        # PDF parsing & legal structure
│       ├── chunking/       # Chunking strategies
│       ├── corpus/         # Corpus extraction
│       ├── retrieval/      # BM25, Dense, Hybrid, Reranker
│       ├── generation/     # LLM answer generation
│       ├── evaluation/     # Metrics & evaluation
│       ├── pipeline/       # End-to-end pipelines
│       └── utils/          # Utilities
├── src/cli/                 # CLI entry points (Typer)
├── scripts/                 # One-off scripts
├── experiments/             # Experiment outputs (auto-generated)
│   └── runs/
│       └── {run_id}/
│           ├── config_snapshot.yaml
│           ├── metrics.json
│           ├── predictions.jsonl
│           └── logs/
├── tests/                   # Unit & integration tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/               # EDA & analysis notebooks
├── reports/                 # Generated reports
├── pyproject.toml           # Package configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
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
legal-rag pipeline data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf --run-id run04

# 1. Parse PDF into structured corpus (page patches in data/ocr/patches are applied automatically)
legal-rag parse pdf data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf

# 2. Run legal-aware chunking
legal-rag chunk legal_aware data/processed/labor_code_2019_corpus.jsonl

# 3. Run all baseline chunking strategies
legal-rag chunk all_baselines data/processed/labor_code_2019_corpus.jsonl

# 4. Build BM25 index
legal-rag index bm25 data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl

# 5. Build dense (embedding) index
legal-rag index dense data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl

# 6. Hybrid retrieval
legal-rag retrieve hybrid "nghỉ phép năm bao nhiêu ngày"

# 7. Hybrid + Reranking
legal-rag retrieve rerank "nghỉ phép năm bao nhiêu ngày"

# 8. Validate chunks
legal-rag chunk validate data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl

# 9. Clean chunks
legal-rag chunk clean data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl
```

### Using Config File

```bash
# Run with custom config
legal-rag --config configs/my_experiment.yaml parse pdf ...
```

## Chunking Strategies

| Strategy | Description | Output Dir |
|----------|-------------|------------|
| `legal_aware` | Preserves legal hierarchy (Chapter→Section→Article→Clause→Point) | `data/chunks/legal_aware/` |
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

Use DVC for data versioning:

```bash
dvc init
dvc add data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf
dvc add data/qa/test.jsonl
```

## Disclaimer

⚠️ **This system is for research purposes only. It does not replace professional legal advice.**

## License

MIT License