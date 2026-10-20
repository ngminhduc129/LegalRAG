# Legal RAG Architecture

## High-Level Data Flow

```mermaid
flowchart TD
    %% Data Layer
    RawPDF["📄 Raw PDF\n(Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf)"]
    Processed["📦 Processed Corpus\n(data/processed/*_corpus.jsonl)"]
    Chunks["✂️ Chunks\n(data/chunks/legal_aware_{run_id}/*.jsonl)"]
    Indexes["🔍 Indexes\n(data/indexes/{bm25,dense}/*.pkl)"]
    QA["❓ Ground Truth QA\n(data/qa/{train,dev,test}.jsonl) (planned)"]

    %% Processing Pipeline
    Parse["📖 Parse\nlegal_rag.parse.pdf"]
    Chunk["✂️ Chunk\nlegal_rag.chunk.*"]
    Index["📇 Index\nlegal_rag.index.*"]

    %% Retrieval
    BM25["🔤 BM25\n(Lexical)"]
    Dense["🧠 Dense\n(Semantic)"]
    Hybrid["⚡ Hybrid\n(RRF/Weighted)"]
    Rerank["🎯 Reranker\n(Cross-encoder)"]

    %% Generation
    Retrieve["🔎 Retrieve\nlegal_rag.retrieve.*"]
    Generate["🤖 Generate\nlegal_rag.generate.*"]

    %% Evaluation
    EvalRet["📊 Eval Retrieval\nlegal_rag.evaluate.retrieval"]
    EvalGen["📝 Eval Generation\nlegal_rag.evaluate.generation"]
    EvalE2E["🔄 Eval E2E\nlegal_rag.evaluate.end_to_end"]

    %% Config
    Config["⚙️ Configs\n(configs/*.yaml)"]

    %% Experiment Tracking
    Exp["🧪 Experiments\n(experiments/runs/{run_id}/)"]

    %% Connections
    RawPDF --> Parse
    Parse --> Processed
    Processed --> Chunk
    Chunk --> Index
    Index --> Indexes
    
    Config -.-> Parse
    Config -.-> Chunk
    Config -.-> Index

    Chunks --> BM25
    Chunks --> Dense
    BM25 --> Hybrid
    Dense --> Hybrid
    Hybrid --> Rerank

    QA --> Retrieve
    Retrieve --> Hybrid
    Rerank --> Retrieve
    Retrieve --> Generate
    Generate --> EvalGen
    Retrieve --> EvalRet
    EvalRet --> Exp
    EvalGen --> Exp
    EvalE2E --> Exp

    style RawPDF fill:#f9f,stroke:#333
    style Processed fill:#bbf,stroke:#333
    style Chunks fill:#bfb,stroke:#333
    style Indexes fill:#ffb,stroke:#333
    style QA fill:#fbf,stroke:#333
```

## Module Architecture

```mermaid
flowchart LR
    subgraph CLI["🖥️ CLI (Typer)"]
        ParseCmd["parse"]
        ChunkCmd["chunk"]
        IndexCmd["index"]
        RetrieveCmd["retrieve"]
        GenerateCmd["generate"]
        EvaluateCmd["evaluate"]
    end

    subgraph Core["⚙️ Core Package (legal_rag)"]
        Config["config.py\n(Pydantic)"]
        
        subgraph Parsing["📖 parsing/"]
            Classifier["classifier.py\n(LineClassifier)"]
            Parser["parser.py\n(LegalStructureParser)"]
            Constants["constants.py"]
        end

        subgraph Chunking["✂️ chunking/"]
            Base["base.py\n(BaseChunker)"]
            LegalAware["legal_aware.py\n(LegalAwareChunker)"]
            Fixed["legal_aware.py\n(FixedSizeChunker)"]
            Para["legal_aware.py\n(ParagraphChunker)"]
            Article["legal_aware.py\n(ArticleChunker)"]
            Clause["legal_aware.py\n(ClauseChunker)"]
            Cleaner["cleaner.py"]
            Validator["validator.py"]
        end

        subgraph Corpus["📦 corpus/"]
            Extractor["extractor.py"]
            Patches["patches.py\n(page patches)"]
        end

        subgraph Retrieval["🔍 retrieval/"]
            BaseR["base.py\n(BaseRetriever)"]
            BM25["bm25.py\n(BM25Index)"]
            Dense["dense.py\n(DenseIndex)"]
            Hybrid["hybrid.py\n(HybridRetriever)"]
            Reranker["reranker.py\n(Reranker)"]
        end

        Pipeline["🔁 pipeline.py\n(run_pipeline)"]

        subgraph GenEval["🤖📊 generation/ / evaluation/"]
            Stub["(Planned - not implemented)"]
        end
    end

    subgraph Data["💾 Data (DVC)"]
        Raw["raw/"]
        Proc["processed/"]
        ChunksD["chunks/{strategy}/"]
        IndexesD["indexes/{bm25,dense}/"]
        QAD["qa/"]
    end

    subgraph Configs["⚙️ Configs"]
        BaseY["base.yaml"]
        ChunkY["chunking/*.yaml"]
        RetrY["retrieval/*.yaml"]
        GenY["generation.yaml"]
        EvalY["evaluation.yaml"]
        ExpY["experiment.yaml"]
    end

    CLI --> Core
    Core --> Data
    Configs -.-> Core
```

## Chunking Strategies Comparison

```mermaid
graph TD
    Corpus["📦 Parsed Corpus\n(JSONL lines)"]
    
    LegalAware["🎯 Legal-Aware\nPreserves hierarchy:\nChapter→Section→Article→Clause→Point\nChunk types: article_intro, clause, clause_intro, point"]
    
    Fixed300["📏 Fixed-300\n300 tokens, 60 overlap"]
    Fixed500["📏 Fixed-500\n500 tokens, 100 overlap"]
    Fixed800["📏 Fixed-800\n800 tokens, 160 overlap"]
    
    Paragraph["📝 Paragraph\nSplit by double newline"]
    Article["📄 Article\nOne chunk per article"]
    Clause["📋 Clause\nOne chunk per clause"]
    
    Corpus --> LegalAware
    Corpus --> Fixed300
    Corpus --> Fixed500
    Corpus --> Fixed800
    Corpus --> Paragraph
    Corpus --> Article
    Corpus --> Clause
    
    LegalAware --> LegalChunks["data/chunks/legal_aware_{run_id}/"]
    Fixed300 --> Fix300Chunks["data/chunks/fixed_300/"]
    Fixed500 --> Fix500Chunks["data/chunks/fixed_500/"]
    Fixed800 --> Fix800Chunks["data/chunks/fixed_800/"]
    Paragraph --> ParaChunks["data/chunks/paragraph/"]
    Article --> ArtChunks["data/chunks/article/"]
    Clause --> ClaChunks["data/chunks/clause/"]
```

## Retrieval Pipeline

```mermaid
flowchart LR
    Query["❓ User Query"]
    
    subgraph BM25Path["🔤 BM25 Path"]
        BM25Tok["Tokenize\n(underthesea/pyvi)"]
        BM25Score["BM25 Score\n(k1=1.5, b=0.75)"]
        BM25Top["Top-K"]
    end
    
    subgraph DensePath["🧠 Dense Path"]
        DenseEnc["Encode\n(SentenceTransformer)"]
        DenseSim["Cosine Similarity\n(normalized)"]
        DenseTop["Top-K"]
    end
    
    subgraph Fusion["⚡ Fusion"]
        RRF["RRF\n(k=60)"]
        Weighted["Weighted\n(0.5/0.5)"]
    end
    
    subgraph Rerank["🎯 Reranker (Optional)"]
        CrossEnc["Cross-Encoder\n(ms-marco-MiniLM-L-6-v2)"]
        ReRank["Re-rank\nTop-50 → Top-K"]
    end
    
    Query --> BM25Tok
    Query --> DenseEnc
    BM25Tok --> BM25Score --> BM25Top
    DenseEnc --> DenseSim --> DenseTop
    BM25Top --> RRF
    DenseTop --> RRF
    BM25Top --> Weighted
    DenseTop --> Weighted
    RRF --> Rerank
    Weighted --> Rerank
    CrossEnc --> ReRank
    ReRank --> Results["📋 Final Results\n(chunk_id, score, metadata)"]
```

## Experiment Tracking

```mermaid
flowchart TD
    Run["🏃 Run Experiment"]
    Config["⚙️ Config Snapshot\n(configs/base.yaml + overrides)"]
    Seed["🎲 Seed (42)"]
    Git["📝 Git Commit"]
    
    Pipeline["⚙️ Pipeline\n(parse → chunk → index → retrieve → generate → evaluate)"]
    
    Metrics["📊 Metrics\n(metrics.json)"]
    Preds["📝 Predictions\n(predictions.jsonl)"]
    Logs["📜 Logs\n(logs/)"]
    
    Reports["📈 Reports\n(reports/)"]
    
    Run --> Config
    Run --> Seed
    Run --> Git
    Config --> Pipeline
    Seed --> Pipeline
    Git --> Pipeline
    Pipeline --> Metrics
    Pipeline --> Preds
    Pipeline --> Logs
    Metrics --> Reports
    Preds --> Reports
```

## Directory Structure

```
LegalRAG/
├── configs/                          # Configuration (YAML)
│   ├── base.yaml                     # Main config
│   ├── chunking/                     # Chunking strategies
│   ├── retrieval/                    # Retrieval configs
│   ├── generation.yaml               # Generation config
│   ├── evaluation.yaml               # Evaluation config
│   └── experiment.yaml               # Experiment tracking
├── data/                             # Data (run outputs are git-ignored)
│   ├── raw/                          # Source PDFs (read-only)
│   ├── ocr/                          # OCR evidence + page patches
│   │   ├── patches/                  # JSON page patches (page-86 fix)
│   │   └── renders/                  # Verification renders (local, ignored)
│   ├── processed/                    # Parsed corpus per run (JSONL)
│   ├── chunks/                       # legal_aware_<run_id>/ (pipeline); baselines on demand
│   ├── indexes/                      # Retrieval indexes (generated)
│   └── qa/                           # Ground truth (planned, frozen)
├── src/
│   └── legal_rag/                    # Installable package
│       ├── config.py                 # Pydantic config management
│       ├── parsing/                  # PDF parsing & legal structure
│       ├── chunking/                 # 5 chunking strategies
│       ├── corpus/                   # Corpus extraction + page patches
│       ├── retrieval/                # BM25, Dense, Hybrid, Reranker
│       ├── pipeline.py               # Standard end-to-end build
│       ├── generation/               # LLM answer generation (planned)
│       ├── evaluation/               # Metrics & evaluation (planned)
│       └── cli/                      # Typer CLI commands
├── tests/                            # Unit & integration tests
│   ├── unit/                         # Unit tests
│   └── integration/
├── notebooks/                        # EDA & analysis
├── reports/                          # Decision logs / generated reports
├── experiments/                      # Experiment outputs (auto-generated, planned)
├── pyproject.toml                    # Package config
├── requirements.txt                  # Production deps
├── requirements-dev.txt              # Dev deps
├── README.md
├── ARCHITECTURE.md
└── AGENTS.md                         # Project rules
```

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Reproducibility** | Config snapshots, git commit, seed, frozen test set |
| **Fair Comparison** | Same corpus, same queries, same k, only chunking strategy varies |
| **Legal Structure Preservation** | Legal-aware chunking keeps Chapter→Section→Article→Clause→Point |
| **Traceability** | Every chunk links to document_id, page, article, clause, point |
| **No Data Leakage** | Gold labels never in index/prompt; train/dev/test split |
| **Modular Config** | Override single component via YAML without code changes |
| **Type Safety** | Pydantic config validation, typed interfaces |
| **Testability** | Abstract base classes, unit tests for each module |