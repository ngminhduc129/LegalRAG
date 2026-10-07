"""Configuration management with Pydantic validation."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_yaml import parse_yaml_raw_as


class CorpusConfig(BaseModel):
    document_id: str
    pdf_path: Path
    document_name: str
    document_number: str
    expected_articles: int

    @field_validator("pdf_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


class LegalAwareChunkingConfig(BaseModel):
    enabled: bool = True
    preserve_hierarchy: bool = True
    chunk_types: List[str] = Field(
        default_factory=lambda: ["article_intro", "clause", "clause_intro", "point"]
    )


class FixedSizeChunkingConfig(BaseModel):
    chunk_size: int
    overlap: int


class BaselineChunkingConfig(BaseModel):
    fixed_size: List[FixedSizeChunkingConfig] = Field(default_factory=list)
    paragraph: bool = True
    article: bool = True
    clause: bool = True


class ChunkingConfig(BaseModel):
    legal_aware: LegalAwareChunkingConfig = Field(default_factory=LegalAwareChunkingConfig)
    baselines: BaselineChunkingConfig = Field(default_factory=BaselineChunkingConfig)


class BM25Config(BaseModel):
    enabled: bool = True
    tokenizer: str = "underthesea"
    k1: float = 1.5
    b: float = 0.75


class DenseConfig(BaseModel):
    enabled: bool = True
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cuda"
    batch_size: int = 32


class HybridConfig(BaseModel):
    enabled: bool = True
    fusion_method: str = "rrf"
    rrf_k: int = 60
    weight_bm25: float = 0.5
    weight_dense: float = 0.5


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 50


class RetrievalConfig(BaseModel):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class EvaluationConfig(BaseModel):
    retrieval_k: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    metrics: List[str] = Field(default_factory=lambda: ["recall", "precision", "mrr", "ndcg"])
    generation_metrics: List[str] = Field(
        default_factory=lambda: [
            "correctness",
            "faithfulness",
            "citation_accuracy",
            "completeness",
            "hallucination_rate",
        ]
    )
    test_set_path: Path = Path("data/qa/test.jsonl")
    dev_set_path: Path = Path("data/qa/dev.jsonl")

    @field_validator("test_set_path", "dev_set_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


class PromptConfig(BaseModel):
    system: str = "You are a legal assistant for Vietnamese Labor Law. Answer based ONLY on the provided context. Cite specific Articles/Clauses/Points."
    template: str = (
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"
    )


class GenerationConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    citation_format: str = "[Điều {article}, Khoản {clause}, Điểm {point}]"


class ExperimentConfig(BaseModel):
    seed: int = 42
    run_id: str = "auto"
    output_dir: Path = Path("experiments/runs")
    log_level: str = "INFO"

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


class LegalRAGConfig(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LegalRAGConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = parse_yaml_raw_as(Dict[str, Any], f.read())
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_yaml())


def load_config(config_path: str | Path = "configs/base.yaml") -> LegalRAGConfig:
    """Load configuration from YAML file."""
    return LegalRAGConfig.from_yaml(config_path)


def merge_configs(base: LegalRAGConfig, overrides: Dict[str, Any]) -> LegalRAGConfig:
    """Merge base config with overrides (for experiment variations)."""
    base_dict = base.model_dump()
    _deep_merge(base_dict, overrides)
    return LegalRAGConfig(**base_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value