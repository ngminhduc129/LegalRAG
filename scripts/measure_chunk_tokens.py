"""Temporary script: measure token-length distribution of existing chunks JSONL.
Reads model from configs/base.yaml (retrieval.dense.model) and uses that model's
own sentence-transformers tokenizer. Outputs percentiles, oversize ratios, and
the 5 longest chunks. NOT committed unless explicitly requested.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("sentence-transformers not installed") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks" / "test_reg_labor" / "all_legal_aware.jsonl"


def load_config_model() -> tuple[str, int, int]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dense = cfg.get("retrieval", {}).get("dense", {})
    model = dense.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_length = int(dense.get("max_length", 256))
    batch_size = int(dense.get("batch_size", 32))
    return model, max_length, batch_size


def build_text(chunk: dict) -> str:
    """Mirror dense.py:add_documents text assembly (hierarchy + content)."""
    parts = []
    if chunk.get("chapter"):
        parts.append(chunk["chapter"])
    if chunk.get("section"):
        parts.append(chunk["section"])
    if chunk.get("article"):
        parts.append(chunk["article"])
    if chunk.get("article_title"):
        parts.append(chunk["article_title"])
    if chunk.get("clause"):
        parts.append(f"Khoản {chunk['clause']}")
    if chunk.get("point"):
        parts.append(f"Điểm {chunk['point']}")
    parts.append(chunk.get("content", ""))
    return " ".join(parts)


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def main() -> None:
    model_name, max_length, batch_size = load_config_model()
    print(f"Config file: {CONFIG_PATH}")
    print(f"Embedding model (from configs/base.yaml retrieval.dense.model): {model_name}")
    print(f"max_length: {max_length}  batch_size: {batch_size}")
    print(f"Chunks file: {CHUNKS_PATH}")
    print()

    if not CHUNKS_PATH.is_file():
        raise SystemExit(f"Chunks file not found: {CHUNKS_PATH}")

    model = SentenceTransformer(model_name, device="cpu")
    tokenizer = model.tokenizer

    chunks: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Total chunks read: {len(chunks)}")

    token_counts: list[int] = []
    for chunk in chunks:
        text = build_text(chunk)
        n = len(tokenizer.encode(text, add_special_tokens=True))
        token_counts.append(n)

    sorted_counts = sorted(token_counts)
    p50 = percentile(sorted_counts, 50)
    p90 = percentile(sorted_counts, 90)
    p99 = percentile(sorted_counts, 99)
    mx = max(token_counts) if token_counts else 0

    print(f"p50 tokens: {p50:.0f}")
    print(f"p90 tokens: {p90:.0f}")
    print(f"p99 tokens: {p99:.0f}")
    print(f"max tokens: {mx}")

    for threshold in (128, 256, 512):
        over = [c for c in token_counts if c > threshold]
        print(
            f"chunks > {threshold} tokens: {len(over)} / {len(token_counts)} "
            f"({100.0 * len(over) / len(token_counts):.1f}%)"
        )

    print()
    print("Top 5 longest chunks (chunk_id, tokens, article/clause/point):")
    indexed = sorted(
        zip(chunks, token_counts), key=lambda pair: pair[1], reverse=True
    )[:5]
    for chunk, n in indexed:
        article = chunk.get("article")
        clause = chunk.get("clause")
        point = chunk.get("point")
        print(
            f"  {chunk.get('chunk_id')} | {n} tokens | "
            f"article={article} clause={clause} point={point}"
        )


if __name__ == "__main__":
    main()
