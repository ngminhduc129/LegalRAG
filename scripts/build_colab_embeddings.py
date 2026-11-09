"""Build reproducible dense indexes for all run07 chunk sets on a Colab GPU."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from legal_rag.retrieval.dense import DenseIndex


MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "BAAI--bge-m3": "BAAI/bge-m3",
}

CHUNK_FILES = (
    Path("legal_aware_run06/labor_code_2019_legal_aware.jsonl"),
    Path("run07/fixed_300/labor_code_2019_fixed_300.jsonl"),
    Path("run07/fixed_500/labor_code_2019_fixed_500.jsonl"),
    Path("run07/fixed_800/labor_code_2019_fixed_800.jsonl"),
    Path("run07/paragraph/labor_code_2019_paragraph.jsonl"),
    Path("run07/article/labor_code_2019_article.jsonl"),
    Path("run07/clause/labor_code_2019_clause.jsonl"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; connect a Colab GPU runtime before embedding")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero")
    if args.max_length != 256:
        raise ValueError("This comparison requires max-length=256 for both models")

    datasets = [(relative, args.input_root / relative) for relative in CHUNK_FILES]
    missing = [str(path) for _, path in datasets if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing chunk JSONL files:\n" + "\n".join(missing))

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "transformers_version": importlib.metadata.version("transformers"),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "indexes": [],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_dir, model_name in MODELS.items():
        print(f"Loading {model_name} on {args.device}")
        index = DenseIndex(
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        if index.model.max_seq_length != args.max_length:
            raise RuntimeError(
                f"Model sequence length is {index.model.max_seq_length}, "
                f"expected {args.max_length}"
            )

        for relative_path, chunks_path in datasets:
            chunks = load_chunks(chunks_path)
            if not chunks:
                raise ValueError(f"No chunks found in {chunks_path}")

            dataset_name = relative_path.parent.as_posix().replace("/", "_")
            output_path = args.output_root / model_dir / dataset_name / "dense_index.pkl"
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing index: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            index.add_documents(chunks)
            index.save(output_path)
            manifest["indexes"].append({
                "model": model_name,
                "dataset": relative_path.as_posix(),
                "chunk_count": len(chunks),
                "chunks_sha256": sha256_file(chunks_path),
                "index_path": output_path.relative_to(args.output_root).as_posix(),
                "index_sha256": sha256_file(output_path),
                "max_length": index.max_length,
            })
            del chunks
            print(f"Completed {relative_path} -> {output_path}")

        del index
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    manifest_path = args.output_root / "embedding_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
