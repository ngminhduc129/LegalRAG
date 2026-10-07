"""Retrieve CLI - Search and retrieve documents."""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="retrieve", help="Retrieve relevant chunks for queries")
console = Console()


@app.command()
def bm25(
    query: str = typer.Argument(..., help="Search query"),
    chunks_path: Path = typer.Option(
        Path("data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl"),
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
    ),
    index_path: Path = typer.Option(
        Path("data/indexes/bm25/bm25_index.pkl"),
        "--index",
        "-i",
        help="Path to BM25 index",
    ),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    k1: float = typer.Option(1.5, "--k1", help="BM25 k1 parameter"),
    b: float = typer.Option(0.75, "--b", help="BM25 b parameter"),
):
    """Search using BM25 only."""
    from legal_rag.retrieval.bm25 import BM25Index, build_bm25_index
    import json

    console.print(f"[bold]BM25 Search:[/bold] '{query}'")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading index...", total=None)
        if index_path.exists():
            index = BM25Index.load(index_path)
        else:
            index = build_bm25_index(chunks_path, index_path, k1=k1, b=b)
        progress.update(task, description="Searching...")

        results = index.search(query, k=k)

    _display_results(results, "BM25")


@app.command()
def dense(
    query: str = typer.Argument(..., help="Search query"),
    chunks_path: Path = typer.Option(
        Path("data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl"),
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
    ),
    index_path: Path = typer.Option(
        Path("data/indexes/dense/dense_index.pkl"),
        "--index",
        "-i",
        help="Path to dense index",
    ),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "--model",
        "-m",
        help="Sentence transformer model name",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cpu)"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
):
    """Search using dense (embedding) only."""
    from legal_rag.retrieval.dense import DenseIndex, build_dense_index

    console.print(f"[bold]Dense Search:[/bold] '{query}'")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading index...", total=None)
        if index_path.exists():
            index = DenseIndex.load(index_path)
        else:
            index = build_dense_index(chunks_path, index_path, model_name=model, device=device, batch_size=batch_size)
        progress.update(task, description="Searching...")

        results = index.search(query, k=k)

    _display_results(results, "Dense")


@app.command()
def hybrid(
    query: str = typer.Argument(..., help="Search query"),
    chunks_path: Path = typer.Option(
        Path("data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl"),
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
    ),
    bm25_path: Path = typer.Option(
        Path("data/indexes/bm25/bm25_index.pkl"),
        "--bm25-index",
        help="Path to BM25 index",
    ),
    dense_path: Path = typer.Option(
        Path("data/indexes/dense/dense_index.pkl"),
        "--dense-index",
        help="Path to dense index",
    ),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of final results"),
    fetch_k: int = typer.Option(50, "--fetch-k", help="Number of results to fetch from each system"),
    fusion: str = typer.Option("rrf", "--fusion", help="Fusion method: rrf or weighted"),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF k parameter"),
    weight_bm25: float = typer.Option(0.5, "--weight-bm25", help="BM25 weight (weighted fusion)"),
    weight_dense: float = typer.Option(0.5, "--weight-dense", help="Dense weight (weighted fusion)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild indexes if needed"),
):
    """Search using hybrid (BM25 + dense) retrieval."""
    from legal_rag.retrieval.hybrid import HybridRetriever

    console.print(f"[bold]Hybrid Search:[/bold] '{query}' (fusion={fusion})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading retriever...", total=None)
        config = {
            "fusion_method": fusion,
            "rrf_k": rrf_k,
            "weight_bm25": weight_bm25,
            "weight_dense": weight_dense,
        }
        retriever = HybridRetriever.from_config(
            chunks_path=chunks_path,
            bm25_index_path=bm25_path,
            dense_index_path=dense_path,
            config=config,
            rebuild=rebuild,
        )
        progress.update(task, description="Searching...")

        results = retriever.search(query, k=k, fetch_k=fetch_k)

    _display_results(results, f"Hybrid ({fusion})")


@app.command()
def rerank(
    query: str = typer.Argument(..., help="Search query"),
    chunks_path: Path = typer.Option(
        Path("data/chunks/legal_aware/labor_code_2019_legal_aware.jsonl"),
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
    ),
    bm25_path: Path = typer.Option(
        Path("data/indexes/bm25/bm25_index.pkl"),
        "--bm25-index",
        help="Path to BM25 index",
    ),
    dense_path: Path = typer.Option(
        Path("data/indexes/dense/dense_index.pkl"),
        "--dense-index",
        help="Path to dense index",
    ),
    k: int = typer.Option(10, "--top-k", "-k", help="Final number of results"),
    fetch_k: int = typer.Option(50, "--fetch-k", help="Initial fetch from each system"),
    rerank_k: int = typer.Option(20, "--rerank-k", help="Number of results to rerank"),
    model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--model",
        "-m",
        help="Cross-encoder model name",
    ),
    fusion: str = typer.Option("rrf", "--fusion", help="Fusion method: rrf or weighted"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild indexes if needed"),
):
    """Search with hybrid retrieval + cross-encoder reranking."""
    from legal_rag.retrieval.hybrid import HybridRetriever
    from legal_rag.retrieval.reranker import Reranker

    console.print(f"[bold]Hybrid + Rerank:[/bold] '{query}'")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading retriever...", total=None)
        config = {
            "fusion_method": fusion,
            "rrf_k": 60,
            "weight_bm25": 0.5,
            "weight_dense": 0.5,
        }
        retriever = HybridRetriever.from_config(
            chunks_path=chunks_path,
            bm25_index_path=bm25_path,
            dense_index_path=dense_path,
            config=config,
            rebuild=rebuild,
        )
        progress.update(task, description="Initial search...")

        initial_results = retriever.search(query, k=fetch_k, fetch_k=fetch_k)

        task2 = progress.add_task(f"Reranking top {rerank_k}...", total=None)
        reranker = Reranker(model_name=model)
        results = reranker.rerank(query, initial_results[:rerank_k], top_k=k)
        progress.update(task2, description="Done")

    _display_results(results, "Hybrid + Rerank")


def _display_results(
    results: List[tuple],
    method: str,
):
    """Display search results in a table."""
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"{method} Results (Top {len(results)})")
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Chunk ID", width=40)
    table.add_column("Article/Clause/Point", width=30)
    table.add_column("Content Preview", width=60)

    for i, (doc_id, score, doc) in enumerate(results, 1):
        location = ""
        if doc.get("article"):
            location += doc["article"]
        if doc.get("clause"):
            location += f" Khoản {doc['clause']}"
        if doc.get("point"):
            location += f" Điểm {doc['point']}"

        content = doc.get("content", "")[:120].replace("\n", " ") + "..."

        table.add_row(
            str(i),
            f"{score:.4f}",
            doc_id[:40],
            location[:30],
            content,
        )

    console.print(table)


if __name__ == "__main__":
    app()