"""Index CLI - Build retrieval indexes."""

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="index", help="Build BM25 and dense retrieval indexes")
console = Console()


@app.command()
def bm25(
    chunks_path: Path = typer.Argument(
        ...,
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    index_path: Path = typer.Option(
        Path("data/indexes/bm25/bm25_index.pkl"),
        "--output",
        "-o",
        help="Output path for BM25 index",
    ),
    k1: float = typer.Option(1.5, "--k1", help="BM25 k1 parameter"),
    b: float = typer.Option(0.75, "--b", help="BM25 b parameter"),
):
    """Build BM25 index from chunks."""
    from legal_rag.retrieval.bm25 import build_bm25_index

    console.print(f"[bold]Building BM25 index:[/bold] {chunks_path} -> {index_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building index...", total=None)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        build_bm25_index(chunks_path, index_path, k1=k1, b=b)
        progress.update(task, description="Done")

    console.print(f"[green]✓[/green] BM25 index saved to {index_path}")


@app.command()
def dense(
    chunks_path: Path = typer.Argument(
        ...,
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    index_path: Path = typer.Option(
        Path("data/indexes/dense/dense_index.pkl"),
        "--output",
        "-o",
        help="Output path for dense index",
    ),
    model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "--model",
        "-m",
        help="Sentence transformer model name",
    ),
    device: str = typer.Option(
        None,
        "--device",
        help="Device (cuda/cpu). Default: auto-detect",
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for encoding"),
):
    """Build dense (embedding) index from chunks."""
    from legal_rag.retrieval.dense import build_dense_index

    console.print(f"[bold]Building dense index:[/bold] {chunks_path} -> {index_path}")
    console.print(f"[bold]Model:[/bold] {model}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building index...", total=None)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        build_dense_index(
            chunks_path,
            index_path,
            model_name=model,
            device=device,
            batch_size=batch_size,
        )
        progress.update(task, description="Done")

    console.print(f"[green]✓[/green] Dense index saved to {index_path}")


@app.command()
def hybrid(
    chunks_path: Path = typer.Argument(
        ...,
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    bm25_path: Path = typer.Option(
        Path("data/indexes/bm25/bm25_index.pkl"),
        "--bm25-path",
        help="BM25 index path",
    ),
    dense_path: Path = typer.Option(
        Path("data/indexes/dense/dense_index.pkl"),
        "--dense-path",
        help="Dense index path",
    ),
    fusion: str = typer.Option("rrf", "--fusion", help="Fusion method: rrf or weighted"),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF k parameter"),
    weight_bm25: float = typer.Option(0.5, "--weight-bm25", help="Weight for BM25 (weighted fusion)"),
    weight_dense: float = typer.Option(0.5, "--weight-dense", help="Weight for dense (weighted fusion)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild indexes if they exist"),
):
    """Build both BM25 and dense indexes for hybrid retrieval."""
    from legal_rag.retrieval.bm25 import build_bm25_index
    from legal_rag.retrieval.dense import build_dense_index

    console.print(f"[bold]Building hybrid indexes:[/bold]")
    console.print(f"  Chunks: {chunks_path}")
    console.print(f"  BM25:   {bm25_path}")
    console.print(f"  Dense:  {dense_path}")
    console.print(f"  Fusion: {fusion}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # BM25
        task1 = progress.add_task("Building BM25...", total=None)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        if bm25_path.exists() and not rebuild:
            progress.update(task1, description="BM25 exists, skipping")
        else:
            build_bm25_index(chunks_path, bm25_path)
            progress.update(task1, description="BM25 done")

        # Dense
        task2 = progress.add_task("Building dense...", total=None)
        dense_path.parent.mkdir(parents=True, exist_ok=True)
        if dense_path.exists() and not rebuild:
            progress.update(task2, description="Dense exists, skipping")
        else:
            build_dense_index(chunks_path, dense_path)
            progress.update(task2, description="Dense done")

    console.print(f"[green]✓[/green] Hybrid indexes ready")


if __name__ == "__main__":
    app()