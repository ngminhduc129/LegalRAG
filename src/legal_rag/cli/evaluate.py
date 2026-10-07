"""Evaluate CLI - Evaluate retrieval and generation."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="evaluate", help="Evaluate retrieval and generation quality")
console = Console()


@app.command()
def retrieval(
    test_set: Path = typer.Argument(
        ...,
        help="Path to test set JSONL (query, relevant_article_ids)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
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
    k_values: str = typer.Option("1,3,5,10,20", "--k", help="Comma-separated k values for Recall@k"),
    fusion: str = typer.Option("rrf", "--fusion", help="Fusion method: rrf or weighted"),
    output: Path = typer.Option(
        Path("output/retrieval_eval.json"),
        "--output",
        "-o",
        help="Output JSON file for metrics",
    ),
):
    """Evaluate retrieval on test set."""
    console.print("[yellow]Not implemented yet.[/yellow]")
    console.print("Will compute Recall@k, Precision@k, MRR, NDCG for each k")


@app.command()
def generation(
    predictions_path: Path = typer.Argument(
        ...,
        help="Path to predictions JSONL (query, answer, citations)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    test_set: Path = typer.Option(
        Path("data/qa/test.jsonl"),
        "--test-set",
        "-t",
        help="Path to test set with reference answers",
    ),
    metrics: str = typer.Option(
        "correctness,faithfulness,citation_accuracy,completeness,hallucination_rate",
        "--metrics",
        "-m",
        help="Comma-separated metrics to compute",
    ),
    output: Path = typer.Option(
        Path("output/generation_eval.json"),
        "--output",
        "-o",
        help="Output JSON file for metrics",
    ),
):
    """Evaluate generation quality."""
    console.print("[yellow]Not implemented yet.[/yellow]")
    console.print("Will compute correctness, faithfulness, citation accuracy, completeness, hallucination rate")


@app.command()
def end_to_end(
    test_set: Path = typer.Argument(
        ...,
        help="Path to test set JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
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
    k: int = typer.Option(5, "--top-k", "-k", help="Number of context chunks"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model name"),
    output: Path = typer.Option(
        Path("output/e2e_eval.json"),
        "--output",
        "-o",
        help="Output JSON file for metrics",
    ),
):
    """Run full end-to-end evaluation (retrieve + generate + evaluate)."""
    console.print("[yellow]Not implemented yet.[/yellow]")


if __name__ == "__main__":
    app()