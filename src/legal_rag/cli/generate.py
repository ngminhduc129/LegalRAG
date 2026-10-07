"""Generate CLI - Generate answers using LLM."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(name="generate", help="Generate answers using LLM")
console = Console()


@app.command()
def answer(
    query: str = typer.Argument(..., help="Question to answer"),
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
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens"),
    fusion: str = typer.Option("rrf", "--fusion", help="Fusion method: rrf or weighted"),
):
    """Generate answer for a single query."""
    console.print(Panel(f"[bold]Question:[/bold] {query}", title="Legal RAG"))

    # TODO: Implement LLM client and answer generation
    console.print("[yellow]Not implemented yet. Use retrieve + manual LLM call for now.[/yellow]")
    console.print("Steps:")
    console.print("  1. legal-rag retrieve hybrid --query \"...\"")
    console.print("  2. Copy context to your LLM")
    console.print("  3. Use prompt template from config")


@app.command()
def batch(
    queries_path: Path = typer.Argument(
        ...,
        help="Path to JSONL file with queries",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        Path("output/answers.jsonl"),
        "--output",
        "-o",
        help="Output path for answers",
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
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens"),
):
    """Generate answers for a batch of queries."""
    console.print("[yellow]Not implemented yet.[/yellow]")


if __name__ == "__main__":
    app()