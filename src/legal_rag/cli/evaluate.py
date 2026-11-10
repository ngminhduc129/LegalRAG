"""Evaluate CLI - Evaluate retrieval and generation."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="evaluate", help="Evaluate retrieval and generation quality")
console = Console()


def _parse_k_values(value: str) -> list[int]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise typer.BadParameter("at least one k value is required")
    return sorted(set(values))


@app.command()
def retrieval(
    test_set: Path = typer.Argument(
        ...,
        help="Path to test set JSONL (question, relevant_articles)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    chunks_path: Path = typer.Option(
        ...,
        "--chunks",
        "-c",
        help="Path to chunks JSONL (data/chunks/<run_id>/all_legal_aware.jsonl)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    method: str = typer.Option(
        "bm25", "--method", "-m", help="Retriever: bm25 | dense | hybrid"
    ),
    k_values: str = typer.Option(
        "1,3,5,10,20", "--k", help="Comma-separated k values"
    ),
    level: str = typer.Option(
        "article", "--level", help="Hit level: article | clause"
    ),
    split: str = typer.Option(
        "", "--split", help="Only evaluate this split (empty = all records)"
    ),
    fusion: str = typer.Option("rrf", "--fusion", help="Hybrid fusion: rrf | weighted"),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF k parameter"),
    fetch_k: int = typer.Option(50, "--fetch-k", help="Candidates per retriever"),
    dense_model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "--dense-model",
        help="Sentence-transformers model for dense/hybrid",
    ),
    output: Path = typer.Option(
        Path("output/retrieval_eval.json"),
        "--output",
        "-o",
        help="Output JSON file for metrics",
    ),
):
    """Evaluate retrieval on the frozen test set with fixed hit definitions."""
    from legal_rag.evaluation.retrieval import retrieval_eval_report

    try:
        result = retrieval_eval_report(
            qa_path=test_set,
            chunks_path=chunks_path,
            method=method,
            k_values=_parse_k_values(k_values),
            split=split or None,
            level=level,
            fusion=fusion,
            rrf_k=rrf_k,
            fetch_k=fetch_k,
            dense_model=dense_model,
        )
    except (ValueError, ImportError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = result["metrics"]
    table = Table(title=f"Retrieval ({method}, {level}-level)")
    table.add_column("k", justify="right")
    table.add_column("Hit@k", justify="right")
    table.add_column("RefRecall@k", justify="right")
    table.add_column("Precision@k", justify="right")
    table.add_column("nDCG@k", justify="right")
    for k in metrics["k_values"]:
        table.add_row(
            str(k),
            f"{metrics['hit_rate_at_k'][str(k)]:.4f}",
            f"{metrics['ref_recall_at_k'][str(k)]:.4f}",
            f"{metrics['precision_at_k'][str(k)]:.4f}",
            f"{metrics['ndcg_at_k'][str(k)]:.4f}",
        )
    console.print(table)
    console.print(f"MRR: {metrics['mrr']:.4f}")
    console.print(f"[green]\u2713[/green] wrote {output}")


@app.command()
def coverage(
    test_set: Path = typer.Argument(
        ...,
        help="Path to test set JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    chunks_path: Path = typer.Option(
        ...,
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    split: str = typer.Option(
        "", "--split", help="Only check this split (empty = all records)"
    ),
    level: str = typer.Option(
        "article", "--level", help="Coverage level: article | clause"
    ),
    output: Path = typer.Option(
        Path("output/coverage_report.json"),
        "--output",
        "-o",
        help="Output JSON file",
    ),
):
    """Check that every gold reference exists in the chunk corpus."""
    from legal_rag.evaluation.coverage import coverage_report

    try:
        result = coverage_report(
            qa_path=test_set,
            chunks_path=chunks_path,
            split=split or None,
            level=level,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(
        f"queries={result['total_queries']} "
        f"query_coverage={result['query_coverage']:.4f} "
        f"ref_coverage={result['ref_coverage']:.4f} "
        f"missing_refs={len(result['missing_refs'])}"
    )
    console.print(f"[green]\u2713[/green] wrote {output}")
    if result["missing_refs"]:
        raise typer.Exit(code=2)


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
        ...,
        "--chunks",
        "-c",
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
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
