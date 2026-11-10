"""QA CLI - validate the frozen evaluation set and map gold refs to chunks."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(name="qa", help="Validate QA data and map gold references to chunks")
console = Console()


@app.command("map")
def map_gold(
    test_set: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="QA JSONL file (see docs/qa_schema.md)",
    ),
    chunks_path: Path = typer.Option(
        ...,
        "--chunks",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Chunks JSONL from a corpus build (all_legal_aware.jsonl)",
    ),
    level: str = typer.Option(
        "article", "--level", help="Match gold refs at article | clause level"
    ),
    split: str = typer.Option(
        "", "--split", help="Only map this split (empty = all records)"
    ),
    output: Path = typer.Option(
        Path("output/qa_gold_chunks.jsonl"),
        "--output",
        "-o",
        help="Per-question gold chunk mapping (JSONL)",
    ),
    report: Optional[Path] = typer.Option(
        None,
        "--report",
        help="Optional JSON summary path",
    ),
):
    """Resolve every gold reference to chunk ids; fail if any reference is missing."""
    import json

    from legal_rag.qa.map import map_report

    try:
        result = map_report(
            qa_path=test_set,
            chunks_path=chunks_path,
            level=level,
            split=split or None,
            output_path=output,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    summary = {key: value for key, value in result.items() if key != "questions"}
    console.print(
        f"questions={result['total_questions']} "
        f"complete={result['complete_questions']} "
        f"question_coverage={result['question_coverage']:.4f} "
        f"ref_coverage={result['ref_coverage']:.4f} "
        f"gold_chunks={result['total_gold_chunk_ids']}"
    )
    console.print(f"[green]\u2713[/green] wrote {output}")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]\u2713[/green] wrote {report}")

    if result["incomplete_questions"]:
        console.print(
            f"[yellow]{result['incomplete_questions']} question(s) have unresolved "
            "gold references (see missing_refs in the output)[/yellow]"
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
