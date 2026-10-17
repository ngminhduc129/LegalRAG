"""Pipeline CLI - standard end-to-end build (PDF to validated legal-aware chunks)."""

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def pipeline(
    pdf_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Source PDF file",
    ),
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="Unique run id (e.g. run04); existing runs are never overwritten",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
    patches: List[Path] = typer.Option(
        [Path("data/ocr/patches")],
        "--patches",
        "-p",
        help="Page patch files or directories (repeatable)",
    ),
    corpus_dir: Path = typer.Option(
        Path("data/processed"),
        "--corpus-dir",
        help="Output directory for the parsed corpus",
    ),
    chunks_dir: Path = typer.Option(
        Path("data/chunks"),
        "--chunks-dir",
        help="Base output directory for chunks",
    ),
):
    """Standard pipeline: PDF -> page patches -> corpus -> legal-aware chunks -> validation.

    All outputs are run-scoped (`<document_id>_<run_id>...`); reusing a run id
    fails instead of overwriting older results.  Validation errors make the
    command exit with code 1.
    """
    from legal_rag.corpus.patches import PatchError
    from legal_rag.pipeline import run_pipeline

    console.print(f"[bold]Pipeline:[/bold] {pdf_path} [dim](run {run_id})[/dim]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Build...", total=None)
            result = run_pipeline(
                pdf_path=pdf_path,
                run_id=run_id,
                document_id=document_id,
                patch_paths=patches,
                corpus_dir=corpus_dir,
                chunks_dir=chunks_dir,
            )
            progress.update(task, description="Done")
    except (FileExistsError, FileNotFoundError, PatchError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    for report in result["patch_reports"]:
        coverage = report.get("ocr_coverage")
        coverage_text = (
            f", OCR coverage {coverage:.4f}" if coverage is not None else ""
        )
        console.print(
            f"[cyan]patched page {report['page']}:[/cyan] "
            f"{report['replaced_lines']} extracted lines -> "
            f"{report['inserted_lines']} verified lines{coverage_text}"
        )

    console.print(
        f"lines: {result['raw_lines']} extracted -> {result['line_count']} after patches"
    )
    console.print(f"[green]\u2713[/green] corpus: {result['corpus_jsonl']}")
    console.print(
        f"[green]\u2713[/green] chunks: {result['chunks_jsonl']} "
        f"({result['final_chunks']})"
    )

    table = Table(title="Chunk Type Distribution")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for chunk_type, count in result["chunk_types"].items():
        table.add_row(chunk_type, str(count))
    console.print(table)

    validation = result.get("validation")
    if validation:
        console.print(
            f"validation: errors={validation['errors']} "
            f"warnings={validation['warnings']} infos={validation['infos']} "
            f"-> {validation['report']}"
        )
        if validation["errors"]:
            console.print(
                "[red]Validation found errors - fix before using this run.[/red]"
            )
            raise typer.Exit(code=1)

    console.print(f"[bold]Run {run_id} ready:[/bold] {result['run_dir']}")
