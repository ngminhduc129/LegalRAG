"""Corpus CLI - audit PDFs and batch-build chunks for the whole registry."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(name="corpus", help="Audit and batch-build the legal corpus")
console = Console()


@app.command("audit")
def audit(
    output: Path = typer.Option(
        Path("data/processed/corpus_audit.json"),
        "--output",
        "-o",
        help="Audit report path (read-only inspection of data/raw)",
    ),
    min_chars: int = typer.Option(
        20,
        "--min-chars",
        help="A page needs at least this many characters to count as having text",
    ),
    document_id: Optional[List[str]] = typer.Option(
        None,
        "--document-id",
        "-d",
        help="Limit to specific registry ids (repeatable)",
    ),
):
    """Report text-layer coverage per PDF/page (never modifies data/raw)."""
    from legal_rag.corpus.audit import audit_registry, write_audit_report

    audits = audit_registry(min_chars=min_chars, document_ids=document_id)
    path = write_audit_report(audits, output)

    table = Table(title="Corpus audit")
    table.add_column("document_id")
    table.add_column("status")
    table.add_column("pages", justify="right")
    table.add_column("text", justify="right")
    table.add_column("need OCR", justify="right")
    table.add_column("U+FFFD", justify="right")
    for item in audits:
        table.add_row(
            item.document_id,
            item.status,
            str(item.pages),
            str(item.pages_with_text),
            str(len(item.pages_needing_ocr)),
            str(item.corruption_hits) if item.corruption_hits else "-",
        )
    console.print(table)
    console.print(
        f"[green]\u2713[/green] wrote {path} "
        f"({sum(a.pages for a in audits)} pages, "
        f"{sum(len(a.pages_needing_ocr) for a in audits)} need OCR)"
    )


@app.command("build")
def build(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="Unique run id; existing runs are never overwritten",
    ),
    strategy: str = typer.Option(
        "legal_aware",
        "--strategy",
        help="legal_aware | fixed_300 | fixed_500 | fixed_800",
    ),
    chunks_dir: Path = typer.Option(
        Path("data/chunks"), "--chunks-dir", help="Base output directory"
    ),
    document_id: Optional[List[str]] = typer.Option(
        None,
        "--document-id",
        "-d",
        help="Limit to specific registry ids (repeatable)",
    ),
    ocr_engine: str = typer.Option(
        "auto", "--ocr-engine", help="auto | tesseract | paddleocr"
    ),
    ocr_cache: Path = typer.Option(
        Path("data/ocr"), "--ocr-cache", help="OCR page cache directory"
    ),
    dpi: int = typer.Option(300, "--dpi", help="Render DPI for OCR"),
    force_ocr: bool = typer.Option(
        False, "--force-ocr", help="Ignore the OCR cache"
    ),
    ocr_workers: int = typer.Option(
        1,
        "--ocr-workers",
        help="Parallel OCR page workers (Tesseract subprocesses; try 4 on a laptop)",
    ),
    patches: Optional[List[Path]] = typer.Option(
        None, "--patches", "-p", help="Page patch files or directories"
    ),
):
    """Build chunks for every registry document into a fresh run directory."""
    from legal_rag.corpus.build import run_corpus_build

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Building {strategy}...", total=None)
            result = run_corpus_build(
                run_id=run_id,
                strategy=strategy,
                document_ids=document_id,
                chunks_dir=chunks_dir,
                patch_paths=patches,
                ocr_engine=ocr_engine,
                ocr_cache=ocr_cache,
                dpi=dpi,
                force_ocr=force_ocr,
                ocr_workers=ocr_workers,
            )
            progress.update(task, description="Done")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title=f"Corpus build {run_id} ({strategy})")
    table.add_column("document_id")
    table.add_column("status")
    table.add_column("pages", justify="right")
    table.add_column("OCR", justify="right")
    table.add_column("chunks", justify="right")
    for item in result["documents"]:
        table.add_row(
            item["document_id"],
            item["status"],
            str(item["pages"]),
            str(item["pages_ocr_required"]),
            str(item["final_chunks"]),
        )
    console.print(table)
    console.print(f"status: {result['status_counts']}")
    console.print(f"total chunks: {result['total_chunks']}")
    console.print(f"[green]\u2713[/green] chunks: {result['all_chunks_path']}")
    console.print(f"[green]\u2713[/green] status: {result['status_path']}")
    console.print(f"[green]\u2713[/green] manifest: {result['manifest_path']}")

    blocked = [
        item
        for item in result["documents"]
        if item["status"]
        in {
            "ocr_required",
            "parse_failed",
            "empty",
            "missing_pdf",
            "source_corrupt",
        }
    ]
    if blocked:
        console.print(
            f"[yellow]{len(blocked)} document(s) need attention:[/yellow] "
            + ", ".join(f"{item['document_id']}({item['status']})" for item in blocked)
        )
        corrupt = [item for item in blocked if item["status"] == "source_corrupt"]
        if corrupt:
            console.print(
                "[red]Corrupt PDFs must be re-downloaded into data/raw/ "
                "(the build never edits source files).[/red]"
            )


if __name__ == "__main__":
    app()
