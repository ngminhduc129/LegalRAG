"""Parse CLI - Extract and parse legal documents."""

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="parse", help="Parse PDF into structured legal corpus")
console = Console()


@app.command()
def pdf(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to PDF file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output-dir",
        "-o",
        help="Output directory for parsed corpus",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Parse PDF into structured legal corpus (JSONL + CSV)."""
    from legal_rag.corpus.extractor import parse_corpus, save_corpus

    console.print(f"[bold]Parsing PDF:[/bold] {pdf_path}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting PDF...", total=None)
        chunks = parse_corpus(pdf_path)
        progress.update(task, description=f"Parsed {len(chunks)} chunks")

        task2 = progress.add_task("Saving corpus...", total=None)
        jsonl_path, csv_path = save_corpus(chunks, output_dir, document_id)
        progress.update(task2, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(chunks)}")


@app.command()
def lines(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to PDF file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        Path("data/processed/lines.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file for extracted lines",
    ),
):
    """Extract raw lines from PDF (for debugging)."""
    from legal_rag.corpus.extractor import extract_pdf_lines
    import json

    console.print(f"[bold]Extracting lines from:[/bold] {pdf_path}")

    lines = extract_pdf_lines(pdf_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    console.print(f"[green]✓[/green] Extracted {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    app()