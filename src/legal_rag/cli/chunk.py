"""Chunk CLI - Run chunking strategies."""

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(name="chunk", help="Run chunking strategies on parsed corpus")
console = Console()


@app.command()
def legal_aware(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/chunks/legal_aware"),
        "--output-dir",
        "-o",
        help="Output directory for chunks",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Run legal-aware chunking (preserves hierarchy)."""
    from legal_rag.chunking.legal_aware import LegalAwareChunker, save_jsonl, save_csv
    import json

    console.print(f"[bold]Legal-aware chunking:[/bold] {input_path}")

    # Load corpus
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        task2 = progress.add_task("Chunking...", total=None)
        chunker = LegalAwareChunker(document_id=document_id)
        result_chunks = chunker.chunk(chunks)
        progress.update(task2, description=f"Created {len(result_chunks)} chunks")

        task3 = progress.add_task("Saving...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{document_id}_legal_aware.jsonl"
        csv_path = output_dir / f"{document_id}_legal_aware.csv"
        save_jsonl(result_chunks, jsonl_path)
        save_csv(result_chunks, csv_path)
        progress.update(task3, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(result_chunks)}")

    # Show chunk type distribution
    from collections import Counter
    type_counts = Counter(c["chunk_type"] for c in result_chunks)
    table = Table(title="Chunk Type Distribution")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for ctype, count in type_counts.most_common():
        table.add_row(ctype, str(count))
    console.print(table)


@app.command()
def fixed_size(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/chunks/fixed_500"),
        "--output-dir",
        "-o",
        help="Output directory for chunks",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Chunk size in tokens/words"),
    overlap: int = typer.Option(100, "--overlap", help="Overlap between chunks"),
):
    """Run fixed-size chunking (baseline)."""
    from legal_rag.chunking.legal_aware import FixedSizeChunker, save_jsonl, save_csv
    import json

    console.print(f"[bold]Fixed-size chunking:[/bold] {input_path} (size={chunk_size}, overlap={overlap})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        task2 = progress.add_task("Chunking...", total=None)
        chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            document_id=document_id,
        )
        result_chunks = chunker.chunk(chunks)
        progress.update(task2, description=f"Created {len(result_chunks)} chunks")

        task3 = progress.add_task("Saving...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{document_id}_fixed_{chunk_size}.jsonl"
        csv_path = output_dir / f"{document_id}_fixed_{chunk_size}.csv"
        save_jsonl(result_chunks, jsonl_path)
        save_csv(result_chunks, csv_path)
        progress.update(task3, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(result_chunks)}")


@app.command()
def paragraph(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/chunks/paragraph"),
        "--output-dir",
        "-o",
        help="Output directory for chunks",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Run paragraph-based chunking (baseline)."""
    from legal_rag.chunking.legal_aware import ParagraphChunker, save_jsonl, save_csv
    import json

    console.print(f"[bold]Paragraph chunking:[/bold] {input_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        task2 = progress.add_task("Chunking...", total=None)
        chunker = ParagraphChunker(document_id=document_id)
        result_chunks = chunker.chunk(chunks)
        progress.update(task2, description=f"Created {len(result_chunks)} chunks")

        task3 = progress.add_task("Saving...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{document_id}_paragraph.jsonl"
        csv_path = output_dir / f"{document_id}_paragraph.csv"
        save_jsonl(result_chunks, jsonl_path)
        save_csv(result_chunks, csv_path)
        progress.update(task3, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(result_chunks)}")


@app.command()
def article(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/chunks/article"),
        "--output-dir",
        "-o",
        help="Output directory for chunks",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Run article-level chunking (baseline)."""
    from legal_rag.chunking.legal_aware import ArticleChunker, save_jsonl, save_csv
    import json

    console.print(f"[bold]Article chunking:[/bold] {input_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        task2 = progress.add_task("Chunking...", total=None)
        chunker = ArticleChunker(document_id=document_id)
        result_chunks = chunker.chunk(chunks)
        progress.update(task2, description=f"Created {len(result_chunks)} chunks")

        task3 = progress.add_task("Saving...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{document_id}_article.jsonl"
        csv_path = output_dir / f"{document_id}_article.csv"
        save_jsonl(result_chunks, jsonl_path)
        save_csv(result_chunks, csv_path)
        progress.update(task3, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(result_chunks)}")


@app.command()
def clause(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("data/chunks/clause"),
        "--output-dir",
        "-o",
        help="Output directory for chunks",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Run clause-level chunking (baseline)."""
    from legal_rag.chunking.legal_aware import ClauseChunker, save_jsonl, save_csv
    import json

    console.print(f"[bold]Clause chunking:[/bold] {input_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        task2 = progress.add_task("Chunking...", total=None)
        chunker = ClauseChunker(document_id=document_id)
        result_chunks = chunker.chunk(chunks)
        progress.update(task2, description=f"Created {len(result_chunks)} chunks")

        task3 = progress.add_task("Saving...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{document_id}_clause.jsonl"
        csv_path = output_dir / f"{document_id}_clause.csv"
        save_jsonl(result_chunks, jsonl_path)
        save_csv(result_chunks, csv_path)
        progress.update(task3, description="Saved")

    console.print(f"[green]✓[/green] JSONL: {jsonl_path}")
    console.print(f"[green]✓[/green] CSV:   {csv_path}")
    console.print(f"[bold]Total chunks:[/bold] {len(result_chunks)}")


@app.command()
def all_baselines(
    input_path: Path = typer.Argument(
        ...,
        help="Path to parsed corpus JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_base: Path = typer.Option(
        Path("data/chunks"),
        "--output-base",
        "-o",
        help="Base output directory",
    ),
    document_id: str = typer.Option(
        "labor_code_2019",
        "--document-id",
        "-d",
        help="Document identifier",
    ),
):
    """Run all baseline chunking strategies."""
    import json
    from legal_rag.chunking.legal_aware import (
        FixedSizeChunker, ParagraphChunker,
        ArticleChunker, ClauseChunker,
        save_jsonl, save_csv,
    )

    console.print(f"[bold]Running all baselines on:[/bold] {input_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading corpus...", total=None)
        chunks = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        progress.update(task, description=f"Loaded {len(chunks)} items")

        strategies = [
            ("fixed_300", FixedSizeChunker(chunk_size=300, overlap=60, document_id=document_id), "data/chunks/fixed_300"),
            ("fixed_500", FixedSizeChunker(chunk_size=500, overlap=100, document_id=document_id), "data/chunks/fixed_500"),
            ("fixed_800", FixedSizeChunker(chunk_size=800, overlap=160, document_id=document_id), "data/chunks/fixed_800"),
            ("paragraph", ParagraphChunker(document_id=document_id), "data/chunks/paragraph"),
            ("article", ArticleChunker(document_id=document_id), "data/chunks/article"),
            ("clause", ClauseChunker(document_id=document_id), "data/chunks/clause"),
        ]

        for name, chunker, out_dir in strategies:
            task2 = progress.add_task(f"Chunking {name}...", total=None)
            result_chunks = chunker.chunk(chunks)
            progress.update(task2, description=f"Created {len(result_chunks)} chunks for {name}")

            task3 = progress.add_task(f"Saving {name}...", total=None)
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            jsonl_path = out_path / f"{document_id}_{name}.jsonl"
            csv_path = out_path / f"{document_id}_{name}.csv"
            save_jsonl(result_chunks, jsonl_path)
            save_csv(result_chunks, csv_path)
            progress.update(task3, description=f"Saved {name}")

    console.print("[green]✓[/green] All baselines completed")


@app.command()
def clean(
    input_path: Path = typer.Argument(
        ...,
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (default: overwrite input)",
    ),
    review_path: Path = typer.Option(
        None,
        "--review",
        "-r",
        help="Path for chunks needing review",
    ),
):
    """Clean chunks (remove page numbers, footnotes, etc.)."""
    from legal_rag.chunking.cleaner import load_chunks, save_chunks, clean_chunk

    console.print(f"[bold]Cleaning chunks:[/bold] {input_path}")

    chunks = load_chunks(input_path)
    console.print(f"Loaded {len(chunks)} chunks")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Cleaning...", total=len(chunks))
        cleaned_chunks = []
        total_cleaned = 0
        total_review = 0

        for chunk in chunks:
            cleaned = clean_chunk(chunk)
            cleaned_chunks.append(cleaned)
            if cleaned["cleaned"]:
                total_cleaned += 1
            if cleaned["needs_review"]:
                total_review += 1
            progress.advance(task)

    out_path = output_path or input_path
    save_chunks(cleaned_chunks, out_path)
    console.print(f"[green]✓[/green] Saved cleaned chunks to {out_path}")

    if review_path:
        review_chunks = [c for c in cleaned_chunks if c["needs_review"]]
        if review_chunks:
            save_chunks(review_chunks, review_path)
            console.print(f"[yellow]⚠[/yellow] {len(review_chunks)} chunks need review -> {review_path}")

    console.print(f"[bold]Cleaned:[/bold] {total_cleaned} | [bold]Need review:[/bold] {total_review}")


@app.command()
def validate(
    input_path: Path = typer.Argument(
        ...,
        help="Path to chunks JSONL",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for validation report",
    ),
):
    """Validate chunks (check structure, metadata, completeness)."""
    from legal_rag.chunking.validator import validate_chunks
    import json

    console.print(f"[bold]Validating chunks:[/bold] {input_path}")

    chunks = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    console.print(f"Loaded {len(chunks)} chunks")

    report = validate_chunks(chunks)

    console.print(report.summary())

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(report.summary())
        console.print(f"[green]✓[/green] Report saved to {output_path}")


if __name__ == "__main__":
    app()