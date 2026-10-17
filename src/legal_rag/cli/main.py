"""Main CLI entry point."""

import typer
from rich.console import Console

from legal_rag.cli.parse import app as parse_app
from legal_rag.cli.chunk import app as chunk_app
from legal_rag.cli.index import app as index_app
from legal_rag.cli.retrieve import app as retrieve_app
from legal_rag.cli.generate import app as generate_app
from legal_rag.cli.evaluate import app as evaluate_app
from legal_rag.cli.pipeline import pipeline as pipeline_command

app = typer.Typer(
    name="legal-rag",
    help="Vietnamese Labor Law Legal RAG System",
    no_args_is_help=True,
)

app.add_typer(parse_app, name="parse")
app.add_typer(chunk_app, name="chunk")
app.add_typer(index_app, name="index")
app.add_typer(retrieve_app, name="retrieve")
app.add_typer(generate_app, name="generate")
app.add_typer(evaluate_app, name="evaluate")
app.command(
    "pipeline",
    help="Standard pipeline: PDF -> page patches -> corpus -> legal-aware chunks -> validation",
)(pipeline_command)

console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Legal RAG - Vietnamese Labor Law Legal RAG System"""
    if verbose:
        console.print("[bold green]Legal RAG CLI[/bold green] - Verbose mode enabled")


if __name__ == "__main__":
    app()