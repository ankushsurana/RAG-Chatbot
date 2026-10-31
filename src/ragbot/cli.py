"""Command-line interface: corpus download, ingestion, questions, serving and evaluation."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragbot.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="RAG chatbot over PDF corpora.")
console = Console()


@app.command("download-corpus")
def download_corpus_command(manifest: Path = Path("corpus/manifest.json")) -> None:
    """Download the public demo corpus and build its scanned document."""
    from ragbot.ingestion.corpus import download_corpus

    items = download_corpus(get_settings(), manifest, log=console.print)
    table = Table("File", "Pages", "MB", "Image-only")
    for item in items:
        table.add_row(item.filename, str(item.pages), f"{item.size_mb:.1f}", str(item.scanned))
    console.print(table)
    console.print(f"{len(items)} PDFs, {sum(item.pages for item in items):,} pages")


@app.command()
def ingest(directory: Path | None = None, force: bool = False) -> None:
    """Ingest every PDF in a directory. Unchanged documents are skipped."""
    from ragbot.ingestion.pipeline import IngestProgress
    from ragbot.service import build_service

    _, pipeline = build_service(get_settings())

    def on_progress(event: IngestProgress) -> None:
        if event.stage != "embed" or event.completed == 0 or event.completed % 800 == 0:
            amount = f" {event.completed}/{event.total}" if event.total else ""
            console.print(f"[{event.stage:>8}] {event.filename}{amount} {event.detail}")

    summary = pipeline.ingest_directory(directory, force, on_progress)
    console.print(
        f"ingested={len(summary.ingested)} skipped={len(summary.skipped)} "
        f"failed={len(summary.failed)}"
    )
    for filename, error in summary.failed.items():
        console.print(f"[red]FAILED[/red] {filename}: {error}")
    if summary.failed:
        raise typer.Exit(code=1)


@app.command()
def documents() -> None:
    """List indexed documents and their ingestion statistics."""
    from ragbot.storage.catalog import Catalog

    catalog = Catalog(get_settings().catalog_path)
    table = Table("File", "Status", "Pages", "OCR pages", "Image OCR", "Skipped", "Chunks", "Lang")
    records = catalog.list_documents()
    for record in records:
        table.add_row(
            record.filename,
            record.status,
            str(record.page_count),
            str(record.ocr_pages),
            str(record.image_ocr_blocks),
            str(record.skipped_pages),
            str(record.chunk_count),
            record.language,
        )
    console.print(table)
    for record in records:
        if record.error:
            console.print(f"{record.filename} failed: {record.error}", markup=False)
    pages = sum(record.page_count for record in records)
    console.print(f"{len(records)} documents, {pages:,} pages, {catalog.chunk_count():,} chunks")


@app.command()
def ask(question: str, show_chunks: bool = False) -> None:
    """Ask one question from the terminal."""
    from ragbot.service import build_service

    service, _ = build_service(get_settings())
    result = service.answer(question)
    console.print(result.answer, markup=False)
    for citation in result.citations:
        console.print(
            f"  [{citation.number}] {citation.filename}, p. {citation.page}", markup=False
        )
    console.print(
        f"status={result.verification.status} total={result.timings.total_ms:.0f}ms "
        f"retrieval={result.timings.retrieval_ms:.0f}ms "
        f"first_token={result.timings.first_token_ms}ms",
        markup=False,
    )
    if show_chunks:
        console.print_json(
            json.dumps([c.model_dump(exclude={"text", "spans"}) for c in result.retrieved])
        )


@app.command()
def serve(host: str | None = None, port: int | None = None) -> None:
    """Start the web UI and API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ragbot.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


@app.command()
def share(port: int | None = None) -> None:
    """Start the chatbot and a free public link to it (the link lasts until you stop this)."""
    import uvicorn

    settings = get_settings()
    port = port or settings.api_port
    tunnel_tool = shutil.which("cloudflared") or settings.data_dir / "tools" / "cloudflared.exe"
    if not Path(tunnel_tool).exists():
        console.print("cloudflared was not found. Put it in data/tools or on your PATH.")
        raise typer.Exit(code=1)

    tunnel = subprocess.Popen(
        [str(tunnel_tool), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stderr=subprocess.PIPE,
        text=True,
    )

    def announce_link() -> None:
        for line in tunnel.stderr:
            match = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if match:
                console.print(
                    f"\n[bold green]Public link:[/bold green] {match.group(0)}\n", soft_wrap=True
                )

    threading.Thread(target=announce_link, daemon=True).start()
    try:
        uvicorn.run("ragbot.api.app:create_app", factory=True, host=settings.api_host, port=port)
    finally:
        tunnel.terminate()


@app.command()
def evaluate(
    golden: Path = Path("eval/golden.jsonl"),
    retrieval_only: bool = False,
    limit: int | None = None,
) -> None:
    """Measure relevance, latency, citation accuracy and hallucination rate on the golden set."""
    from ragbot.evaluation.runner import run_evaluation

    settings = get_settings()
    report = run_evaluation(settings, golden, retrieval_only=retrieval_only, limit=limit)
    console.print_json(json.dumps(report["summary"]))


@app.command("evaluate-ocr")
def evaluate_ocr(
    pages: int = 30,
    scanned: str = "Think_Python_2e_SCANNED.pdf",
) -> None:
    """Measure OCR accuracy by comparing OCR output against the original document's text layer."""
    from ragbot.evaluation.ocr_eval import run_ocr_evaluation

    console.print_json(json.dumps(run_ocr_evaluation(get_settings(), scanned, pages)))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app()


if __name__ == "__main__":
    main()
