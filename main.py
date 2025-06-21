from __future__ import annotations

"""Entry-point script for the slimmed-down OCR → extraction pipeline.

Usage (unchanged):

    uv run main.py

It performs three high-level steps:

1. Discover images in ``input_images/``
2. Transcribe each page via Gemini (cached per-page JSON)
3. Extract & deduplicate petition entries
4. Write artefacts to ``output_text/``
"""

import json
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

import extraction
import ocr
from gemini_client import GeminiClient
from models import Page, Entry
from utils import discover_images

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_WORKERS = 10
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output_text")
TRANSCRIPTION_DIR = OUTPUT_DIR / "page_transcriptions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_artifacts(pages: List[Page], entries: List[Entry], out_dir: Path) -> None:
    """Save JSON / TXT artefacts for downstream analysis."""

    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Per-page JSON list
    with (out_dir / "pages.json").open("w") as fh:
        json.dump([p.model_dump() for p in pages], fh, indent=2)

    # 2. Extracted entries
    with (out_dir / "extracted_entries.json").open("w") as fh:
        json.dump([e.model_dump() for e in entries], fh, indent=2)

    # 3. Full transcription text (for manual inspection)
    full_text = "\n\n".join(p.text for p in pages)
    (out_dir / "complete_transcription.txt").write_text(full_text)

    # 4. Simple metadata summary
    flagged = [e for e in entries if e.confidence_score < 0.7]
    metadata = {
        "total_pages": len(pages),
        "total_entries": len(entries),
        "flagged_low_confidence": len(flagged),
    }
    with (out_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple script entry-point
    """Run the complete OCR + extraction pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    load_dotenv()

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        client = GeminiClient()
    except ValueError as exc:
        logging.error("%s", exc)
        return

    # ------------------------------------------------------------------
    # 1. Discover images & transcribe
    # ------------------------------------------------------------------

    image_paths = discover_images(INPUT_DIR)
    if not image_paths:
        logging.warning("No image files found in %s – nothing to do.", INPUT_DIR)
        return

    pages = ocr.transcribe_pages(
        image_paths=image_paths,
        client=client,
        transcription_dir=TRANSCRIPTION_DIR,
        max_workers=MAX_WORKERS,
    )

    # ------------------------------------------------------------------
    # 2. Extract + deduplicate entries
    # ------------------------------------------------------------------

    entries = extraction.extract_entries(pages, client)

    # ------------------------------------------------------------------
    # 3. Persist results
    # ------------------------------------------------------------------

    _write_artifacts(pages, entries, OUTPUT_DIR)
    logging.info(
        "Pipeline complete – %d page(s) transcribed, %d unique entries extracted.",
        len(pages),
        len(entries),
    )


if __name__ == "__main__":
    main()