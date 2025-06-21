from __future__ import annotations

"""OCR helper functions – simpler replacement for TwoPhaseProcessor phase-1 logic.

The public surface of this module is a single function:

    transcribe_pages(image_paths, client, transcription_dir, max_workers=10) -> list[Page]

It handles caching (reusing existing JSON files), parallel Gemini calls and
returns a list[Page] suitable for downstream processing.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

import prompts
from gemini_client import GeminiClient
from models import Page
from utils import extract_number_from_filename

logger = logging.getLogger(__name__)

__all__ = ["transcribe_pages"]


class _TranscriptionSchema(BaseModel):
    """Internal schema for Gemini structured transcription response."""

    transcribed_text: str = Field(description="Full page transcription as a single text block.")
    confidence: str = Field(description="Confidence label: high | medium | low")
    issues: List[str] = Field(default_factory=list, description="OCR challenges or unreadable fragments.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_pages(
    image_paths: List[Path],
    client: GeminiClient,
    transcription_dir: Path,
    max_workers: int = 10,
) -> List[Page]:
    """Transcribe *image_paths* in parallel and cache JSON results.

    Parameters
    ----------
    image_paths
        List of image files to transcribe. They should be pre-sorted in natural
        order (e.g., using ``utils.discover_images``).
    client
        :class:`GeminiClient` instance used for the API calls.
    transcription_dir
        Folder where per-page JSON files are stored. The directory will be
        created if it does not exist.
    max_workers
        Maximum number of threads for concurrent transcription.
    """

    transcription_dir.mkdir(exist_ok=True, parents=True)

    pages: List[Page] = []

    # Identify which images still need processing
    images_to_process: List[Path] = []
    for path in image_paths:
        if not (transcription_dir / f"{path.stem}.json").exists():
            images_to_process.append(path)

    if images_to_process:
        logger.info("Transcribing %d new image(s)…", len(images_to_process))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_transcribe_single_page, path, client): path
                for path in images_to_process
            }

            for i, future in enumerate(as_completed(future_map), 1):
                path = future_map[future]
                try:
                    page = future.result()
                except Exception as exc:  # pragma: no cover
                    logger.error("Error transcribing %s: %r", path.name, exc)
                    continue

                if page is None:
                    continue

                _save_page_json(page, transcription_dir)
                pages.append(page)
                logger.info("[%d/%d] Transcribed %s", i, len(images_to_process), path.name)

    # Load existing pages (including those just created above)
    for path in image_paths:
        json_file = transcription_dir / f"{path.stem}.json"
        page = _load_page_json(json_file)
        if page:
            pages.append(page)

    # Deduplicate & sort (in case of overlaps)
    seen_files: set[str] = set()
    unique_pages: List[Page] = []
    for page in pages:
        if page.filename not in seen_files:
            seen_files.add(page.filename)
            unique_pages.append(page)

    unique_pages.sort(key=lambda p: p.number)
    logger.info("Transcription phase complete – %d pages total.", len(unique_pages))
    return unique_pages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _transcribe_single_page(image_path: Path, client: GeminiClient) -> Optional[Page]:
    """Run Gemini OCR on a single *image_path* and return a :class:`Page`."""

    page_number = extract_number_from_filename(image_path)
    prompt = prompts.TRANSCRIPTION_ONLY_PROMPT

    try:
        response = client.generate(
            prompt=prompt,
            image_path=image_path,
            response_schema=_TranscriptionSchema,
        )

        return Page(
            number=page_number,
            filename=image_path.name,
            text=response.transcribed_text,
            confidence=response.confidence,
            issues=response.issues,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Gemini transcription failed for %s: %r", image_path.name, exc)
        return None


def _save_page_json(page: Page, out_dir: Path) -> None:
    """Persist *page* as JSON inside *out_dir*."""

    json_path = out_dir / f"{Path(page.filename).stem}.json"
    try:
        with json_path.open("w") as fh:
            json.dump(page.model_dump(), fh, indent=2)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to write transcription JSON %s: %r", json_path, exc)


def _load_page_json(json_path: Path) -> Optional[Page]:
    """Load a :class:`Page` from *json_path*. Returns *None* on failure."""

    if not json_path.exists():
        return None

    try:
        with json_path.open() as fh:
            data = json.load(fh)
        return Page(**data)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read transcription JSON %s: %r", json_path, exc)
        return None 