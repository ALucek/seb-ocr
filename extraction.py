from __future__ import annotations

"""Entity–extraction helper – slimmer replacement for TwoPhaseProcessor phase-2 & phase-3 logic."""

from typing import List, Optional
import logging

from pathlib import Path

from pydantic import BaseModel, Field

import prompts
from gemini_client import GeminiClient
from models import Page, Entry
from utils import hash_text

logger = logging.getLogger(__name__)

__all__ = [
    "extract_entries",
    "deduplicate_entries",
]

# ---------------------------------------------------------------------------
# Constants – keep same defaults previously used in main.py
# ---------------------------------------------------------------------------

CHUNK_OVERLAP_CHARS = 2_000  # Character overlap between chunks
MAX_CHUNK_SIZE = 40_000      # Target chunk size in characters (well below Gemini limit)


# ---------------------------------------------------------------------------
# Internal pydantic schema for structured Gemini response
# ---------------------------------------------------------------------------

class _EntrySchema(BaseModel):
    """Matches the JSON format we expect Gemini to emit per petition entry."""

    individual: str
    title_or_position: Optional[str] = None
    location: Optional[str] = None
    full_identifier: str
    text: str
    confidence_score: float


_ResponseSchema = list[_EntrySchema]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_entries(pages: List[Page], client: GeminiClient) -> List[Entry]:
    """Run entity extraction over *pages* and return a list[Entry]."""

    if not pages:
        logger.warning("No pages provided to extract_entries(); returning empty list.")
        return []

    # Combine transcription text with page markers to aid chunking
    full_text_parts: List[str] = []
    for p in pages:
        full_text_parts.append(f"=== PAGE {p.number} ({p.filename}) ===\n")
        full_text_parts.append(p.text)
        full_text_parts.append("\n\n")
    full_text = "".join(full_text_parts)

    chunks = _create_overlapping_chunks(full_text)
    logger.info("Created %d overlapping chunk(s) for extraction.", len(chunks))

    all_entries: List[Entry] = []
    for idx, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d (length: %d chars)…", idx + 1, len(chunks), len(chunk))
        extracted = _extract_from_chunk(chunk, idx, client)
        all_entries.extend(extracted)
        logger.info("→ %d entries extracted", len(extracted))

    logger.info("Extraction finished – %d raw entries total.", len(all_entries))
    return all_entries


def deduplicate_entries(entries: List[Entry]) -> List[Entry]:
    """Remove duplicate *entries* using the source_hash field (high-confidence first)."""

    seen: set[str] = set()
    unique: List[Entry] = []
    # Sort by confidence descending so we keep the best duplicate
    for entry in sorted(entries, key=lambda e: e.confidence_score, reverse=True):
        if entry.source_hash not in seen:
            seen.add(entry.source_hash)
            unique.append(entry)
    return unique

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_overlapping_chunks(text: str) -> List[str]:
    """Slice *text* into overlapping segments suitable for the Gemini context window."""

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Attempt a graceful break at a page marker or paragraph
        break_point = end
        page_marker = text.rfind("=== PAGE", start, end)
        if page_marker > start + MAX_CHUNK_SIZE // 2:
            break_point = page_marker
        else:
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + MAX_CHUNK_SIZE // 2:
                break_point = para_break

        chunks.append(text[start:break_point])
        start = max(start + 1, break_point - CHUNK_OVERLAP_CHARS)

    return chunks


def _extract_from_chunk(chunk: str, chunk_id: int, client: GeminiClient) -> List[Entry]:
    """Call Gemini to extract entries from a single chunk."""

    prompt = prompts.ENTITY_EXTRACTION_PROMPT.format(text=chunk)
    try:
        response = client.generate_text_only(
            prompt=prompt,
            response_schema=_ResponseSchema,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Gemini extraction failed for chunk %d: %r", chunk_id, exc)
        return []

    if not response:
        logger.warning("Empty response for chunk %d", chunk_id)
        return []

    entries: List[Entry] = []
    for obj in response:
        try:
            entry = Entry(
                individual=obj.individual,
                title_or_position=obj.title_or_position,
                location=obj.location,
                full_identifier=obj.full_identifier,
                text=obj.text,
                confidence_score=obj.confidence_score,
                source_hash=hash_text(obj.text),
            )
            entries.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to parse entry in chunk %d: %r", chunk_id, exc)
    return entries 