from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

__all__ = ["Page", "Entry"]


class Page(BaseModel):
    """Represents a single transcribed page."""

    number: int = Field(description="Sequential page number extracted from the filename.")
    filename: str = Field(description="Original image filename (e.g. 'page_12.jpg').")
    text: str = Field(description="Complete transcribed text for the page.")
    confidence: str = Field(description="OCR confidence label: 'high', 'medium', or 'low'.")
    issues: List[str] = Field(default_factory=list, description="OCR challenges or unreadable fragments noted during transcription.")


class Entry(BaseModel):
    """Represents a single structured petition entry extracted from the transcription."""

    individual: str = Field(description="Name of the individual, including honorifics if present.")
    title_or_position: Optional[str] = Field(None, description="Professional title or role (e.g., 'Manager', 'Barrister').")
    location: Optional[str] = Field(None, description="Geographic location associated with the entry.")
    full_identifier: str = Field(description="Complete identifying string as it appears in the source text.")
    text: str = Field(description="Full original text of the individual's statement or petition.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Extraction confidence score between 0.0 and 1.0.")
    source_hash: str = Field(description="MD5 hash of the entry text, used for deduplication.") 