from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EntityFromLLM(BaseModel):
    """Represents the direct output from the LLM for a single entity."""

    individual: str = Field(
        ..., description="Person's name only (e.g., 'Babu Subh Narain Choudhry')"
    )
    title_or_position: Optional[str] = Field(
        None, description="Professional title or role (e.g., 'Manager', 'Barrister')"
    )
    location: Optional[str] = Field(
        None,
        description="Geographic location ONLY (e.g., 'Raj Darbhanga, Kajha, P. O., Purnea')",
    )
    full_identifier: str = Field(
        ...,
        description="The complete original text as written for the person (e.g., 'Manager, Raj Darbhanga, Kajha, P. O., Purnea')",
    )
    text: str = Field(..., description="The person's complete statement/opinion")


class LLMResponse(BaseModel):
    """A container for the list of entities returned by the LLM."""

    entities: List[EntityFromLLM]


class DetectedEntity(EntityFromLLM):
    """Represents a single extracted entity, enriched with application metadata."""

    page_numbers: List[int] = Field(
        default_factory=list,
        description="List of page numbers the source text for this entity came from.",
    ) 