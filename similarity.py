from __future__ import annotations

import logging
import math
from typing import List

from gemini_client import GeminiClient
from schemas import DetectedEntity

logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v * v for v in vec1))
    magnitude2 = math.sqrt(sum(v * v for v in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def deduplicate_entities_by_similarity(
    entities: list[DetectedEntity], client: GeminiClient, similarity_threshold: float = 0.98
) -> list[DetectedEntity]:
    """
    Deduplicate a list of entities based on semantic similarity of their text.

    This function generates embeddings for each entity's text, calculates the
    cosine similarity between them, and groups entities that are sufficiently
    similar. From each group, it selects the entity that appears on the latest
    page as the canonical representation.
    """
    if len(entities) < 2:
        return entities

    logger.info(
        "Starting semantic deduplication for %d entities with threshold %.2f.",
        len(entities),
        similarity_threshold,
    )

    # Generate embeddings for all entities in a single batch call.
    entity_texts = [entity.text for entity in entities]
    embeddings = client.embed_contents(entity_texts)

    # Group entities by similarity.
    groups: list[list[int]] = []
    visited = [False] * len(entities)

    for i in range(len(entities)):
        if visited[i]:
            continue

        # Start a new group with the current entity.
        current_group = [i]
        visited[i] = True

        # Check for similar entities to add to the group.
        for j in range(i + 1, len(entities)):
            if not visited[j]:
                similarity = _cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= similarity_threshold:
                    current_group.append(j)
                    visited[j] = True
        groups.append(current_group)

    # From each group, select the best entity (the one from the latest page).
    final_entities: list[DetectedEntity] = []
    for group in groups:
        if not group:
            continue

        if len(group) == 1:
            final_entities.append(entities[group[0]])
            continue

        # Find the entity in the group with the highest (latest) page number.
        best_entity_in_group = max(
            (entities[i] for i in group), key=lambda e: max(e.page_numbers)
        )
        final_entities.append(best_entity_in_group)
        logger.info(
            "Consolidated %d similar entities into one based on latest page.", len(group)
        )

    logger.info(
        "Semantic deduplication reduced %d entities to %d.",
        len(entities),
        len(final_entities),
    )

    return final_entities 