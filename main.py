from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Type, Union

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gemini_client import GeminiClient
import prompts
from similarity import deduplicate_entities_by_similarity

from pydantic import BaseModel
from schemas import DetectedEntity, EntityFromLLM, LLMResponse

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

# The maximum number of concurrent workers for processing images.
MAX_WORKERS = 10

# Configuration for the sliding window entity extraction.
WINDOW_SIZE = 5  # Number of pages to include in each window.
WINDOW_STEP = 2  # Number of pages to slide the window forward.


class ExtractedEntities(BaseModel):
    """A container for the final list of extracted entities."""

    entities: List[DetectedEntity]


def _log_retry_attempt(retry_state):
    """Log information before we sleep between retry attempts."""
    # The first argument to process_image is image_path
    image_path = retry_state.args[0]
    logging.warning(
        "Retrying processing for %s due to error: %s. Attempt #%d, waiting %.2fs…",
        image_path.name,
        retry_state.outcome.exception(),
        retry_state.attempt_number,
        retry_state.next_action.sleep,
    )


def _extract_number_from_filename(path: Path) -> int:
    """Return an integer found in *path* to allow natural sorting."""
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _discover_images(directory: Path) -> List[Path]:
    """Return a sorted list of image files inside *directory*."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))

    paths.sort(key=_extract_number_from_filename)
    return paths


def _discover_transcriptions(directory: Path) -> List[Path]:
    """Return a sorted list of transcription files inside *directory*."""
    paths = sorted(directory.glob("*.txt"), key=_extract_number_from_filename)
    return paths


@retry(
    retry=retry_if_not_exception_type(FileNotFoundError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    before_sleep=_log_retry_attempt,
)
def process_image(
    image_path: Path,
    client: GeminiClient,
    prompt: str,
) -> str:
    """Run the Gemini client on *image_path* using *prompt* to get transcription."""

    # We only expect text back for transcription
    result = client.generate(prompt=prompt, image_path=image_path)

    if not isinstance(result, str):
        logger.warning(
            "Expected a string from Gemini for transcription of %s, but got %s. Converting.",
            image_path.name,
            type(result).__name__,
        )
        return str(result)
    return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    client: GeminiClient,
    prompt: str,
    max_workers: int = MAX_WORKERS,
) -> None:
    """Process every supported image inside *input_dir* and write transcriptions to *output_dir*."""
    all_images = _discover_images(input_dir)
    if not all_images:
        logger.warning("No images found in '%s'. Please add some images to process.", input_dir)
        return

    images_to_process = []
    for path in all_images:
        txt_output = output_dir / f"{path.stem}.txt"
        if not txt_output.exists():
            images_to_process.append(path)

    total_found = len(all_images)
    total_to_process = len(images_to_process)
    total_skipped = total_found - total_to_process

    logger.info("Found %d total images in '%s'.", total_found, input_dir)
    if total_skipped > 0:
        logger.info("Skipping %d images that have already been processed.", total_skipped)

    if not images_to_process:
        logger.info("All images have already been processed. Nothing to do.")
        return

    logger.info("Starting transcription process for %d new images…", total_to_process)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_image, path, client, prompt): path
            for path in images_to_process
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                result = future.result()
                logger.info(
                    "[%d/%d] Finished transcribing %s.",
                    i,
                    total_to_process,
                    path.name,
                )

                target_file = output_dir / f"{path.stem}.txt"
                target_file.write_text(result)

                logger.info("Saved transcription to %s", target_file)

            except Exception as exc:
                logger.error(
                    "[%d/%d] Failed to process %s after multiple retries: %s",
                    i,
                    total_to_process,
                    path.name,
                    exc,
                )


def _process_window(
    window_files: List[Path],
    client: GeminiClient,
    prompt_template: str,
) -> List[DetectedEntity]:
    """Processes a single window of transcriptions to extract entities."""
    window_page_numbers = [_extract_number_from_filename(p) for p in window_files]
    window_text = "\n\n---\n\n".join(p.read_text() for p in window_files)

    try:
        response_data = client.generate_structured_from_text(
            prompt_template=prompt_template,
            text_input=window_text,
            response_schema=LLMResponse,
        )

        if not response_data or not response_data.entities:
            logger.info("No entities found in window %s.", window_page_numbers)
            return []

        entities = [
            DetectedEntity(**item.model_dump(), page_numbers=window_page_numbers)
            for item in response_data.entities
        ]

        logger.info(
            "Extracted %d entities from window %s.",
            len(entities),
            window_page_numbers,
        )
        return entities

    except Exception as exc:
        logger.error(
            "Failed to extract entities from window %s: %s", window_page_numbers, exc
        )
        return []


def run_entity_extraction(
    input_dir: Path,
    client: GeminiClient,
    prompt_template: str,
    max_workers: int = MAX_WORKERS,
) -> None:
    """
    Run entity extraction on transcribed text files using a sliding window.

    This function reads text files from *input_dir*, processes them in
    overlapping windows to handle context that spans multiple pages, extracts
    structured entities, deduplicates the results, and saves them to a
    JSON file.
    """
    transcriptions = _discover_transcriptions(input_dir)
    if not transcriptions:
        logger.info("No transcriptions found in '%s' to extract entities from.", input_dir)
        return

    # Generate all unique windows of transcription files to be processed.
    windows_to_process: list[list[Path]] = []
    processed_keys = set()

    for i in range(0, len(transcriptions) - WINDOW_SIZE + 1, WINDOW_STEP):
        window_files = transcriptions[i : i + WINDOW_SIZE]
        key = tuple(p.name for p in window_files)
        if key not in processed_keys:
            windows_to_process.append(window_files)
            processed_keys.add(key)

    # Ensure the final window is always included if it was missed.
    if len(transcriptions) >= WINDOW_SIZE:
        final_window_files = transcriptions[-WINDOW_SIZE:]
        key = tuple(p.name for p in final_window_files)
        if key not in processed_keys:
            windows_to_process.append(final_window_files)
            processed_keys.add(key)

    if not windows_to_process:
        logger.info("No windows to process.")
        return

    logger.info(
        "Found %d transcriptions. Starting entity extraction for %d windows with window size %d and step %d.",
        len(transcriptions),
        len(windows_to_process),
        WINDOW_SIZE,
        WINDOW_STEP,
    )

    all_entities: List[DetectedEntity] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_window = {
            executor.submit(_process_window, window, client, prompt_template): window
            for window in windows_to_process
        }

        for i, future in enumerate(
            concurrent.futures.as_completed(future_to_window), 1
        ):
            window = future_to_window[future]
            try:
                entities_from_window = future.result()
                if entities_from_window:
                    all_entities.extend(entities_from_window)
            except Exception as exc:
                window_key = tuple(_extract_number_from_filename(p) for p in window)
                logger.error(
                    "[%d/%d] Window %s generated an unexpected exception: %s",
                    i,
                    len(windows_to_process),
                    list(window_key),
                    exc,
                )

    # Deduplicate entities based on their `full_identifier`. When duplicates
    # are found, the entity from a later page (the last one seen) is kept,
    # as it is assumed to be more complete from a wider context window.
    unique_entities: dict[str, DetectedEntity] = {}
    for entity in all_entities:
        # By overwriting the dictionary entry, we ensure that the last seen
        # version of an entity is the one that's kept.
        unique_entities[entity.full_identifier] = entity

    final_entities = list(unique_entities.values())
    logger.info(
        "Extracted %d total entities, with %d unique entities after initial deduplication.",
        len(all_entities),
        len(final_entities),
    )

    # Stage 2: Perform semantic deduplication on the remaining entities.
    if len(final_entities) > 1:
        final_entities = deduplicate_entities_by_similarity(
            entities=final_entities, client=client
        )

    # Sort entities by their first page number.
    final_entities.sort(key=lambda e: min(e.page_numbers))

    # Save the final list of entities to a JSON file.
    output_path = Path("output/entities.json")
    output_data = ExtractedEntities(entities=final_entities)
    output_path.write_text(output_data.model_dump_json(indent=2))

    logger.info("Saved %d unique entities to %s", len(final_entities), output_path)


def main() -> None:
    """Run the OCR pipeline."""
    parser = argparse.ArgumentParser(description="Run the OCR pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["transcribe", "extract", "all"],
        default="all",
        help="The mode to run: 'transcribe' for image transcription, "
        "'extract' for entity extraction from transcriptions, "
        "or 'all' to run both stages. Defaults to 'all'.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Suppress verbose logs from Google API client libraries
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.WARNING)
    load_dotenv()

    logger.info("Running in '%s' mode.", args.mode)

    prompt_name = os.environ.get("PROMPT", "TRANSCRIPTION_PROMPT")
    prompt_text = getattr(prompts, prompt_name, None)

    if prompt_text is None:
        logger.error(
            "Prompt '%s' not found in prompts.py. Falling back to default.",
            prompt_name,
        )
        prompt_name = "TRANSCRIPTION_PROMPT"
        prompt_text = prompts.TRANSCRIPTION_PROMPT

    logger.info("Using prompt: '%s'", prompt_name)

    input_dir = Path("input_images")
    output_dir = Path("output")

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    try:
        client = GeminiClient()
    except ValueError as exc:
        logger.error("%s", exc)
        logger.error("Please ensure GEMINI_API_KEY is available as an environment variable or inside a .env file.")
        return

    if args.mode in ["transcribe", "all"]:
        # Stage 1: Transcribe images to text.
        logger.info("--- Starting Transcription Stage ---")
        process_directory(
            input_dir,
            output_dir,
            client,
            prompt=prompt_text,
        )

    if args.mode in ["extract", "all"]:
        # Stage 2: Extract structured entities from transcriptions.
        logger.info("--- Starting Entity Extraction Stage ---")
        run_entity_extraction(
            input_dir=output_dir,
            client=client,
            prompt_template=prompts.ENTITY_EXTRACTION_PROMPT,
            max_workers=MAX_WORKERS,
        )


if __name__ == "__main__":
    main() 