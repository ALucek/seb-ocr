from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from gemini_client import GeminiClient
from prompts import HISTORICAL_DOCUMENT_PROMPT

logger = logging.getLogger(__name__)


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


def process_image(image_path: Path, client: GeminiClient, prompt: str) -> dict | str:
    """Run the Gemini client on *image_path* using *prompt*."""
    structured_text = client.generate(prompt=prompt, image_path=image_path)
    try:
        return json.loads(structured_text)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON for %s", image_path.name)
        return structured_text


def process_directory(
    input_dir: Path,
    output_dir: Path,
    client: GeminiClient,
    prompt: str = HISTORICAL_DOCUMENT_PROMPT,
) -> None:
    """Process every supported image inside *input_dir* and write JSON to *output_dir*."""
    all_images = _discover_images(input_dir)
    if not all_images:
        logger.warning("No images found in '%s'. Please add some images to process.", input_dir)
        return

    images_to_process = []
    for path in all_images:
        json_output = output_dir / f"{path.stem}.json"
        error_output = output_dir / f"{path.stem}_error.txt"
        if not (json_output.exists() or error_output.exists()):
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

    logger.info("Starting OCR process for %d new images…", total_to_process)

    for i, path in enumerate(images_to_process, 1):
        logger.info("Processing image %d of %d: %s…", i, total_to_process, path.name)
        result = process_image(path, client, prompt)

        if isinstance(result, dict):
            target_file = output_dir / f"{path.stem}.json"
            with target_file.open("w") as fh:
                json.dump(result, fh, indent=4)
        else:
            target_file = output_dir / f"{path.stem}_error.txt"
            target_file.write_text(result)

        logger.info("Saved output to %s", target_file)


def main() -> None:
    """Run the OCR pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_dotenv()

    input_dir = Path("input_images")
    output_dir = Path("output_text")

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    try:
        client = GeminiClient()
    except ValueError as exc:
        logger.error("%s", exc)
        logger.error("Please ensure GEMINI_API_KEY is available as an environment variable or inside a .env file.")
        return

    process_directory(input_dir, output_dir, client)


if __name__ == "__main__":
    main() 