from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

from dotenv import load_dotenv
from gemini_client import GeminiClient
import prompts
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

MAX_WORKERS = 10
CHUNK_OVERLAP_CHARS = 2000  # Character overlap between chunks
MAX_CHUNK_SIZE = 40000     # Target chunk size in characters


@dataclass
class TranscriptionResult:
    """Result from Phase 1 transcription."""
    page_number: int
    filename: str
    transcribed_text: str
    confidence: str
    issues: List[str]


@dataclass
class ExtractedEntry:
    """Individual entry extracted in Phase 2."""
    entry_id: str
    individual: str  # Now required
    title_or_position: Optional[str]
    location: Optional[str]
    full_identifier: str
    text: str
    source_chunk: int
    confidence_score: float
    text_hash: str  # For deduplication


# Pydantic models for structured output from Gemini
class TranscriptionModel(BaseModel):
    """Pydantic model for transcription output."""
    transcribed_text: str = Field(description="Complete text transcription as a single flowing text block")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    issues: List[str] = Field(default_factory=list, description="List of OCR challenges or unclear sections")

class ExtractedEntryModel(BaseModel):
    """Pydantic model for a single extracted entry."""
    individual: str = Field(description="The name of the individual, including any honorifics (e.g., 'Babu Subh Narain Choudhry', 'Mr. John Smith'). This field is required - if no clear name is found, use '[Name unclear]' or similar.")
    title_or_position: Optional[str] = Field(None, description="The title or position of the individual.")
    location: Optional[str] = Field(None, description="The location associated with the entry.")
    full_identifier: str = Field(description="The full identifying string for the person (e.g., 'John Smith, blacksmith, of London').")
    text: str = Field(description="The full original text of the entry.")
    confidence_score: float = Field(description="A confidence score from 0.0 to 1.0 for the extraction.", ge=0.0, le=1.0)

# We will ask Gemini to return a list of ExtractedEntryModel directly
ExtractionResponseSchema = list[ExtractedEntryModel]


class TwoPhaseProcessor:
    """Two-phase OCR processor for historical documents."""
    
    def __init__(self, client: GeminiClient):
        self.client = client
        self.transcription_results: List[TranscriptionResult] = []
        self.extracted_entries: List[ExtractedEntry] = []
    
    def process_directory(self, input_dir: Path, output_dir: Path, max_workers: int = MAX_WORKERS):
        """Run the complete two-phase processing pipeline."""
        logger.info("Starting two-phase processing pipeline...")
        
        # Phase 1: Transcription
        self._phase1_transcription(input_dir, output_dir, max_workers)
        
        # Phase 2: Entity extraction
        self._phase2_entity_extraction(output_dir)
        
        # Phase 3: Conflict resolution and final output
        self._phase3_conflict_resolution(output_dir)
        
        logger.info("Two-phase processing complete!")
    
    def _phase1_transcription(self, input_dir: Path, output_dir: Path, max_workers: int):
        """Phase 1: Pure transcription of all pages."""
        logger.info("=== PHASE 1: TRANSCRIPTION ===")
        
        transcription_dir = output_dir / "phase1_transcriptions"
        transcription_dir.mkdir(exist_ok=True)
        
        all_images = self._discover_images(input_dir)
        if not all_images:
            logger.warning("No images found in '%s'", input_dir)
            return
        
        # Check for existing transcriptions
        images_to_process = []
        for path in all_images:
            transcription_file = transcription_dir / f"{path.stem}.json"
            if not transcription_file.exists():
                images_to_process.append(path)
        
        if not images_to_process:
            logger.info("All transcriptions already exist. Loading existing results...")
            self._load_existing_transcriptions(transcription_dir, all_images)
            return
        
        logger.info("Transcribing %d new images...", len(images_to_process))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._transcribe_single_page, path): path
                for path in images_to_process
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_path), 1):
                path = future_to_path[future]
                try:
                    result = future.result()
                    self.transcription_results.append(result)
                    
                    # Save individual transcription
                    output_file = transcription_dir / f"{path.stem}.json"
                    with output_file.open("w") as f:
                        json.dump(asdict(result), f, indent=2)
                    
                    logger.info("[%d/%d] Transcribed %s", i, len(images_to_process), path.name)
                    
                except Exception as exc:
                    logger.error("Error transcribing %s: %r", path.name, exc)
        
        # Load any existing transcriptions we didn't process
        self._load_existing_transcriptions(transcription_dir, all_images)
        
        # Sort by page number
        self.transcription_results.sort(key=lambda x: x.page_number)
        
        logger.info("Phase 1 complete. Transcribed %d pages total.", len(self.transcription_results))
    
    def _transcribe_single_page(self, image_path: Path) -> TranscriptionResult:
        """Transcribe a single page using structured output."""
        page_number = self._extract_number_from_filename(image_path)
        
        # Use simplified transcription prompt with structured output
        prompt = prompts.TRANSCRIPTION_ONLY_PROMPT
        
        try:
            response = self.client.generate(
                prompt=prompt, 
                image_path=image_path,
                response_schema=TranscriptionModel
            )
            
            # Response is now a TranscriptionModel instance
            return TranscriptionResult(
                page_number=page_number,
                filename=image_path.name,
                transcribed_text=response.transcribed_text,
                confidence=response.confidence,
                issues=response.issues
            )
            
        except Exception as exc:
            return TranscriptionResult(
                page_number=page_number,
                filename=image_path.name,
                transcribed_text="",
                confidence="failed",
                issues=[f"Transcription failed: {str(exc)}"]
            )
    
    def _phase2_entity_extraction(self, output_dir: Path):
        """Phase 2: Extract entities from complete transcribed text."""
        logger.info("=== PHASE 2: ENTITY EXTRACTION ===")
        
        if not self.transcription_results:
            logger.error("No transcription results available for entity extraction")
            return
        
        # Check if client has text-only method
        if not hasattr(self.client, 'generate_text_only'):
            logger.error("GeminiClient does not have generate_text_only method. Please update gemini_client.py")
            return
        
        # Combine all transcribed text
        full_text = self._combine_transcriptions()
        
        # Create overlapping chunks
        chunks = self._create_overlapping_chunks(full_text)
        logger.info("Created %d overlapping chunks for extraction", len(chunks))
        
        # Extract entities from each chunk
        all_extractions = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (length: %d chars)...", i+1, len(chunks), len(chunk))
            try:
                chunk_extractions = self._extract_entities_from_chunk(chunk, i)
                all_extractions.extend(chunk_extractions)
                logger.info("Extracted %d entries from chunk %d", len(chunk_extractions), i)
            except Exception as exc:
                logger.error("Error processing chunk %d: %r", i, exc)
        
        # Store raw extractions
        raw_output = output_dir / "phase2_raw_extractions.json"
        with raw_output.open("w") as f:
            json.dump([asdict(entry) for entry in all_extractions], f, indent=2)
        
        self.extracted_entries = all_extractions
        logger.info("Phase 2 complete. Extracted %d raw entries.", len(all_extractions))
    
    def _phase3_conflict_resolution(self, output_dir: Path):
        """Phase 3: Resolve conflicts and create final output."""
        logger.info("=== PHASE 3: CONFLICT RESOLUTION ===")
        
        if not self.extracted_entries:
            logger.error("No extracted entries available for conflict resolution")
            return
        
        # Deduplicate entries
        deduplicated = self._deduplicate_entries()
        
        # Flag low-confidence entries
        flagged_entries = [entry for entry in deduplicated if entry.confidence_score < 0.7]
        
        # Create final output
        final_output = {
            "processing_metadata": {
                "total_pages_processed": len(self.transcription_results),
                "total_entries_extracted": len(deduplicated),
                "entries_flagged_for_review": len(flagged_entries),
                "processing_date": None  # Could add timestamp
            },
            "entries": [asdict(entry) for entry in deduplicated],
            "flagged_entries": [asdict(entry) for entry in flagged_entries],
            "transcription_summary": {
                "pages_with_issues": [
                    {"page": r.page_number, "filename": r.filename, "issues": r.issues}
                    for r in self.transcription_results if r.issues
                ]
            }
        }
        
        # Save final results
        final_output_file = output_dir / "final_extracted_entries.json"
        with final_output_file.open("w") as f:
            json.dump(final_output, f, indent=2)
        
        # Save full transcription for reference
        full_transcription_file = output_dir / "complete_transcription.txt"
        with full_transcription_file.open("w") as f:
            f.write(self._combine_transcriptions())
        
        logger.info("Phase 3 complete. Final output saved to %s", final_output_file)
        logger.info("Summary: %d entries extracted, %d flagged for review", 
                   len(deduplicated), len(flagged_entries))
    
    def _combine_transcriptions(self) -> str:
        """Combine all transcription results into a single text."""
        combined = []
        for result in sorted(self.transcription_results, key=lambda x: x.page_number):
            combined.append(f"=== PAGE {result.page_number} ({result.filename}) ===\n")
            combined.append(result.transcribed_text)
            combined.append("\n\n")
        return "".join(combined)
    
    def _create_overlapping_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text for processing."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + MAX_CHUNK_SIZE
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (page boundary or paragraph)
            break_point = end
            
            # Look for page boundaries first
            page_marker = text.rfind("=== PAGE", start, end)
            if page_marker > start + MAX_CHUNK_SIZE // 2:
                break_point = page_marker
            else:
                # Look for paragraph breaks
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + MAX_CHUNK_SIZE // 2:
                    break_point = para_break
            
            chunks.append(text[start:break_point])
            
            # Set next start with overlap
            start = max(start + 1, break_point - CHUNK_OVERLAP_CHARS)
        
        return chunks
    
    def _extract_entities_from_chunk(self, chunk: str, chunk_id: int) -> List[ExtractedEntry]:
        """Extract entities from a single chunk."""
        prompt = prompts.ENTITY_EXTRACTION_PROMPT.format(text=chunk)
        
        try:
            # Use text-only generation method with structured output
            response_data = self.client.generate_text_only(
                prompt=prompt,
                response_schema=ExtractionResponseSchema
            )
            
            if not response_data:
                logger.warning("Structured response for chunk %d was empty.", chunk_id)
                return []
            
            entries = []
            for i, entry_data in enumerate(response_data):
                entry = ExtractedEntry(
                    entry_id=f"chunk_{chunk_id}_entry_{i}",
                    individual=entry_data.individual,
                    title_or_position=entry_data.title_or_position,
                    location=entry_data.location,
                    full_identifier=entry_data.full_identifier,
                    text=entry_data.text,
                    source_chunk=chunk_id,
                    confidence_score=entry_data.confidence_score,
                    text_hash=self._hash_text(entry_data.text)
                )
                entries.append(entry)
            
            return entries
            
        except Exception as exc:
            logger.error("Failed to extract entities from chunk %d: %r", chunk_id, exc)
            logger.error("Exception type: %s", type(exc).__name__)
            import traceback
            logger.error("Full traceback: %s", traceback.format_exc())
            return []
    
    def _deduplicate_entries(self) -> List[ExtractedEntry]:
        """Remove duplicate entries based on text similarity."""
        seen_hashes = set()
        deduplicated = []
        
        # Sort by confidence score (descending) to prefer higher-confidence duplicates
        sorted_entries = sorted(self.extracted_entries, key=lambda x: x.confidence_score, reverse=True)
        
        for entry in sorted_entries:
            if entry.text_hash not in seen_hashes:
                seen_hashes.add(entry.text_hash)
                deduplicated.append(entry)
        
        return deduplicated
    
    def _hash_text(self, text: str) -> str:
        """Create a hash of text for deduplication."""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _discover_images(self, directory: Path) -> List[Path]:
        """Return a sorted list of image files inside directory."""
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(directory.glob(pattern))
        
        paths.sort(key=self._extract_number_from_filename)
        return paths
    
    def _extract_number_from_filename(self, path: Path) -> int:
        """Return an integer found in path to allow natural sorting."""
        match = re.search(r"(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    def _load_existing_transcriptions(self, transcription_dir: Path, all_images: List[Path]):
        """Load existing transcription results."""
        for image_path in all_images:
            transcription_file = transcription_dir / f"{image_path.stem}.json"
            if transcription_file.exists():
                try:
                    with transcription_file.open() as f:
                        data = json.load(f)
                    result = TranscriptionResult(**data)
                    self.transcription_results.append(result)
                except Exception as exc:
                    logger.error("Error loading transcription %s: %r", transcription_file, exc)


def main() -> None:
    """Run the two-phase OCR pipeline."""
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
        return

    processor = TwoPhaseProcessor(client)
    processor.process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()