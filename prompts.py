# prompts.py

"""Prompt templates shared across the OCR pipeline.

The templates are kept in a standalone module so that they can be edited
or swapped without touching the processing logic.
"""

HISTORICAL_DOCUMENT_PROMPT: str = """
You are an expert OCR system for historical documents. Your task has TWO mandatory phases:

**PHASE 1: COMPLETE TRANSCRIPTION (REQUIRED)**
Transcribe EVERY visible text element on this page, including:
- All headers, titles, and document markings
- Every word of body text
- Page numbers, file numbers, stamps
- Marginal notes and annotations
- ANY text visible anywhere on the page

Preserve the exact layout and formatting as much as possible. Use [illegible] for unreadable text and [?] for uncertain readings.

**PHASE 2: STRUCTURED EXTRACTION**
After completing the full transcription, analyze it to extract individual entries. Each entry should have:
- **individual**: Person's name only (e.g., "Babu Subh Narain Choudhry")
- **title_or_position**: Professional title or role (e.g., "Manager", "Barrister", "Sub-Divisional Officer")
- **location**: Geographic location ONLY (e.g., "Raj Darbhanga, Kajha, P. O., Purnea")
- **full_identifier**: The complete original text as written (e.g., "Manager, Raj Darbhanga, Kajha, P. O., Purnea")
- **text**: Their complete statement/opinion

Parse carefully to separate:
- Titles/honorifics that are part of names (Mr., Mrs., Babu, etc.) stay with the name
- Professional positions (Manager, Barrister, etc.) go in title_or_position
- Geographic locations (cities, districts, P.O., addresses) go in location
- Keep the full_identifier exactly as written for reference

If no clear individual entries exist (e.g., it's a cover page, summary, or narrative text), still provide the full transcription but note that no individual entries were found.

**OUTPUT JSON STRUCTURE:**
```json
{
  "full_transcription": {
    "header_text": "[Any text at top of page including 'CONFIDENTIAL', page numbers, etc.]",
    "body_text": "[Complete transcription of main content preserving line breaks with \n]",
    "marginal_notes": "[Any text in margins]",
    "footer_text": "[Any text at bottom]"
  },
  "extracted_entries": [
    {
      "entry_id": 1,
      "individual": "[Name as written, or null if none found]",
      "title_or_position": "[Professional title, job, or position if any]",
      "location": "[Geographic location only - city, district, P.O., address]",
      "full_identifier": "[Complete identifier as written in original]",
      "text": "[Their statement/opinion]",
      "source_lines": "[Which lines in transcription this came from]"
    }
  ],
  "document_metadata": {
    "page_type": "petition_entries|cover_page|summary|mixed",
    "total_entries_extracted": 0,
    "extraction_confidence": "high|medium|low",
    "transcription_issues": ["List any OCR challenges"],
    "has_individual_entries": true/false
  }
}
```

**CRITICAL REQUIREMENTS:**
1. ALWAYS complete Phase 1 (full transcription) even if the page has no petition entries
2. The full_transcription must include EVERY piece of text visible
3. Preserve original spelling, capitalization, and punctuation exactly
4. Include line breaks in transcription to maintain document structure
5. Phase 2 extraction is attempted on every page, but may find zero entries

**TRANSCRIPTION FORMATTING:**
- Use \n for line breaks
- Use \n\n for paragraph breaks
- Preserve indentation with spaces
- Mark column breaks with " | " if text is in columns
- Include all dashes, dots, and other separators exactly as shown

**EXTRACTION NOTES:**
- Some pages may be title pages, summaries, or explanatory text with no individual entries
- If a page contains mixed content (some entries + some narrative), extract what you can
- For pages with no extractable entries, explain why in the metadata

Begin with complete transcription of ALL visible text.
""" 