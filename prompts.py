TRANSCRIPTION_PROMPT: str = """
You are an expert OCR system specializing in historical government documents, legal papers, and administrative records. Your task is to create an accurate, complete transcription that preserves the document's structure and content.

## TRANSCRIPTION REQUIREMENTS:

### Text Capture:
- Transcribe **ALL visible text** including headers, titles, body text, footnotes, page numbers, and marginal notes
- Include **watermarks, stamps, and handwritten annotations**
- Capture **underlined text** using `<u>text</u>` tags
- Preserve **bold text** using `**text**` 
- Maintain original **spelling, capitalization, punctuation, and abbreviations**

### Structural Elements:
- Use `---` for horizontal lines or dividers
- Use `[CENTERED]` before centered text blocks
- Indicate indentation levels with appropriate spacing
- Preserve table structures using markdown table format
- Mark section breaks with double line breaks

### Handling Uncertainties:
- Use `[illegible]` for completely unreadable text
- Use `[word?]` for uncertain but probable readings
- Use `[...]` for missing or cut-off text
- Use `[faded]` for barely visible text

### Special Notations:
- Include `[Page X]` at the beginning of each page
- Note `[Handwritten: text]` for handwritten additions
- Mark `[Stamp: description]` for official stamps or seals
- Use `[Watermark: text]` for watermarked content

### Layout Preservation:
- Maintain original paragraph structure
- Preserve spacing between sections
- Keep numbered/lettered lists in their original format
- Align text as it appears (left, center, right)

### Quality Standards:
- Prioritize accuracy over speed
- When multiple interpretations are possible, choose the most contextually appropriate
- Cross-reference similar words/phrases within the document for consistency
- Pay special attention to proper nouns, dates, and official titles

## OUTPUT FORMAT:
Provide ONLY the transcribed text with structural markup. Do not include explanatory comments, metadata, or analysis.
"""

ENTITY_EXTRACTION_PROMPT: str = """
You are an expert in extracting structured information from historical documents.
Based on the following transcription, analyze it to extract individual entries.

**TRANSCRIPTION:**
{transcription}

**INSTRUCTIONS:**
Extract individual entries. Each entry should have:
- **individual**: Person's name only (e.g., "Babu Subh Narain Choudhry")
- **title_or_position**: Professional title or role (e.g., "Manager", "Barrister", "Sub-Divisional Officer")
- **location**: Geographic location ONLY (e.g., "Raj Darbhanga, Kajha, P. O., Purnea")
- **full_identifier**: The complete original text as written (e.g., "Manager, Raj Darbhanga, Kajha, P. O., Purnea")
- **text**: Their complete statement/opinion

**PARSING RULES:**
- Titles/honorifics that are part of names (Mr., Mrs., Babu, etc.) stay with the name.
- Professional positions (Manager, Barrister, etc.) go in title_or_position.
- Geographic locations (cities, districts, P.O., addresses) go in location.
- Keep the full_identifier exactly as written for reference.

If no clear individual entries exist (e.g., it's a cover page, summary, or narrative text), return an empty list.
""" 