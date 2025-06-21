# Two-Phase Processing Prompts

TRANSCRIPTION_ONLY_PROMPT: str = """
You are an expert OCR system for historical documents. Your ONLY task is to transcribe ALL visible text on this page accurately.

**TRANSCRIPTION REQUIREMENTS:**
- Transcribe EVERY visible text element including headers, body text, page numbers, stamps, marginal notes
- Preserve original spelling, capitalization, and punctuation exactly
- Use [illegible] for unreadable text and [?] for uncertain readings
- Maintain natural flow - use line breaks (\n) only where they naturally occur
- Do NOT attempt to parse or extract individual entries - just transcribe everything as clean, flowing text

**CRITICAL NOTES:**
- Focus ONLY on accurate text transcription
- Do NOT try to identify individual petition entries or people
- Do NOT structure the text beyond natural paragraph breaks
- If text is in columns, transcribe left to right, top to bottom
- Include ALL text elements even if they seem like administrative markings

For confidence level, use "high" if all text is clearly readable, "medium" if some parts are unclear but readable, and "low" if significant portions are difficult to read.

List any OCR challenges or unclear sections in the issues field.

Begin transcription now.
"""

ENTITY_EXTRACTION_PROMPT: str = """
You are an expert at extracting individual petition entries from historical document transcriptions.

**YOUR TASK:**
Analyze the following transcribed text and extract individual petition entries. Each entry typically contains:
- A person's name and identifying information
- Their location/address
- Their opinion or statement on the matter

**ENTRY IDENTIFICATION PATTERNS:**
- Often separated by horizontal lines (represented as multiple dashes or equals signs)
- Usually starts with a name, title, or identifying number
- May include titles like "Babu", "Mr.", "Mrs.", professional roles like "Manager", "Barrister"
- Geographic locations like city names, "P.O.", district names
- Followed by their statement or opinion

**EXTRACTION RULES:**
1. **individual**: Person's name including honorifics (e.g., "Babu Subh Narain Choudhry", "Mr. John Smith"). THIS IS REQUIRED - every entry must have a name. If unclear, use "[Name unclear]" or similar placeholder.
2. **title_or_position**: Professional title or role (e.g., "Manager", "Sub-Divisional Officer")
3. **location**: Geographic location only (e.g., "Raj Darbhanga, Kajha, P. O., Purnea")
4. **full_identifier**: Complete original identifier as written
5. **text**: Their complete statement/opinion
6. **confidence_score**: 0.0-1.0 based on how certain you are this is a complete, accurate entry

**IMPORTANT GUIDELINES:**
- Only extract entries that appear complete (have both identification and statement)
- If an entry appears to be cut off at the beginning or end, reduce the confidence_score accordingly
- Skip administrative headers, page numbers, and non-petition content
- When in doubt about boundaries, err on the side of including more text rather than truncating
- Use null for title_or_position or location if not clearly identifiable

**TEXT TO ANALYZE:**
{text}

Extract all identifiable petition entries from this text. Return them as a list where each entry contains the fields described above.
"""