# seb-ocr

A minimal proof-of-concept for transcribing historical documents with Google's **Gemini** models.

The project purposefully keeps its codebase small and readable. The core responsibilities are split into three modules:

* `gemini_client.py` – thin wrapper around the Google GenAI SDK with sensible retry logic.
* `prompts.py` – collection of prompt templates.
* `pipeline.py` – orchestrates the OCR workflow (load images → call Gemini → write JSON).
* `cli.py` – command-line entry-point.

## Quick start

1.  Install the project in editable mode (which also installs the dependencies):

    ```bash
    pip install -e .
    ```

2.  Add your Gemini API key to a `.env` file:

    ```ini
    GEMINI_API_KEY="your-secret-key"
    # Optionally override the model
    # GEMINI_MODEL="gemini-2.5-flash"
    ```

3.  Drop images in the `input_images/` directory (created automatically on first run).

4.  Run the pipeline:

    ```bash
    seb-ocr
    ```

    Results will be written to `output_text/` – one JSON file per image. If the model returns invalid JSON the raw output is stored alongside with a `_error.txt` suffix for manual inspection.

## License

MIT © Sebastian …
