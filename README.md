# seb-ocr

<img src="./seb_ocr_logo.png" width=600>

A vision language model pipeline for transcribing scanned historical documents with Google's [Gemini](https://ai.google.dev/gemini-api/docs/models) models, built specifically for ongoing political science research.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/ALucek/seb-ocr.git
    cd seb-ocr
    ```

2. Install dependencies using [uv](https://docs.astral.sh/uv/):

    ```bash
    uv sync
    ```

3.  Create a `.env` file for your relevant environment variables.

    ```ini
    GEMINI_API_KEY="your-secret-key"
    PROMPT="HISTORICAL_DOCUMENT_PROMPT"
    # Optionally override the model
    # GEMINI_MODEL="gemini-2.5-flash"
    ```

4.  Drop images in the `input_images/` directory (created automatically on first run).

5.  Run the pipeline:

    ```bash
    uv run main.py
    ```

    Results will be written to `output_text/` – one JSON file per image. If the model returns invalid JSON the raw output is stored alongside with a `_error.txt` suffix for manual inspection.

## Adding/Changing Prompts

All prompts are stored as Python strings inside `prompts.py`. You can add new prompts by creating new string variables in this file.

To switch between prompts, set the `PROMPT` environment variable in your `.env` file to the name of the desired prompt variable (e.g., `PROMPT="YOUR_NEW_PROMPT_NAME"`). If this variable is not set, the pipeline defaults to using `HISTORICAL_DOCUMENT_PROMPT`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) for details.
