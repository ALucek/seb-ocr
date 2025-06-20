from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, Union

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

__all__ = ["GeminiClient"]

# Load environment variables from a .env file if present.
load_dotenv()

# Helper for tenacity retry logging
def _log_retry_attempt(retry_state):
    """Log information before we sleep between retry attempts."""

    logging.warning(
        "Retrying Gemini API call due to error: %s. Attempt #%d, waiting %.2fs…",
        retry_state.outcome.exception(),
        retry_state.attempt_number,
        retry_state.next_action.sleep,
    )


class GeminiClient:
    """A small convenience wrapper around the google-genai client."""

    def __init__(self, api_key: Optional[str] | None = None, model: Optional[str] | None = None):
        """Create a new :class:`GeminiClient` instance.

        Parameters
        ----------
        api_key
            The Gemini API key. If *None* the value is taken from the
            ``GEMINI_API_KEY`` environment variable.
        model
            Name of the generative model to use. If *None* the value is taken
            from the ``GEMINI_MODEL`` environment variable, defaulting to
            ``"gemini-2.5-flash"``.
        """

        default_model = "gemini-2.5-flash"
        self.model_name = model or os.environ.get("GEMINI_MODEL", default_model)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via GEMINI_API_KEY environment variable."
            )

        self._client = genai.Client(api_key=self.api_key)
        logging.info("Using Gemini model: %s", self.model_name)

    # Main generate method

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.GatewayTimeout,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry_attempt,
    )
    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """Generate structured text from an image & prompt via Gemini.

        The method always returns the *raw* text returned by the model. The
        caller is responsible for any downstream parsing/validation.
        """

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            with PIL.Image.open(image_path) as image:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=[prompt, image]
                )

            text_response: str = response.text.strip()
            # The model sometimes wraps JSON in markdown code fences – strip them.
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]

            return text_response
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate content with Gemini API model: {exc}"
            ) from exc 