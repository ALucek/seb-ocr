from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel
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

        default_embedding_model = "gemini-embedding-exp-03-07"
        self.embedding_model_name = os.environ.get(
            "GEMINI_EMBEDDING_MODEL", default_embedding_model
        )

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via GEMINI_API_KEY environment variable."
            )

        self._client = genai.Client(api_key=self.api_key)
        logging.info("Using Gemini model: %s", self.model_name)
        logging.info("Using Gemini embedding model: %s", self.embedding_model_name)

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
    def generate(
        self,
        prompt: str,
        image_path: Union[str, Path],
    ) -> Any:
        """Generate text from an image & prompt via Gemini.

        The method always returns the *raw* text returned by the model. The
        caller is responsible for any downstream parsing/validation.
        """

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            with PIL.Image.open(image_path) as image:
                # Build arguments for the generate_content call
                call_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "contents": [prompt, image],
                }

                response = self._client.models.generate_content(**call_kwargs)

            text_response: str = response.text.strip()
            # The model sometimes wraps the response in markdown code fences – strip them.
            text_response = re.sub(r"^```[a-z]*\n?", "", text_response)
            text_response = re.sub(r"\n?```$", "", text_response)

            return text_response
        except google_exceptions.GoogleAPIError:
            # Propagate Google API errors so the tenacity retry decorator can
            # handle them properly.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate content with Gemini API model: {exc}"
            ) from exc

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
    def generate_structured_from_text(
        self,
        prompt_template: str,
        text_input: str,
        response_schema: Union[Type[BaseModel], list[Type[BaseModel]]],
    ) -> Any:
        """Generate structured data from text via Gemini.

        The method returns parsed Pydantic objects as specified by the
        `response_schema`.
        """
        prompt = prompt_template.format(transcription=text_input)

        try:
            # Build arguments for the generate_content call
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
            return response.parsed
        except google_exceptions.FailedPrecondition as exc:
            logging.error(
                "Gemini API failed to follow the response schema. Response text: %s",
                response.text,
            )
            raise RuntimeError(f"Failed to parse Gemini response into schema: {exc}") from exc
        except google_exceptions.GoogleAPIError:
            # Propagate Google API errors so the tenacity retry decorator can
            # handle them properly.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate structured content with Gemini API model: {exc}"
            ) from exc

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
    def embed_contents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts in a single batch."""
        if not texts:
            return []
        try:
            # The API expects the keyword argument `contents` for batching.
            result = self._client.models.embed_content(
                model=self.embedding_model_name, contents=texts
            )
            # The response contains a list of ContentEmbedding objects; we need to
            # extract the `values` (the actual vector) from each one.
            return [e.values for e in result.embeddings]
        except google_exceptions.GoogleAPIError:
            # Propagate Google API errors for tenacity to handle.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate embeddings with Gemini API model: {exc}"
            ) from exc
