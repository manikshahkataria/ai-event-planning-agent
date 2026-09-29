import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError


def create_reliable_completion(
    client,
    messages,
    response_format,
    max_retries=2,
):
    """
    Send an LLM request using OpenRouter's free
    model router with retry handling.

    This keeps the application on free models
    while reducing failures caused by temporary
    provider overloads and rate limits.
    """

    model = "openrouter/free"

    for attempt in range(max_retries + 1):

        try:

            print(
                f"Contacting AI provider (attempt {attempt + 1}/{max_retries + 1})...",
                flush=True,
            )

            response = client.chat.completions.create(
                model=model,

                messages=messages,

                response_format=response_format,

                extra_body={
                    "provider": {
                        "allow_fallbacks": True,
                        "require_parameters": True,
                        "sort": "latency",
                    }
                },

                timeout=httpx.Timeout(
                    connect=5.0, read=30.0, write=10.0, pool=5.0
                ),
            )

            return response

        except Exception as error:

            temporary_error = isinstance(
                error, (APITimeoutError, APIConnectionError)
            ) or (
                isinstance(error, APIStatusError)
                and (
                    error.status_code in (408, 409, 429)
                    or 500 <= error.status_code < 600
                )
            )

            if (
                temporary_error
                and attempt < max_retries
            ):

                wait_time = 2 ** attempt

                print(
                    "\n⚠️ AI provider temporarily busy."
                )

                print(
                    f"Retrying in {wait_time} second(s)...",
                    flush=True,
                )

                time.sleep(wait_time)

                continue

            print(
                "\n❌ AI service is currently unavailable."
            )

            print(
                "Please try again shortly."
            )

            return None
