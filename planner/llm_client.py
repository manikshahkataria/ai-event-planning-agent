import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "openrouter/free",
    },
}
_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


def create_llm_client():
    """Configure one provider; default to Groq, without automatic fallback."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError("LLM_PROVIDER must be groq or openrouter.")
    config = _PROVIDERS[provider]
    api_key = os.getenv(config["key_env"], "").strip()
    if not api_key:
        raise ValueError(f"{config['key_env']} is not configured.")
    return OpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=_TIMEOUT,
    )


def _client_provider(client):
    # Bind model/options to the actual endpoint, not a mutable environment value.
    base_url = str(client.base_url).rstrip("/")
    for provider, config in _PROVIDERS.items():
        if base_url == config["base_url"]:
            return provider
    raise ValueError("Unsupported LLM client endpoint.")


def get_llm_model(client):
    """Expose the centralized model for the existing planner signatures."""
    return _PROVIDERS[_client_provider(client)]["model"]


def create_reliable_completion(
    client,
    messages,
    response_format,
    max_retries=2,
):
    """
    Send a structured completion with application-controlled retries.
    Phase timeouts are not a strict overall operation deadline.
    """

    try:
        provider = _client_provider(client)
        model = get_llm_model(client)
    except Exception:
        print("\nAI provider configuration is unavailable.")
        return None

    options = {}
    if provider == "openrouter":
        options["extra_body"] = {
            "provider": {
                "allow_fallbacks": True,
                "require_parameters": True,
                "sort": "latency",
            }
        }
    else:
        options["reasoning_effort"] = "low"

    for attempt in range(max_retries + 1):

        try:

            print(
                f"Contacting AI provider {provider} / {model} "
                f"(attempt {attempt + 1}/{max_retries + 1})...",
                flush=True,
            )

            response = client.chat.completions.create(
                model=model,

                messages=messages,

                response_format=response_format,

                timeout=_TIMEOUT,
                **options,
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
