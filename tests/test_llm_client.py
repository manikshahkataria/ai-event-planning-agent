"""Offline provider configuration and retry contract checks."""

import os
import unittest
from unittest.mock import Mock, call, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from planner import llm_client as llm
from planner.requirements import EVENT_SCHEMA
from planner.strategy import STRATEGY_SCHEMA
from planner.event_plan import EVENT_PLAN_SCHEMA
from planner.budget import BUDGET_SCHEMA
from planner.timeline import TIMELINE_SCHEMA


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.output = patch("builtins.print").start()
        patch.object(llm, "load_dotenv").start()
        patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-secret",
                               "OPENROUTER_API_KEY": "test-router-secret"}, clear=True).start()
        self.addCleanup(patch.stopall)

    def client(self, provider="groq"):
        return Mock(base_url=llm._PROVIDERS[provider]["base_url"] + "/")

    def test_default_groq_and_explicit_openrouter(self):
        with llm.create_llm_client() as client:
            self.assertEqual(str(client.base_url), "https://api.groq.com/openai/v1/")
            self.assertEqual(llm.get_llm_model(client), "openai/gpt-oss-120b")
            self.assertEqual(client.max_retries, 0)
            self.assertEqual((client.timeout.connect, client.timeout.read, client.timeout.write, client.timeout.pool), (5, 30, 10, 5))
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}):
            with llm.create_llm_client() as client:
                self.assertEqual(str(client.base_url), "https://openrouter.ai/api/v1/")
                self.assertEqual(llm.get_llm_model(client), "openrouter/free")
                self.assertEqual(client.max_retries, 0)

    def test_missing_key_and_invalid_provider_no_fallback(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                llm.create_llm_client()
        with patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}):
            with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                llm.create_llm_client()

    def test_all_schemas_pass_through_unchanged(self):
        for schema in (EVENT_SCHEMA, STRATEGY_SCHEMA, EVENT_PLAN_SCHEMA, BUDGET_SCHEMA, TIMELINE_SCHEMA):
            with self.subTest(schema=schema):
                client = self.client()
                messages = [{"role": "user", "content": "Fixture"}]
                output_format = {"type": "json_schema", "json_schema": {"name": "fixture", "strict": True, "schema": schema}}
                self.assertIs(llm.create_reliable_completion(client, messages, output_format), client.chat.completions.create.return_value)
                client.chat.completions.create.assert_called_once()
                args = client.chat.completions.create.call_args.kwargs
                self.assertIs(args["response_format"], output_format)
                self.assertIs(args["messages"], messages)
                self.assertNotIn("extra_body", args)
                self.assertEqual(args["model"], "openai/gpt-oss-120b")

    def test_openrouter_options_only_for_openrouter(self):
        client = self.client("openrouter")
        with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}):
            llm.create_reliable_completion(client, [], {})
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "openrouter/free")
        self.assertTrue(args["extra_body"]["provider"]["require_parameters"])
        self.assertNotIn("reasoning_effort", args)

    def test_retryable_errors_three_attempts(self):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        errors = [APITimeoutError(request=request), APIConnectionError(request=request)]
        errors += [APIStatusError("test-groq-secret", response=httpx.Response(code, request=request), body=None) for code in (408, 409, 429, 500, 503)]
        for error in errors:
            client = self.client()
            client.chat.completions.create.side_effect = error
            with patch.object(llm.time, "sleep") as sleep:
                self.assertIsNone(llm.create_reliable_completion(client, [], {}))
            self.assertEqual(client.chat.completions.create.call_count, 3)
            self.assertEqual(sleep.call_args_list, [call(1), call(2)])
        self.assertNotIn("test-groq-secret", str(self.output.call_args_list))

    def test_nonretryable_and_unexpected_errors(self):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        for error in (APIStatusError("Bad schema", response=httpx.Response(400, request=request), body=None), RuntimeError("test-groq-secret")):
            client = self.client()
            client.chat.completions.create.side_effect = error
            with patch.object(llm.time, "sleep") as sleep:
                self.assertIsNone(llm.create_reliable_completion(client, [], {}))
                sleep.assert_not_called()
            self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertNotIn("test-groq-secret", str(self.output.call_args_list))

    def test_single_attempt_option_and_visible_messages(self):
        client = self.client()
        client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        with patch.object(llm.time, "sleep") as sleep:
            self.assertIsNone(llm.create_reliable_completion(client, [], {}, max_retries=0))
            sleep.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 1)
        attempts = [c for c in self.output.call_args_list if "Contacting" in str(c)]
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].kwargs["flush"])


if __name__ == "__main__":
    unittest.main()
