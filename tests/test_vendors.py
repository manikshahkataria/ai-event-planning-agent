"""Offline vendor integration tests; all HTTP access is mocked."""

import ast
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from planner import vendors


def body(value):
    return BytesIO(json.dumps(value).encode())


def feature(place_id="one", name="Fixture Restaurant"):
    return {"properties": {"name": name, "place_id": place_id,
                           "formatted": "Fixture address", "city": "Delhi",
                           "categories": ["catering.restaurant"]}}


def category(types, name="Dining reservation"):
    return {"category": name, "covers": ["venue", "food", "music"],
            "reason": "Bundled approach", "vendor_types": types}


GEO = {"results": [{"lon": 77.2, "lat": 28.6}]}


class VendorTests(unittest.TestCase):
    def setUp(self):
        self.http = patch.object(vendors, "urlopen", side_effect=AssertionError("Unexpected HTTP call")).start()
        patch.object(vendors, "load_dotenv").start()
        patch.dict(os.environ, {"GEOAPIFY_API_KEY": "offline-test-key"}).start()
        self.output = patch("builtins.print").start()
        self.addCleanup(patch.stopall)

    def recommend(self, categories, responses):
        self.http.side_effect = responses
        strategy = {"categories": categories,
                    "omissions": [{"service": "DJ"}],
                    "alternatives": [{"approach": "Separate venue and caterer"}]}
        before = deepcopy(strategy)
        result = vendors.recommend_vendors("Delhi", strategy)
        self.assertEqual(strategy, before)
        return result

    def test_bundle_one_restaurant_search(self):
        result = self.recommend([category(["restaurant"])], [body(GEO), body({"features": [feature()]})])
        self.assertEqual(result[0]["status"], "ok")
        self.assertEqual(result[0]["covers"], ["venue", "food", "music"])
        self.assertEqual(self.http.call_count, 2)  # One geocode and one place search.
        query = parse_qs(urlparse(self.http.call_args.args[0]).query)
        self.assertEqual(query["categories"], ["catering.restaurant"])
        self.assertEqual(query["limit"], ["5"])

    def test_all_four_verified_mappings(self):
        for kind in ("restaurant", "cafe", "bar", "pub"):
            with self.subTest(kind=kind):
                self.recommend([category([kind])], [body(GEO), body({"features": []})])
                query = parse_qs(urlparse(self.http.call_args.args[0]).query)
                self.assertEqual(query["categories"], ["catering." + kind])

    def test_no_types_no_requests(self):
        result = self.recommend([category([])], [])
        self.assertEqual(result[0]["status"], "not_required")
        self.http.assert_not_called()

    def test_unsupported_no_guessing(self):
        result = self.recommend([category(["photographer"])], [])
        self.assertEqual(result[0]["status"], "unsupported")
        self.assertEqual(result[0]["unsupported_types"], ["photographer"])
        self.http.assert_not_called()

    def test_mixed_supported_and_unsupported(self):
        result = self.recommend([category(["restaurant", "DJ"])], [body(GEO), body({"features": []})])
        self.assertEqual(result[0]["unsupported_types"], ["DJ"])
        query = parse_qs(urlparse(self.http.call_args.args[0]).query)
        self.assertEqual(query["categories"], ["catering.restaurant"])

    def test_duplicate_intents_use_one_search(self):
        result = self.recommend([category(["restaurant", "cafe"]), category(["cafe", "restaurant", "cafe"], "Other")],
                                [body(GEO), body({"features": [feature()]})])
        self.assertEqual(self.http.call_count, 2)
        self.assertEqual([g["status"] for g in result], ["ok", "ok"])

    def test_missing_key(self):
        with patch.dict(os.environ, {"GEOAPIFY_API_KEY": ""}):
            result = self.recommend([category(["restaurant"]), category([])], [])
        self.assertEqual([g["status"] for g in result], ["unavailable", "not_required"])
        self.http.assert_not_called()

    def test_partial_failure_keeps_success_and_geocodes_once(self):
        result = self.recommend([category(["restaurant"]), category(["cafe"], "Cafe")],
                                [body(GEO), body({"features": [feature()]}), URLError("offline-test-key")])
        self.assertEqual([g["status"] for g in result], ["ok", "unavailable"])
        self.assertEqual(self.http.call_count, 3)
        self.assertNotIn("offline-test-key", str(self.output.call_args_list))

    def test_failures_are_not_empty(self):
        failures = [URLError("offline-test-key"), TimeoutError(),
                    HTTPError("https://example.invalid", 429, "Rate limited", None, None),
                    BytesIO(b"bad json"), body({}), body({"features": [None]})]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                result = self.recommend([category(["restaurant"])], [body(GEO), failure])
                self.assertEqual(result[0]["status"], "unavailable")
        result = self.recommend([category(["restaurant"])], [body(GEO), body({"features": []})])
        self.assertEqual(result[0]["status"], "empty")

    def test_geocoding_failure(self):
        for failed_geo in (body({"results": []}), body({"results": [None]}), URLError("offline")):
            result = self.recommend([category(["restaurant"])], [failed_geo])
            self.assertEqual(result[0]["status"], "unavailable")

    def test_dedup_by_id_not_name(self):
        result = self.recommend([category(["restaurant"])], [body(GEO), body({"features": [
            feature("one"), feature("one"), feature("two"), feature(None), feature(None),
        ]})])
        self.assertEqual(len(result[0]["places"]), 4)

    def test_no_invented_fields_or_contacts(self):
        record = feature()
        record["properties"].update(price=123, rating=5, availability="yes")
        result = self.recommend([category(["restaurant"])], [body(GEO), body({"features": [record]})])
        place = result[0]["places"][0]
        self.assertIsNone(place["phone"])
        self.assertIsNone(place["website"])
        self.assertEqual(set(place), {"name", "address", "city", "categories", "latitude", "longitude", "website", "phone", "source"})

    def test_public_search_compatibility(self):
        self.http.side_effect = [body(GEO), body({"features": [feature()]})]
        self.assertEqual(len(vendors.search_vendors("Delhi", ["catering.restaurant"])), 1)
        self.http.side_effect = [body(GEO), URLError("offline")]
        self.assertEqual(vendors.search_vendors("Delhi", ["catering.restaurant"]), [])

    def test_main_invocation_after_timeline_and_failure_isolation(self):
        # Execute only the existing post-plan branch with all stages stubbed.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
        branch = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "event_plan")
        code = compile(ast.Module(body=[branch], type_ignores=[]), "<post-plan>", "exec")
        plan = {"event_summary": "Fixture", "categories": [], "total_estimated_cost": 0, "remaining_budget": 100}
        budget = {"budget_strategy": "Fixture", "allocations": [], "total_allocated": 0, "remaining_budget": 100}
        for timeline in (None, {"timeline_summary": "Fixture", "tasks": []}):
            order = []
            discovery = Mock(side_effect=lambda *a: order.append("vendors") or [])
            scope = {"event_plan": plan, "event_state": {"location": "Delhi"}, "strategy": {}, "client": None, "MODEL": None,
                     "optimize_budget": Mock(return_value=budget),
                     "generate_timeline": Mock(side_effect=lambda **kw: order.append("timeline") or timeline),
                     "recommend_vendors": discovery}
            exec(code, scope)
            self.assertEqual(order, ["timeline", "vendors"])
            scope["recommend_vendors"] = Mock(side_effect=RuntimeError("Provider failed"))
            exec(code, scope)  # Must not raise or mutate planning output.
            self.assertEqual(scope["event_plan"], plan)
            scope["optimize_budget"].return_value = None
            scope["recommend_vendors"].reset_mock()
            exec(code, scope)
            scope["recommend_vendors"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
