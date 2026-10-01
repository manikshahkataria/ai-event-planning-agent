"""Offline vendor integration tests; all HTTP access is mocked."""

from copy import deepcopy
from io import BytesIO
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from planner import vendors
from planner.replanning import generate_revision
from planner import replanning


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

    def test_cache_reuses_facts_and_regroups_changed_labels(self):
        cache = {}
        strategy = {"categories": [category(["restaurant"])]}
        self.http.side_effect = [body(GEO), body({"features": [feature()]})]
        original = vendors.recommend_vendors("Delhi", strategy, cache=cache)
        strategy["categories"][0].update(category="New label", covers=["food", "space"])
        current = vendors.recommend_vendors("Delhi", strategy, cache=cache)
        self.assertEqual(self.http.call_count, 2)
        self.assertEqual(current[0]["search_action"], "reused")
        self.assertEqual(current[0]["category"], "New label")
        self.assertEqual(current[0]["covers"], ["food", "space"])
        self.assertEqual(current[0]["places"], original[0]["places"])
        current[0]["places"][0]["name"] = "Edited display"
        again = vendors.recommend_vendors("Delhi", strategy, cache=cache)
        self.assertEqual(again[0]["places"][0]["name"], "Fixture Restaurant")

    def test_cache_key_location_type_radius_and_limit(self):
        cache = {}
        strategy = {"categories": [category(["restaurant"])]}
        self.http.side_effect = [body(GEO), body({"features": []}), body(GEO), body({"features": []}),
                                 body({"features": []}), body({"features": []}), body({"features": []})]
        vendors.recommend_vendors("Delhi", strategy, cache=cache)
        vendors.recommend_vendors("Noida", strategy, cache=cache)
        self.assertEqual(self.http.call_count, 4)
        strategy["categories"][0]["vendor_types"] = ["cafe"]
        vendors.recommend_vendors("Noida", strategy, cache=cache)
        self.assertEqual(self.http.call_count, 5)  # Reuse Noida coordinates.
        vendors.recommend_vendors("Noida", strategy, cache=cache, radius_m=5000)
        self.assertEqual(self.http.call_count, 6)
        vendors.recommend_vendors("Noida", strategy, cache=cache, radius_m=5000, limit_per_category=3)
        self.assertEqual(self.http.call_count, 7)
        self.assertTrue(all(key[0] == "Geoapify" for key in cache["searches"]))

    def test_cached_empty_and_failed_searches_do_not_repeat(self):
        for places in (body({"features": []}), URLError("offline")):
            cache = {}
            strategy = {"categories": [category(["restaurant"])]}
            self.http.reset_mock()
            self.http.side_effect = [body(GEO), places]
            first = vendors.recommend_vendors("Delhi", strategy, cache=cache)
            second = vendors.recommend_vendors("Delhi", strategy, cache=cache)
            self.assertEqual(first[0]["status"], second[0]["status"])
            self.assertEqual(second[0]["search_action"], "reused")
            self.assertEqual(self.http.call_count, 2)

    def test_cache_reuses_existing_type_and_searches_only_added_type(self):
        cache = {}
        strategy = {"categories": [category(["restaurant"])]}
        self.http.side_effect = [body(GEO), body({"features": [feature()]}), body({"features": []})]
        vendors.recommend_vendors("Delhi", strategy, cache=cache)
        strategy["categories"].append(category(["cafe"], "Cafe alternative"))
        groups = vendors.recommend_vendors("Delhi", strategy, cache=cache)
        self.assertEqual(self.http.call_count, 3)
        self.assertEqual([g["search_action"] for g in groups], ["reused", "searched"])

    def test_replanning_reuses_vendor_facts_for_guest_budget_date_and_noop(self):
        requirements = {"event_type": "party", "location": "Delhi", "date": "2026-11-15",
                        "guest_count": 20, "budget": 25000, "preferences": []}
        strategy = {"categories": [category(["restaurant"])]}
        self.http.side_effect = [body(GEO), body({"features": [feature()]})]
        with patch.object(replanning, "assess_event_strategy", return_value={"status": "ready", "issues": [], "strategy": strategy}) as assess, \
                patch.object(replanning, "generate_event_plan", return_value={"event_summary": "Plan"}) as plan, \
                patch.object(replanning, "optimize_budget", return_value={"budget_strategy": "Budget"}) as budget, \
                patch.object(replanning, "generate_timeline", return_value={"tasks": []}) as timeline:
            session = generate_revision(None, None, requirements, strategy)
            for field, value in (("guest_count", 35), ("budget", 20000), ("date", "2026-12-01")):
                before = deepcopy(session)
                proposal = {"updates": [{"field": field, "value": value, "source": str(value)}],
                            "preference_changes": [], "issues": []}
                result = replanning.replan_event(None, None, session, proposal)
                self.assertEqual(session, before)
                self.assertEqual(result["status"], "ready")
                session = result["session"]
                self.assertEqual(session["vendor_groups"][0]["search_action"], "reused")
                self.assertEqual(self.http.call_count, 2)
            for mock in (assess, plan, budget, timeline):
                mock.reset_mock()
            self.assertEqual(replanning.replan_event(None, None, session, proposal)["status"], "noop")
            for mock in (assess, plan, budget, timeline):
                mock.assert_not_called()
            self.assertEqual(self.http.call_count, 2)

    def test_location_refresh_failure_never_returns_old_city_facts(self):
        cache = {}
        strategy = {"categories": [category(["restaurant"])]}
        self.http.side_effect = [body(GEO), body({"features": [feature()]}), URLError("offline")]
        old = vendors.recommend_vendors("Delhi", strategy, cache=cache)
        new = vendors.recommend_vendors("Noida", strategy, cache=cache)
        self.assertTrue(old[0]["places"])
        self.assertEqual(new[0]["status"], "unavailable")
        self.assertEqual(new[0]["places"], [])
        self.assertEqual(self.http.call_count, 3)

    def test_main_invocation_after_timeline_and_failure_isolation(self):
        # main delegates the same stage ordering to the revision builder.
        requirements = {"event_type": "party", "location": "Delhi", "date": "2026-11-15",
                        "guest_count": 20, "budget": 100, "preferences": []}
        strategy = {"categories": [category(["restaurant"])]}
        plan = {"event_summary": "Fixture", "categories": [], "total_estimated_cost": 0, "remaining_budget": 100}
        budget = {"budget_strategy": "Fixture", "allocations": [], "total_allocated": 0, "remaining_budget": 100}
        for timeline in (None, {"timeline_summary": "Fixture", "tasks": []}):
            order = []
            with patch("planner.replanning.generate_event_plan", return_value=plan), \
                    patch("planner.replanning.optimize_budget", return_value=budget) as allocate, \
                    patch("planner.replanning.generate_timeline", side_effect=lambda **kw: order.append("timeline") or timeline), \
                    patch("planner.replanning.recommend_vendors", side_effect=lambda *a, **kw: order.append("vendors") or []) as discovery:
                session = generate_revision(None, None, requirements, strategy)
                self.assertEqual(order, ["timeline", "vendors"])
                discovery.side_effect = RuntimeError("Provider failed")
                session = generate_revision(None, None, requirements, strategy)
                self.assertEqual(session["event_plan"], plan)
                self.assertEqual(session["component_statuses"]["vendors"], "unavailable")
                allocate.return_value = None
                discovery.reset_mock()
                self.assertIsNone(generate_revision(None, None, requirements, strategy))
                discovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
