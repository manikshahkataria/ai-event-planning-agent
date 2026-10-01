"""Offline transactional replanning checks. Every AI/provider boundary is mocked."""

from copy import deepcopy
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main
from planner import replanning as rp
from planner.requirements import extract_requirement_changes
from planner.validation import parse_event_date, validate_requirements


REQUIREMENTS = {"event_type": "corporate gathering", "location": "Delhi",
                "date": "2026-11-15", "guest_count": 20, "budget": 25000,
                "preferences": ["luxury dining", "music"]}
STRATEGY = {
    "approach": "Dining reservation", "rationale": "Food and space together",
    "priorities": ["luxury dining", "music"],
    "categories": [{"category": "Dining", "covers": ["venue", "food"],
                    "reason": "Bundle", "vendor_types": ["restaurant"]}],
    "omissions": [], "alternatives": [], "assumptions": [],
}
READY = {"status": "ready", "issues": [], "strategy": STRATEGY}


def change(**updates):
    return {"updates": [{"field": k, "value": v, "source": str(v)} for k, v in updates.items()],
            "preference_changes": [], "issues": []}


def preferences(action, target=None, value=None):
    result = change()
    result["preference_changes"] = [{"action": action, "target": target, "value": value}]
    return result


def completion(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])


def accepted():
    return deepcopy({"revision": 1, "requirements": REQUIREMENTS, "strategy": STRATEGY,
        "event_plan": {"event_summary": "Old plan"}, "budget_plan": {"budget_strategy": "Old budget"},
        "timeline": {"timeline_summary": "Old timeline"},
        "vendor_groups": [{"category": "Old Delhi restaurant", "places": [{"city": "Delhi"}]}],
        "vendor_cache": {},
        "component_statuses": {"strategy": "ready", "event_plan": "ready", "budget_plan": "ready",
                               "timeline": "ready", "vendors": "ready"}})


class ReplanningTests(unittest.TestCase):
    def setUp(self):
        self.before = accepted()
        self.session = deepcopy(self.before)
        self.stages = {}
        for name, value in (("assess_event_strategy", READY), ("generate_event_plan", {"event_summary": "New plan"}),
                            ("optimize_budget", {"budget_strategy": "New budget"}),
                            ("generate_timeline", {"timeline_summary": "New timeline"}), ("recommend_vendors", [])):
            self.stages[name] = patch.object(rp, name, return_value=deepcopy(value)).start()
        patch("builtins.print").start()
        self.addCleanup(patch.stopall)

    def run_change(self, proposal):
        result = rp.replan_event(None, "test-model", self.session, proposal, "Explicit edit")
        self.assertEqual(self.session, self.before)
        return result

    def test_each_scalar_field_regenerates_core_and_timeline_once(self):
        for field, value in (("guest_count", 35), ("budget", 20000), ("location", "Noida"),
                             ("date", "2026-12-15"), ("event_type", "birthday")):
            with self.subTest(field=field):
                for mock in self.stages.values():
                    mock.reset_mock()
                result = self.run_change(change(**{field: value}))
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["session"]["revision"], 2)
                self.assertEqual(result["changes"], {field: {"old": REQUIREMENTS[field], "new": value}})
                for mock in self.stages.values():
                    mock.assert_called_once()
                strategy_args = self.stages["assess_event_strategy"].call_args.kwargs
                self.assertEqual(strategy_args["previous_strategy"], STRATEGY)
                self.assertEqual(strategy_args["changes"], result["changes"])
                self.assertEqual(result["impact"]["vendors"], "refresh" if field == "location" else "check_search_inputs")

    def test_multiple_updates_preserve_unmentioned_fields(self):
        result = self.run_change(change(guest_count=35, budget=20000))
        self.assertEqual(set(result["changes"]), {"guest_count", "budget"})
        self.assertEqual(result["session"]["requirements"], {**REQUIREMENTS, "guest_count": 35, "budget": 20000})

    def test_each_preference_operation(self):
        cases = [(preferences("add", value="photography"), ["luxury dining", "music", "photography"]),
                 (preferences("remove", target="music"), ["luxury dining"]),
                 (preferences("replace", target="luxury dining", value="simple buffet"), ["simple buffet", "music"]),
                 (preferences("clear"), [])]
        for proposal, expected in cases:
            with self.subTest(proposal=proposal):
                result = self.run_change(proposal)
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["session"]["requirements"]["preferences"], expected)

    def test_noop_and_equivalent_representations_skip_all_stages(self):
        for proposal in (change(), change(guest_count=20, budget=25000.0), change(date="15 November 2026"),
                         change(location=" Delhi "), preferences("add", value="  luxury   dining ")):
            with self.subTest(proposal=proposal):
                self.assertEqual(self.run_change(proposal)["status"], "noop")
        for mock in self.stages.values():
            mock.assert_not_called()
        equivalent = {**REQUIREMENTS, "guest_count": 20.0, "preferences": ["music", "luxury dining", "music"]}
        self.assertEqual(rp.compare_requirements(REQUIREMENTS, equivalent), {})

    def test_invalid_values_and_mixed_patches_block_all_downstream(self):
        cases = [change(date=value) for value in ("41 November", "31 February 2026", "29 February 2025", "03/04/2026")]
        cases += [change(guest_count=value) for value in (0, -5, 2.5, True, "lots", 20.0)]
        cases += [change(budget=value) for value in (-100, float("inf"), float("nan"), True)]
        cases += [change(budget=20000, date="41 November"), change(location=" ")]
        for proposal in cases:
            with self.subTest(proposal=proposal):
                result = self.run_change(proposal)
                self.assertEqual(result["status"], "needs_clarification")
                self.assertIsNone(result["session"])
                self.assertTrue(any(i["blocking"] for i in result["draft"]["issues"]))
        for mock in self.stages.values():
            mock.assert_not_called()

    def test_source_date_cannot_be_silently_repaired(self):
        proposal = change(date="2026-11-30")
        proposal["updates"][0]["source"] = "41 November"
        self.assertEqual(self.run_change(proposal)["status"], "needs_clarification")
        self.stages["assess_event_strategy"].assert_not_called()

    def test_absent_ambiguous_and_substring_preference_targets_clarify(self):
        for target in ("photography", "luxury", "Music"):
            self.assertEqual(self.run_change(preferences("remove", target=target))["status"], "needs_clarification")
        state = {**REQUIREMENTS, "preferences": ["music", " music "]}
        draft = rp.apply_change_patch(state, preferences("replace", "music", "DJ"))
        self.assertTrue(draft["issues"])
        self.assertEqual(state["preferences"], ["music", " music "])

    def test_malformed_duplicate_and_unresolved_operations_block(self):
        duplicate = change(budget=20000)
        duplicate["updates"] *= 2
        unclear = change()
        unclear["issues"] = [{"field": "preferences", "reason": "Which music?", "question": "Which preference?", "blocking": True}]
        for proposal in ({}, None, duplicate, unclear, preferences("clear", target="music")):
            self.assertEqual(self.run_change(proposal)["status"], "needs_clarification")
        for mock in self.stages.values():
            mock.assert_not_called()

    def test_core_failure_preserves_accepted_revision_and_stops_later_stages(self):
        order = list(self.stages)
        for stage in order[:3]:
            for failure in (None, RuntimeError("offline failure")):
                with self.subTest(stage=stage, failure=failure):
                    for mock in self.stages.values():
                        mock.reset_mock()
                    mock = self.stages[stage]
                    saved = mock.return_value
                    if isinstance(failure, Exception):
                        mock.side_effect = failure
                    else:
                        mock.return_value = None
                    self.assertEqual(self.run_change(change(budget=20000))["status"], "failed")
                    for downstream in order[order.index(stage) + 1:]:
                        self.stages[downstream].assert_not_called()
                    mock.side_effect = None
                    mock.return_value = saved

    def test_contextual_clarification_preserves_accepted_and_blocks_generation(self):
        self.stages["assess_event_strategy"].return_value = {"status": "needs_clarification", "strategy": None,
            "issues": [{"fields": ["budget"], "reason": "Conflict", "question": "Simpler dining?", "blocking": True}]}
        result = self.run_change(change(budget=5000, guest_count=500))
        self.assertEqual(result["status"], "needs_clarification")
        for name in list(self.stages)[1:]:
            self.stages[name].assert_not_called()

    def test_strategy_changes_propagate_to_plan_budget_and_vendor_intent(self):
        new = deepcopy(STRATEGY)
        new["approach"] = "Cafe gathering"
        new["categories"][0].update(category="Cafe", vendor_types=["cafe"])
        self.stages["assess_event_strategy"].return_value = {**READY, "strategy": new}
        result = self.run_change(change(budget=20000))
        self.assertTrue(result["strategy_changes"]["material"])
        self.assertTrue(result["strategy_changes"]["vendor_intent_changed"])
        for name in ("generate_event_plan", "optimize_budget"):
            self.assertEqual(self.stages[name].call_args.kwargs["strategy"], new)
        self.assertEqual(self.stages["recommend_vendors"].call_args.args[1], new)
        self.assertEqual(self.stages["generate_timeline"].call_args.kwargs["budget_plan"], result["session"]["budget_plan"])

    def test_label_change_does_not_change_vendor_intent(self):
        new = deepcopy(STRATEGY)
        new["categories"][0]["category"] = "Restaurant bundle"
        self.assertFalse(rp.compare_strategy(STRATEGY, new)["vendor_intent_changed"])

    def test_optional_failures_do_not_reuse_stale_outputs(self):
        self.stages["generate_timeline"].return_value = None
        self.stages["recommend_vendors"].side_effect = RuntimeError("unavailable")
        result = self.run_change(change(location="Noida"))
        self.assertEqual(result["status"], "ready")
        new = result["session"]
        self.assertIsNone(new["timeline"])
        self.assertEqual(new["vendor_groups"], [])
        self.assertEqual(new["component_statuses"]["timeline"], "unavailable")
        self.assertEqual(new["component_statuses"]["vendors"], "unavailable")
        self.assertNotEqual(new["event_plan"], self.before["event_plan"])

    def test_malformed_vendor_results_do_not_discard_core(self):
        for value in (None, {}, [None]):
            with self.subTest(value=value):
                self.stages["recommend_vendors"].return_value = value
                result = self.run_change(change(location="Noida"))
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["session"]["vendor_groups"], [])
                self.assertEqual(result["session"]["component_statuses"]["vendors"], "unavailable")

    def test_accepted_and_draft_inputs_isolated_from_stage_mutations(self):
        def fail(**kw):
            kw["event_state"]["preferences"].clear()
            kw["strategy"]["categories"].clear()
            raise RuntimeError("failed after mutation")
        self.stages["generate_event_plan"].side_effect = fail
        result = self.run_change(change(budget=20000))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["draft"]["requirements"]["preferences"], REQUIREMENTS["preferences"])

    def test_summary_reports_actual_reuse_and_optional_failures(self):
        self.stages["recommend_vendors"].return_value = [{"status": "ok", "search_action": "reused"}]
        self.stages["generate_timeline"].return_value = None
        summary = rp.build_change_summary(self.run_change(change(guest_count=35, budget=20000.5)))
        self.assertIn("Guest count: 20 → 35", summary)
        self.assertIn("₹25,000 → ₹20,000.5", summary)
        self.assertIn("search inputs unchanged", summary)
        self.assertIn("Timeline unavailable", summary)
        self.assertNotIn("New searches", summary)

    def test_repeated_cli_revisions_use_latest_accepted_state(self):
        with patch("builtins.input", side_effect=["35 guests", "budget 20000", "done"]), \
                patch("planner.controller.extract_requirement_changes", side_effect=[change(guest_count=35), change(budget=20000)]) as extract, \
                patch("main.display_session") as display:
            final = main.handle_changes(None, None, self.session)
        self.assertEqual(final["revision"], 3)
        self.assertEqual(final["requirements"]["guest_count"], 35)
        self.assertEqual(final["requirements"]["budget"], 20000)
        self.assertEqual(extract.call_args_list[1].args[2]["guest_count"], 35)
        self.assertEqual(display.call_count, 2)
        self.assertEqual(self.session, self.before)

    def test_mixed_patch_clarification_keeps_all_changes_pending(self):
        proposals = [change(guest_count=35, date="41 November"), change(guest_count=35, date="2026-12-01")]
        with patch("builtins.input", side_effect=["35 guests and 41 November", "2026-12-01", "done"]), \
                patch("planner.controller.extract_requirement_changes", side_effect=proposals) as extract, \
                patch("main.display_session"):
            final = main.handle_changes(None, None, self.session)
        self.assertEqual(final["revision"], 2)
        self.assertEqual(final["requirements"]["guest_count"], 35)
        self.assertEqual(final["requirements"]["date"], "2026-12-01")
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(extract.call_args_list[1].args[2], REQUIREMENTS)
        self.assertIn("2026-12-01", extract.call_args_list[1].args[4])
        self.stages["assess_event_strategy"].assert_called_once()
        self.assertEqual(self.session, self.before)

    def test_cli_cancel_exit_and_interpretation_failure_preserve_session(self):
        for answer in ("cancel", "quit", "exit", "done"):
            with patch("builtins.input", side_effect=["bad date", answer, "done"]), \
                    patch("planner.controller.extract_requirement_changes", return_value=change(date="41 November")):
                self.assertIs(main.handle_changes(None, None, self.session), self.session)
        with patch("builtins.input", side_effect=["update", "done"]), \
                patch("planner.controller.extract_requirement_changes", return_value=None):
            self.assertIs(main.handle_changes(None, None, self.session), self.session)
        self.assertEqual(self.session, self.before)
        for mock in self.stages.values():
            mock.assert_not_called()

    def test_import_main_does_not_initialize_client_or_prompt(self):
        with patch("planner.llm_client.create_llm_client") as factory, patch("builtins.input") as ask:
            importlib.reload(main)
        factory.assert_not_called()
        ask.assert_not_called()

    def test_initial_cli_planning_then_change_and_exit_offline(self):
        client = Mock()
        manager = Mock()
        manager.__enter__ = Mock(return_value=client)
        manager.__exit__ = Mock(return_value=False)
        with patch("main.create_llm_client", return_value=manager), \
                patch("main.get_llm_model", return_value="offline-model"), \
                patch("planner.controller.extract_requirements_result", return_value={"status": "success", "requirements": deepcopy(REQUIREMENTS), "error": None}), \
                patch("planner.controller.assess_event_strategy", return_value=deepcopy(READY)), \
                patch("planner.controller.extract_requirement_changes", return_value=change(guest_count=35)), \
                patch("builtins.input", side_effect=["initial event", "35 guests", "done"]), \
                patch("main.display_session") as display:
            main.main()
        self.assertEqual(display.call_count, 2)
        self.assertEqual(display.call_args_list[0].args[0]["revision"], 1)
        self.assertEqual(display.call_args_list[1].args[0]["revision"], 2)
        self.assertEqual(display.call_args_list[0].args[0]["requirements"]["guest_count"], 20)
        self.assertEqual(display.call_args_list[1].args[0]["requirements"]["guest_count"], 35)
        manager.__exit__.assert_called_once()


class ChangeExtractionTests(unittest.TestCase):
    def test_one_structured_request_preserves_accepted_inputs(self):
        state = deepcopy(REQUIREMENTS)
        expected = change(guest_count=35, budget=20000)
        with patch("planner.requirements.create_reliable_completion", return_value=completion(expected)) as call:
            result = extract_requirement_changes(None, None, state, "35 guests; budget 20000", "Correction")
        call.assert_called_once()
        self.assertEqual(result, expected)
        self.assertEqual(state, REQUIREMENTS)
        args = call.call_args.kwargs
        self.assertTrue(args["response_format"]["json_schema"]["strict"])
        context = json.loads(args["messages"][1]["content"])
        self.assertEqual(context["accepted_requirements"], REQUIREMENTS)
        self.assertEqual(context["clarification_context"], "Correction")

    def test_service_and_malformed_failures_are_distinct_from_noop(self):
        for value in (None, completion({}), completion([]), SimpleNamespace(choices=[])):
            with patch("planner.requirements.create_reliable_completion", return_value=value), patch("builtins.print"):
                self.assertIsNone(extract_requirement_changes(None, None, REQUIREMENTS, "edit"))
        with patch("planner.requirements.create_reliable_completion", return_value=completion(change())):
            self.assertEqual(extract_requirement_changes(None, None, REQUIREMENTS, "no change"), change())


class DateValidationRegressionTests(unittest.TestCase):
    def test_valid_and_invalid_dates(self):
        for value in ("41 November 2026", "31 February 2026", "29 February 2025", "03/04/2026", "41 November"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_event_date(value)
        for value in ("15 November 2026", "29 February 2028", "November 15, 2026", "15/11/2026"):
            self.assertFalse(validate_requirements({**REQUIREMENTS, "date": value}))
        self.assertEqual(parse_event_date("2026-11-15"), parse_event_date("15 November 2026"))

    def test_budget_zero_remains_valid(self):
        self.assertEqual(validate_requirements({**REQUIREMENTS, "budget": 0}), [])


if __name__ == "__main__":
    unittest.main()
