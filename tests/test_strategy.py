"""Offline checks: no application startup or real completion requests."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from planner.budget import optimize_budget
from planner.event_plan import generate_event_plan
import main
from planner.strategy import assess_event_strategy


STATE = {
    "event_type": "casual corporate gathering", "location": "Delhi",
    "date": "15 November 2026", "guest_count": 20, "budget": 25000,
    "preferences": ["casual"],
}
READY = {
    "status": "ready", "issues": [],
    "strategy": {
        "approach": "A bundled dining reservation",
        "rationale": "A simple format for a small gathering.",
        "priorities": ["casual"],
        "categories": [{
            "category": "Dining reservation", "covers": ["food", "venue"],
            "reason": "One reservation covers the required services.",
            "vendor_types": ["restaurant"],
        }],
        "omissions": [{"service": "Professional photography", "reason": "Not requested."}],
        "alternatives": [{"approach": "Private venue", "tradeoff": "More separate services."}],
        "assumptions": ["The proposed reservation would include dining space."],
    },
}


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(value))
    )])


def clarification(fields, reason, question):
    return {
        "status": "needs_clarification",
        "issues": [{"fields": fields, "reason": reason, "question": question, "blocking": True}],
        "strategy": None,
    }


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.output = patch("builtins.print").start()
        self.addCleanup(patch.stopall)

    def assess(self, value, state=None, context="User description"):
        state = deepcopy(STATE if state is None else state)
        before = deepcopy(state)
        with patch("planner.strategy.create_reliable_completion", return_value=response(value)) as complete:
            result = assess_event_strategy(None, None, state, context)
        complete.assert_called_once()
        self.assertEqual(state, before)
        return result, complete.call_args.kwargs

    def test_normal_ready(self):
        result, args = self.assess(READY)
        self.assertEqual(result, READY)
        self.assertEqual(json.loads(args["messages"][1]["content"])["budget_per_guest"], 1250)
        self.assertTrue(args["response_format"]["json_schema"]["strict"])

    def test_literal_mars_clarification(self):
        issue = clarification(["location"], "Literal physical destination is infeasible.", "Do you mean a theme on Earth?")
        state = {**STATE, "location": "Mars"}
        result, args = self.assess(issue, state, "A physical party on Mars")
        self.assertEqual(result, issue)
        self.assertIn("literal physical", args["messages"][0]["content"])
        self.assertIn("Mars", args["messages"][1]["content"])

    def test_mars_theme_on_earth_ready(self):
        result, args = self.assess(READY, {**STATE, "preferences": ["Mars theme"]})
        self.assertEqual(result["status"], "ready")
        self.assertIn("Mars-themed event on Earth", args["messages"][0]["content"])

    def test_luxury_budget_conflict(self):
        issue = clarification(["budget", "guest_count", "preferences"], "Only 10 per guest for a luxury dinner.", "Which constraint can change?")
        state = {**STATE, "budget": 5000, "guest_count": 500, "preferences": ["luxury dinner"]}
        result, args = self.assess(issue, state)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(json.loads(args["messages"][1]["content"])["budget_per_guest"], 10)

    def test_corporate_bundle(self):
        result, _ = self.assess(READY)
        self.assertEqual(result["strategy"]["categories"][0]["covers"], ["food", "venue"])
        self.assertEqual(len(result["strategy"]["categories"]), 1)

    def test_explicit_photographer_preserved(self):
        ready = deepcopy(READY)
        ready["strategy"]["priorities"].append("professional photographer")
        ready["strategy"]["omissions"] = []
        ready["strategy"]["categories"].append({
            "category": "Photography", "covers": ["professional photographer"],
            "reason": "Explicitly requested.",
            "vendor_types": [],
        })
        result, args = self.assess(ready, {**STATE, "preferences": ["professional photographer"]})
        self.assertIn("Explicitly requested services must be covered", args["messages"][0]["content"])
        self.assertEqual(result, ready)
        self.assertEqual(result["strategy"]["omissions"], [])

    def test_malformed_and_inconsistent_outputs(self):
        bad_values = [None, [], {}, {**READY, "strategy": None}]
        blocked_ready = deepcopy(READY)
        blocked_ready["issues"] = clarification(["budget"], "Conflict", "Change budget?")["issues"]
        bad_values.append(blocked_ready)
        bad_values.append({"status": "needs_clarification", "issues": [], "strategy": None})
        no_question = clarification(["location"], "Ambiguous", " ")
        bad_values.append(no_question)
        for mutate in (
            lambda s: s["strategy"].update(alternatives=[{"approach": "a", "tradeoff": "b"}] * 3),
            lambda s: s["strategy"]["categories"].append(s["strategy"]["categories"][0]),
            lambda s: s["strategy"]["categories"][0].update(covers="food"),
            lambda s: s["strategy"].pop("assumptions"),
        ):
            value = deepcopy(READY)
            mutate(value)
            bad_values.append(value)
        for value in bad_values:
            with self.subTest(value=value):
                self.assertIsNone(self.assess(value)[0])
        with patch("planner.strategy.create_reliable_completion", return_value=SimpleNamespace(choices=[])):
            self.assertIsNone(assess_event_strategy(None, None, STATE))
        invalid_json = response(READY)
        invalid_json.choices[0].message.content = "not json"
        with patch("planner.strategy.create_reliable_completion", return_value=invalid_json):
            self.assertIsNone(assess_event_strategy(None, None, STATE))

    def test_service_none(self):
        with patch("planner.strategy.create_reliable_completion", return_value=None) as complete:
            self.assertIsNone(assess_event_strategy(None, None, STATE))
        complete.assert_called_once()

    def test_vendor_type_vocabulary(self):
        for types in (["restaurant"], ["cafe", "bar", "pub"], []):
            ready = deepcopy(READY)
            ready["strategy"]["categories"][0]["vendor_types"] = types
            self.assertEqual(self.assess(ready)[0], ready)
        for types in (["photographer"], ["catering.restaurant"], ["Business Name"], "restaurant"):
            ready = deepcopy(READY)
            ready["strategy"]["categories"][0]["vendor_types"] = types
            self.assertIsNone(self.assess(ready)[0])
        ready = deepcopy(READY)
        del ready["strategy"]["categories"][0]["vendor_types"]
        self.assertIsNone(self.assess(ready)[0])

    def run_gate(self, assessments, initial=None, corrections=()):
        # Exercise the callable gate without starting the application or a client.
        state = deepcopy(STATE if initial is None else initial)
        downstream = Mock()
        scope = {}
        with patch("planner.strategy.create_reliable_completion", side_effect=assessments) as complete, \
                patch("main.extract_requirements", side_effect=corrections) as extraction, \
                patch("builtins.input", return_value="correction"):
            collected = main.collect_requirements(None, None, state, "Initial description")
        if collected is not None:
            scope["event_state"], scope["strategy"], scope["conversation_context"] = collected
            for stage in ("plan", "budget", "timeline"):
                downstream(stage)
        return scope, complete, extraction, downstream

    def test_gate_blocks_on_malformed_or_unavailable(self):
        for value in (None, response({}), response({**READY, "strategy": None})):
            with self.subTest(value=value):
                _, complete, _, downstream = self.run_gate([value])
                complete.assert_called_once()
                downstream.assert_not_called()

    def test_gate_clarifies_revalidates_then_plans(self):
        issue = clarification(["location"], "Literal Mars", "What Earth location?")
        invalid_correction = {**STATE, "guest_count": 0}
        scope, complete, extraction, downstream = self.run_gate(
            [response(issue), response(READY)], {**STATE, "location": "Mars"},
            [invalid_correction, STATE],
        )
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(extraction.call_count, 2)
        self.assertEqual(downstream.call_count, 3)
        self.assertEqual(scope["strategy"], READY["strategy"])
        self.assertIn("What Earth location?", scope["conversation_context"])
        self.assertIn("positive whole number", scope["conversation_context"])

    def test_context_only_clarification_can_resolve(self):
        issue = clarification(["preferences"], "Unclear intent", "Is this a theme?")
        _, complete, _, downstream = self.run_gate([response(issue), response(READY)], corrections=[STATE])
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(downstream.call_count, 3)
        self.assertFalse(any("couldn't understand" in str(c) for c in self.output.call_args_list))

    def test_planning_and_budget_use_bundle_and_preserve_correction(self):
        strategy = READY["strategy"]
        plan = {
            "event_summary": "Dining event", "categories": [{
                "category": "Dining reservation", "recommendation": "Food and space together",
                "estimated_budget": 30000,
            }], "total_estimated_cost": 30000, "remaining_budget": -5000, "planning_notes": [],
        }
        with patch("planner.event_plan.create_reliable_completion", return_value=response(plan)) as complete:
            result = generate_event_plan(None, None, STATE, strategy)
        self.assertEqual(result["total_estimated_cost"], 25000)
        self.assertIn(json.dumps(strategy, indent=2), complete.call_args.kwargs["messages"][1]["content"])
        budget = {
            "budget_strategy": "Bundle", "allocations": [{
                "category": "Dining reservation", "priority": "high",
                "allocated_budget": 30000, "reason": "Food and venue together",
            }], "total_allocated": 30000, "remaining_budget": -5000, "tradeoffs": [],
        }
        with patch("planner.budget.create_reliable_completion", return_value=response(budget)) as complete:
            allocated = optimize_budget(None, None, STATE, result, strategy)
        self.assertEqual(allocated["total_allocated"], 25000)
        self.assertIn(json.dumps(strategy, indent=2), complete.call_args.kwargs["messages"][1]["content"])
        plan["categories"].append({"category": "Venue", "recommendation": "Duplicate", "estimated_budget": 1})
        with patch("planner.event_plan.create_reliable_completion", return_value=response(plan)):
            self.assertIsNone(generate_event_plan(None, None, STATE, strategy))
        budget["allocations"].append({"category": "Venue", "allocated_budget": 1})
        with patch("planner.budget.create_reliable_completion", return_value=response(budget)):
            self.assertIsNone(optimize_budget(None, None, STATE, result, strategy))


if __name__ == "__main__":
    unittest.main()
