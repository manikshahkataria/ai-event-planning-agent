"""Offline action-based controller contracts; no real providers or CLI input."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from planner import controller as ctl, replanning as rp
from planner.requirements import extract_requirements, extract_requirements_result


REQUIREMENTS = {"event_type": "birthday", "location": "Delhi", "date": "2026-11-15",
                "guest_count": 20, "budget": 25000, "preferences": ["music"]}
STRATEGY = {"approach": "Dining", "rationale": "Bundle", "priorities": ["music"],
            "categories": [{"category": "Dining", "covers": ["food", "venue"],
                            "reason": "Bundle", "vendor_types": ["restaurant"]}],
            "omissions": [], "alternatives": [], "assumptions": []}
READY = {"status": "ready", "issues": [], "strategy": STRATEGY}
ISSUE = {"fields": ["preferences"], "reason": "Unclear intent", "question": "Is this a theme?", "blocking": True}
BLOCKED = {"status": "needs_clarification", "issues": [ISSUE], "strategy": None}


def changed(**values):
    return {"updates": [{"field": field, "value": value, "source": str(value)} for field, value in values.items()],
            "preference_changes": [], "issues": []}


def extraction(requirements):
    return {"status": "success", "requirements": deepcopy(requirements), "error": None}


def accepted():
    return deepcopy({"revision": 1, "requirements": REQUIREMENTS, "strategy": STRATEGY,
        "event_plan": {"event_summary": "Old plan"}, "budget_plan": {"budget_strategy": "Old budget"},
        "timeline": {"timeline_summary": "Old timeline"},
        "vendor_groups": [{"status": "ok", "places": [{"city": "Delhi"}]}], "vendor_cache": {},
        "component_statuses": {"strategy": "ready", "event_plan": "ready", "budget_plan": "ready",
                               "timeline": "ready", "vendors": "ready"}})


class ControllerTests(unittest.TestCase):
    def setUp(self):
        patch("builtins.print").start()
        self.extract = patch.object(ctl, "extract_requirements_result", return_value=extraction(REQUIREMENTS)).start()
        self.change = patch.object(ctl, "extract_requirement_changes", return_value=changed(budget=20000)).start()
        self.strategy = patch.object(ctl, "assess_event_strategy", return_value=deepcopy(READY)).start()
        patch.object(rp, "assess_event_strategy", self.strategy).start()
        self.plan = patch.object(rp, "generate_event_plan", return_value={"event_summary": "New plan"}).start()
        self.budget = patch.object(rp, "optimize_budget", return_value={"budget_strategy": "New budget"}).start()
        self.timeline = patch.object(rp, "generate_timeline", return_value={"timeline_summary": "New timeline"}).start()
        self.vendors = patch.object(rp, "recommend_vendors", return_value=[]).start()
        self.workflow = ctl.new_workflow(accepted())
        self.before = deepcopy(self.workflow)
        self.addCleanup(patch.stopall)

    def assert_no_planning(self):
        for mock in (self.strategy, self.plan, self.budget, self.timeline, self.vendors):
            mock.assert_not_called()

    def test_new_workflow_is_detached_and_makes_no_calls(self):
        self.assertEqual(ctl.new_workflow()["phase"], "EMPTY")
        session = accepted()
        state = ctl.new_workflow(session)
        state["accepted"]["requirements"]["preferences"].clear()
        self.assertEqual(session["requirements"]["preferences"], ["music"])
        self.assert_no_planning()
        self.extract.assert_not_called()
        self.change.assert_not_called()

    def test_initial_plan_publishes_complete_revision(self):
        state = ctl.start_plan(None, None, "Birthday in Delhi")
        self.assertEqual(state["phase"], "READY")
        self.assertEqual(state["accepted"]["revision"], 1)
        self.assertEqual(state["accepted"]["requirements"], REQUIREMENTS)
        self.assertIsNone(state["pending"])
        self.assertIsNone(state["error"])
        for mock in (self.extract, self.strategy, self.plan, self.budget, self.timeline, self.vendors):
            mock.assert_called_once()

    def test_missing_initial_requirement_returns_question_then_completes(self):
        self.extract.return_value = extraction({**REQUIREMENTS, "date": None})
        state = ctl.start_plan(None, None, "Birthday")
        before = deepcopy(state)
        self.assertEqual(state["phase"], "NEEDS_CLARIFICATION")
        self.assertEqual(state["pending"]["question"]["fields"], ["date"])
        self.assert_no_planning()
        self.extract.return_value = extraction(REQUIREMENTS)
        result = ctl.submit_clarification(None, None, state, "2026-11-15")
        self.assertEqual(result["phase"], "READY")
        self.assertEqual(state, before)
        self.assertIn("The planner asked:", self.extract.call_args.kwargs["user_message"])
        self.assertIn("2026-11-15", self.strategy.call_args.kwargs["conversation_context"])

    def test_invalid_initial_requirement_is_not_planned(self):
        self.extract.return_value = extraction({**REQUIREMENTS, "guest_count": 0})
        state = ctl.start_plan(None, None, "Zero guests", self.workflow)
        self.assertEqual(state["phase"], "NEEDS_CLARIFICATION")
        self.assertEqual(state["pending"]["question"]["kind"], "validation")
        self.assertEqual(state["accepted"], self.before["accepted"])
        self.assertEqual(self.workflow, self.before)
        self.assert_no_planning()

    def test_context_only_initial_clarification_reassesses_strategy(self):
        self.strategy.side_effect = [deepcopy(BLOCKED), deepcopy(READY)]
        state = ctl.start_plan(None, None, "Mars birthday theme")
        self.assertEqual(state["pending"]["question"]["kind"], "strategy")
        self.plan.assert_not_called()
        result = ctl.submit_clarification(None, None, state, "Yes, a theme on Earth")
        self.assertEqual(result["phase"], "READY")
        self.assertEqual(self.strategy.call_count, 2)
        self.assertIsNone(result["notice"])
        self.assertIn("theme on Earth", self.strategy.call_args.kwargs["conversation_context"])

    def test_unchanged_answer_retains_question_and_notice(self):
        self.extract.return_value = extraction({**REQUIREMENTS, "date": None})
        state = ctl.start_plan(None, None, "Birthday")
        result = ctl.submit_clarification(None, None, state, "not sure")
        self.assertEqual(result["phase"], "NEEDS_CLARIFICATION")
        self.assertIn("couldn't understand", result["notice"])
        self.assert_no_planning()

    def test_initial_extraction_failure_is_error_not_unchanged_success(self):
        self.extract.return_value = {"status": "error", "requirements": REQUIREMENTS, "error": "private provider text"}
        state = ctl.start_plan(None, None, "Birthday", self.workflow)
        self.assertEqual(state["phase"], "ERROR")
        self.assertEqual(state["error"]["code"], "extraction_failed")
        self.assertNotIn("private provider text", str(state["error"]))
        self.assertEqual(state["accepted"], self.before["accepted"])
        self.assert_no_planning()

    def test_followup_extraction_failure_keeps_draft_and_accepted(self):
        self.extract.return_value = extraction({**REQUIREMENTS, "date": None})
        state = ctl.start_plan(None, None, "Birthday", self.workflow)
        self.extract.return_value = {"status": "error", "requirements": REQUIREMENTS}
        result = ctl.submit_clarification(None, None, state, "2026-11-15")
        self.assertEqual(result["phase"], "ERROR")
        self.assertIsNone(result["pending"]["requirements"]["date"])
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.assert_no_planning()

    def test_successful_change_publishes_and_summarizes(self):
        result = ctl.submit_change(None, None, self.workflow, "Reduce budget to 20000")
        self.assertEqual(result["accepted"]["revision"], 2)
        self.assertEqual(result["accepted"]["requirements"]["budget"], 20000)
        self.assertIn("UPDATED", result["last_summary"])
        self.assertEqual(result["outcome"], "ready")
        self.assertEqual(self.workflow, self.before)
        self.change.assert_called_once()

    def test_noop_preserves_revision_and_skips_downstream(self):
        self.change.return_value = changed(budget=25000.0)
        result = ctl.submit_change(None, None, self.workflow, "Keep 25000")
        self.assertEqual(result["outcome"], "noop")
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.assertIn("No effective changes", result["last_summary"])
        self.assert_no_planning()

    def test_mixed_patch_clarifies_as_whole_then_publishes(self):
        self.change.side_effect = [changed(budget=20000, date="41 November"),
                                   changed(budget=20000, date="2026-12-01")]
        request = "Budget 20000 and date 41 November"
        state = ctl.submit_change(None, None, self.workflow, request)
        self.assertEqual(state["accepted"], self.before["accepted"])
        self.assertEqual(state["pending"]["draft"]["requirements"]["budget"], 20000)
        self.assert_no_planning()
        result = ctl.submit_clarification(None, None, state, "2026-12-01")
        self.assertEqual(result["accepted"]["requirements"]["budget"], 20000)
        self.assertEqual(result["accepted"]["requirements"]["date"], "2026-12-01")
        self.assertEqual(self.change.call_args.args[2], REQUIREMENTS)
        self.assertEqual(self.change.call_args.args[3], request)
        self.assertIn("2026-12-01", self.change.call_args.args[4])
        self.strategy.assert_called_once()

    def test_change_contextual_clarification_can_resolve_without_new_patch(self):
        self.strategy.side_effect = [deepcopy(BLOCKED), deepcopy(READY)]
        state = ctl.submit_change(None, None, self.workflow, "Change budget")
        self.assertEqual(state["pending"]["question"]["fields"], ["preferences"])
        self.plan.assert_not_called()
        result = ctl.submit_clarification(None, None, state, "Yes, a theme")
        self.assertEqual(result["phase"], "READY")
        self.assertEqual(self.strategy.call_count, 2)
        self.plan.assert_called_once()

    def test_core_failures_never_publish(self):
        for stage in (self.strategy, self.plan, self.budget):
            saved = stage.return_value
            for failure in (None, RuntimeError("secret-provider-error")):
                with self.subTest(stage=stage, failure=failure):
                    stage.return_value = None
                    stage.side_effect = failure
                    result = ctl.submit_change(None, None, self.workflow, "Change budget")
                    self.assertEqual(result["phase"], "ERROR")
                    self.assertEqual(result["accepted"], self.before["accepted"])
                    self.assertNotIn("secret-provider-error", str(result["error"]))
                    self.assertEqual(self.workflow, self.before)
            stage.side_effect = None
            stage.return_value = saved

    def test_initial_core_failure_preserves_previous_event(self):
        self.budget.return_value = None
        result = ctl.start_plan(None, None, "A different event", self.workflow)
        self.assertEqual(result["phase"], "ERROR")
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.timeline.assert_not_called()
        self.vendors.assert_not_called()

    def test_change_extraction_failure_preserves_previous_summary(self):
        self.workflow["last_summary"] = "Previous summary"
        self.change.return_value = None
        result = ctl.submit_change(None, None, self.workflow, "Change budget")
        self.assertEqual(result["error"]["code"], "extraction_failed")
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.assertEqual(result["last_summary"], "Previous summary")
        self.assert_no_planning()

    def test_unexpected_mutating_extractor_cannot_change_accepted(self):
        def fail(client, model, requirements, *args):
            requirements["preferences"].clear()
            raise RuntimeError("secret")
        self.change.side_effect = fail
        result = ctl.submit_change(None, None, self.workflow, "Edit")
        self.assertEqual(result["phase"], "ERROR")
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.assertEqual(self.workflow, self.before)

    def test_cancel_pending_preserves_accepted_and_calls_no_provider(self):
        self.change.return_value = changed(date="41 November")
        state = ctl.submit_change(None, None, self.workflow, "Bad date")
        before = deepcopy(state)
        self.change.reset_mock()
        result = ctl.cancel_pending(state)
        self.assertEqual(result["phase"], "READY")
        self.assertIsNone(result["pending"])
        self.assertEqual(result["accepted"], self.before["accepted"])
        self.assertEqual(state, before)
        self.change.assert_not_called()
        self.assert_no_planning()
        self.assertEqual(ctl.cancel_pending(ctl.new_workflow())["phase"], "EMPTY")

    def test_repeated_revisions_use_latest_accepted_requirements(self):
        self.change.side_effect = [changed(guest_count=35), changed(budget=20000)]
        second = ctl.submit_change(None, None, self.workflow, "35 guests")
        third = ctl.submit_change(None, None, second, "20000 budget")
        self.assertEqual(third["accepted"]["revision"], 3)
        self.assertEqual(third["accepted"]["requirements"]["guest_count"], 35)
        self.assertEqual(self.change.call_args.args[2]["guest_count"], 35)
        self.assertEqual(second["accepted"]["requirements"]["budget"], 25000)

    def test_optional_failures_do_not_restore_old_outputs(self):
        self.timeline.side_effect = RuntimeError("offline")
        self.vendors.side_effect = RuntimeError("offline")
        self.change.return_value = changed(location="Noida")
        result = ctl.submit_change(None, None, self.workflow, "Move to Noida")
        self.assertEqual(result["phase"], "READY")
        session = result["accepted"]
        self.assertEqual(session["requirements"]["location"], "Noida")
        self.assertIsNone(session["timeline"])
        self.assertEqual(session["vendor_groups"], [])
        self.assertEqual(session["component_statuses"]["timeline"], "unavailable")
        self.assertEqual(session["component_statuses"]["vendors"], "unavailable")

    def test_vendor_cache_passes_through_without_mutating_old_session(self):
        self.workflow["accepted"]["vendor_cache"] = {"searches": {"fixture": {"status": "ok"}}}
        before = deepcopy(self.workflow)
        def discover(*args, cache):
            self.assertEqual(cache, before["accepted"]["vendor_cache"])
            cache["new"] = "cache update"
            return [{"status": "ok", "search_action": "reused"}]
        self.vendors.side_effect = discover
        result = ctl.submit_change(None, None, self.workflow, "Change budget")
        self.assertEqual(self.workflow, before)
        self.assertIn("search inputs unchanged", result["last_summary"])
        self.assertIn("new", result["accepted"]["vendor_cache"])

    def test_stale_clarification_is_rejected_without_calls(self):
        self.change.return_value = changed(date="41 November")
        state = ctl.submit_change(None, None, self.workflow, "Bad date")
        state["accepted"]["revision"] = 2
        self.change.reset_mock()
        result = ctl.submit_clarification(None, None, state, "2026-12-01")
        self.assertEqual(result["error"]["code"], "stale_draft")
        self.change.assert_not_called()
        self.assert_no_planning()

    def test_actions_without_required_state_or_text_do_not_call_providers(self):
        for result in (ctl.start_plan(None, None, " "),
                       ctl.submit_change(None, None, ctl.new_workflow(), "edit"),
                       ctl.submit_change(None, None, self.workflow, " "),
                       ctl.submit_clarification(None, None, self.workflow, "answer")):
            self.assertEqual(result["phase"], "ERROR")
        self.extract.assert_not_called()
        self.change.assert_not_called()
        self.assert_no_planning()

    def test_progress_names_follow_actual_stages(self):
        progress = Mock()
        result = ctl.start_plan(None, None, "Birthday", progress=progress)
        self.assertEqual(result["phase"], "READY")
        self.assertEqual([call.args[0] for call in progress.call_args_list],
                         ["requirements", "validation", "strategy", "event_plan", "budget", "timeline", "vendors"])
        progress.reset_mock()
        self.change.return_value = changed()
        ctl.submit_change(None, None, self.workflow, "No change", progress=progress)
        self.assertEqual([call.args[0] for call in progress.call_args_list], ["change_extraction", "validation"])

    def test_progress_callback_failure_cannot_discard_valid_plan(self):
        result = ctl.submit_change(None, None, self.workflow, "Change budget",
                                   progress=Mock(side_effect=RuntimeError("UI failure")))
        self.assertEqual(result["phase"], "READY")
        self.assertEqual(self.workflow, self.before)

    def test_controller_has_no_terminal_or_streamlit_dependencies(self):
        tree = ast.parse(Path(ctl.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"input", "print"})
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith("streamlit") for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("streamlit"))


class ExtractionResultTests(unittest.TestCase):
    def response(self, value):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])

    def test_unchanged_success_has_explicit_status_and_one_request(self):
        state = deepcopy(REQUIREMENTS)
        with patch("planner.requirements.create_reliable_completion", return_value=self.response(REQUIREMENTS)) as complete, \
                patch("builtins.print"):
            result = extract_requirements_result(None, None, "Same event", state)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["requirements"], state)
        self.assertIsNone(result["error"])
        complete.assert_called_once()

    def test_failures_and_malformed_outputs_return_explicit_error(self):
        for response in (None, self.response({}), self.response([]), self.response(None), SimpleNamespace(choices=[])):
            with self.subTest(response=response), \
                    patch("planner.requirements.create_reliable_completion", return_value=response), patch("builtins.print"):
                state = deepcopy(REQUIREMENTS)
                result = extract_requirements_result(None, None, "Update", state)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["requirements"], REQUIREMENTS)
                self.assertEqual(state, REQUIREMENTS)

    def test_legacy_wrapper_retains_state_and_merge_behavior(self):
        state = deepcopy(REQUIREMENTS)
        with patch("planner.requirements.create_reliable_completion", return_value=None), patch("builtins.print"):
            self.assertIs(extract_requirements(None, None, "Update", state), state)
        proposed = {**dict.fromkeys(REQUIREMENTS), "guest_count": 35, "preferences": ["music", "photography"]}
        with patch("planner.requirements.create_reliable_completion", return_value=self.response(proposed)) as complete, \
                patch("builtins.print"):
            result = extract_requirements(None, None, "35 guests and photography", state)
        self.assertEqual(result, {**REQUIREMENTS, "guest_count": 35, "preferences": ["music", "photography"]})
        self.assertEqual(state, REQUIREMENTS)
        complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
