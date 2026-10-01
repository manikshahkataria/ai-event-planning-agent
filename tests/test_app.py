"""Streamlit interaction tests with no browser server or live provider calls."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from planner import controller


APP = Path(__file__).resolve().parents[1] / "app.py"


def session():
    return {
        "revision": 1,
        "requirements": {"event_type": "Birthday", "location": "Delhi", "date": "2026-11-15",
                         "guest_count": 20, "budget": 25000, "preferences": ["Music"]},
        "strategy": {"approach": "Dining reservation", "rationale": "Food and venue together",
                     "priorities": ["Music"], "categories": [{"category": "Dining", "covers": ["food", "venue"],
                     "reason": "Bundle", "vendor_types": ["restaurant"]}],
                     "assumptions": ["Dining space included"], "omissions": [{"service": "Photography", "reason": "Not requested"}],
                     "alternatives": [{"approach": "Private room", "tradeoff": "More coordination"}]},
        "event_plan": {"event_summary": "Birthday dinner", "categories": [{"category": "Dining", "recommendation": "Shared dining",
                       "estimated_budget": 22000}], "total_estimated_cost": 22000, "remaining_budget": 3000, "planning_notes": []},
        "budget_plan": {"budget_strategy": "Reserve contingency", "allocations": [{"category": "Dining", "priority": "high",
                        "allocated_budget": 22000, "reason": "Main experience"}], "total_allocated": 22000,
                        "remaining_budget": 3000, "tradeoffs": ["Keep decorations simple"]},
        "timeline": {"timeline_summary": "Prepare the dinner", "tasks": [
            {"task_id": 1, "task": "Choose format", "category": "Dining", "priority": "high", "due_date": "2026-10-15",
             "dependencies": [], "reason": "Confirm scope"},
            {"task_id": 2, "task": "Send invitations", "category": "Dining", "priority": "medium", "due_date": "2026-10-20",
             "dependencies": [1], "reason": "Confirm attendance"}], "critical_tasks": [1], "execution_notes": []},
        "vendor_groups": [{"category": "Dining", "covers": ["food", "venue"], "unsupported_types": [], "status": "ok",
                           "search_action": "reused", "places": [{"name": "Fixture Restaurant", "address": "Fixture address",
                           "website": "https://example.com", "phone": "12345", "source": "Geoapify"}]}],
        "vendor_cache": {"searches": {}},
        "component_statuses": {"strategy": "ready", "event_plan": "ready", "budget_plan": "ready", "timeline": "ready", "vendors": "ready"},
    }


def ready():
    state = controller.new_workflow(session())
    state["outcome"] = "ready"
    return state


def clarification(state=None, kind="initial"):
    state = deepcopy(state) if state else controller.new_workflow()
    state.update(phase="NEEDS_CLARIFICATION", outcome="clarification", pending={
        "kind": kind, "request": "Initial request", "context": "Initial request",
        "base_revision": state["accepted"]["revision"] if state["accepted"] else None,
        "question": {"question": "What date did you intend?", "reason": "The date needs clarification.",
                     "kind": "validation", "fields": ["date"]}, "issues": [],
    })
    return state


@pytest.fixture
def boundaries():
    client = MagicMock()
    client.__enter__.return_value = client
    with patch("planner.llm_client.create_llm_client", return_value=client) as factory, \
            patch("planner.llm_client.get_llm_model", return_value="offline-model"), \
            patch.object(controller, "start_plan", return_value=ready()) as start, \
            patch.object(controller, "submit_change", return_value=ready()) as change, \
            patch.object(controller, "submit_clarification", return_value=ready()) as clarify, \
            patch("httpx.Client.send", side_effect=AssertionError("Unexpected live AI call")) as ai, \
            patch("planner.vendors.urlopen", side_effect=AssertionError("Unexpected live Geoapify call")) as geo:
        yield {"factory": factory, "start": start, "change": change, "clarify": clarify}
        ai.assert_not_called()
        geo.assert_not_called()


def app(state=None):
    test = AppTest.from_file(str(APP), default_timeout=15)
    if state is not None:
        test.session_state["workflow"] = deepcopy(state)
    test.run()
    assert not test.exception
    return test


def button(test, label):
    return next(item for item in test.button if item.label == label)


def text_content(test):
    return "\n".join(str(item.value) for kind in ("text", "caption", "warning", "info", "error") for item in test.get(kind))


def test_empty_reruns_and_unsubmitted_input_do_not_call_providers(boundaries):
    test = app()
    assert test.title[0].value == "AI Event Planner"
    test.text_area[0].set_value("A birthday in Delhi").run()
    test.run()
    assert test.session_state["workflow"]["accepted"] is None
    for mock in boundaries.values():
        mock.assert_not_called()


def test_create_submits_once_and_rerenders_saved_results(boundaries):
    test = app()
    test.text_area[0].set_value("Birthday in Delhi")
    button(test, "Create Plan").click().run()
    assert not test.exception
    assert test.session_state["workflow"]["accepted"]["revision"] == 1
    assert [tab.label for tab in test.tabs] == ["Strategy", "Plan & Budget", "Timeline", "Real Places"]
    assert len(test.metric) == 4
    assert len(test.dataframe) == 2
    assert "Choose format" in test.dataframe[1].value["Depends on"].tolist()
    test.run()
    test.text_area[0].set_value("Unsubmitted change").run()
    boundaries["start"].assert_called_once()
    boundaries["factory"].assert_called_once()
    boundaries["change"].assert_not_called()


def test_clarification_form_uses_controller_one_answer_at_a_time(boundaries):
    boundaries["start"].return_value = clarification()
    test = app()
    test.text_area[0].set_value("Initial event")
    button(test, "Create Plan").click().run()
    assert "What date did you intend?" in text_content(test)
    test.run()
    boundaries["clarify"].assert_not_called()
    test.text_area[0].set_value("2026-11-15")
    button(test, "Submit Answer").click().run()
    assert not test.exception
    boundaries["clarify"].assert_called_once()
    assert boundaries["clarify"].call_args.args[3] == "2026-11-15"
    assert test.session_state["workflow"]["phase"] == "READY"


def test_revision_summary_and_latest_state_are_saved(boundaries):
    updated = ready()
    updated["accepted"]["revision"] = 2
    updated["accepted"]["requirements"]["guest_count"] = 35
    updated["last_summary"] = "UPDATED\nGuest count: 20 → 35\nREASSESSED\nStrategy\nPRESERVED\nLocation"
    boundaries["change"].return_value = updated
    test = app(ready())
    test.text_area[0].set_value("Increase guests to 35")
    button(test, "Update Plan").click().run()
    assert not test.exception
    assert updated["last_summary"] in text_content(test)
    test.run()
    boundaries["change"].assert_called_once()
    test.text_area[0].set_value("Reduce budget")
    button(test, "Update Plan").click().run()
    assert boundaries["change"].call_args.args[2]["accepted"]["revision"] == 2


def test_failed_revision_preserves_accepted_and_shows_error(boundaries):
    failed = ready()
    failed.update(phase="ERROR", outcome="error", error={"code": "replanning_failed", "message": "Your accepted revision is unchanged."})
    boundaries["change"].return_value = failed
    test = app(ready())
    test.text_area[0].set_value("Change budget")
    button(test, "Update Plan").click().run()
    assert test.session_state["workflow"]["accepted"] == session()
    assert "Your accepted revision is unchanged." in text_content(test)
    test.run()
    boundaries["change"].assert_called_once()


def test_noop_displays_controller_summary_without_render_calls(boundaries):
    result = ready()
    result.update(outcome="noop", last_summary="No effective changes. All requirements and outputs preserved.")
    boundaries["change"].return_value = result
    test = app(ready())
    test.text_area[0].set_value("Keep the budget")
    button(test, "Update Plan").click().run()
    test.run()
    assert result["last_summary"] in text_content(test)
    assert test.session_state["workflow"]["accepted"]["revision"] == 1
    boundaries["change"].assert_called_once()


def test_pending_change_discard_keeps_accepted_without_client(boundaries):
    test = app(clarification(ready(), "change"))
    assert len(test.tabs) == 4
    button(test, "Discard pending request").click().run()
    assert test.session_state["workflow"]["accepted"] == session()
    assert test.session_state["workflow"]["pending"] is None
    assert button(test, "Update Plan")
    boundaries["factory"].assert_not_called()


def test_optional_failures_and_vendor_facts_render_without_calls(boundaries):
    state = ready()
    state["accepted"]["timeline"] = None
    state["accepted"]["component_statuses"].update(timeline="unavailable", vendors="partial")
    state["accepted"]["vendor_groups"].append({"category": "Cafe", "covers": [], "status": "unavailable"})
    test = app(state)
    content = text_content(test)
    assert "timeline is unavailable" in content
    assert "temporarily unavailable" in content
    assert "Fixture Restaurant" in content
    assert "capacity, availability, price, suitability, or budget fit" in content
    assert "Saved place results reused" in content
    assert "Phone: 12345" in content
    assert len(test.get("link_button")) == 1
    assert test.get("link_button")[0].proto.url == "https://example.com"
    test.run()
    assert test.session_state["workflow"] == state
    boundaries["factory"].assert_not_called()


def test_missing_contacts_and_unsafe_urls_never_create_links(boundaries):
    state = ready()
    place = state["accepted"]["vendor_groups"][0]["places"][0]
    place.update(website="javascript:alert(1)", phone=None)
    test = app(state)
    assert not test.get("link_button")
    assert "Phone:" not in text_content(test)
    boundaries["factory"].assert_not_called()


def test_client_failure_is_safe_preserves_plan_and_input(boundaries):
    boundaries["factory"].side_effect = RuntimeError("secret-key-and-headers")
    test = app(ready())
    test.text_area[0].set_value("35 guests")
    button(test, "Update Plan").click().run()
    assert not test.exception
    assert test.session_state["workflow"]["accepted"] == session()
    assert "secret-key-and-headers" not in text_content(test)
    assert test.text_area[0].value == "35 guests"
    assert not test.session_state["busy"]
    test.run()
    boundaries["factory"].assert_called_once()
    boundaries["change"].assert_not_called()


def test_unexpected_mutating_action_cannot_corrupt_accepted(boundaries):
    def fail(client, model, workflow, request, **kwargs):
        workflow["accepted"]["requirements"]["preferences"].clear()
        raise RuntimeError("internal details")
    boundaries["change"].side_effect = fail
    test = app(ready())
    test.text_area[0].set_value("Change music")
    button(test, "Update Plan").click().run()
    assert test.session_state["workflow"]["accepted"] == session()
    assert "internal details" not in text_content(test)


def test_render_failure_does_not_erase_or_regenerate_session(boundaries):
    state = ready()
    del state["accepted"]["budget_plan"]["budget_strategy"]
    test = app(state)
    assert not test.exception
    assert "section could not be displayed" in text_content(test)
    assert test.session_state["workflow"] == state
    boundaries["factory"].assert_not_called()


def test_browser_sessions_are_isolated(boundaries):
    first = app(ready())
    second = app()
    assert second.session_state["workflow"]["accepted"] is None
    assert first.session_state["workflow"]["accepted"] == session()


def test_blank_submit_never_creates_client(boundaries):
    test = app()
    button(test, "Create Plan").click().run()
    assert "Please enter a description" in text_content(test)
    boundaries["factory"].assert_not_called()
    boundaries["start"].assert_not_called()


@pytest.mark.parametrize("url", [None, "", "www.example.com", "javascript:alert(1)", "file:///secret", "https://user:password@example.com", "https://example.com\nother", "https://[broken"])
def test_website_url_rejects_missing_or_unsafe_values(url):
    from app import _website_url
    assert _website_url(url) is None
