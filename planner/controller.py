"""UI-independent, one-action-at-a-time orchestration for CLI and future UIs.

Actions return detached workflow state; callers retain the returned state between
submissions. Only successful Milestone 6 results replace its accepted session.
No action runs implicitly on reads. Progress callbacks receive stage names only.
"""

from copy import deepcopy

from planner.requirements import (
    REQUIRED_FIELDS, extract_requirements_result, extract_requirement_changes,
    get_missing_fields,
)
from planner.validation import validate_requirements
from planner.strategy import assess_event_strategy
from planner.replanning import (
    generate_revision, replan_event, build_change_summary, notify_progress,
)


QUESTIONS = {
    "event_type": "What type of event are you planning?",
    "location": "Where would you like to organize the event?",
    "date": "What date are you planning the event for?",
    "guest_count": "Approximately how many guests are you expecting?",
    "budget": "What is your total budget for the event?",
}


def new_workflow(accepted=None):
    """Create workflow state without contacting any service."""
    return {"accepted": deepcopy(accepted), "pending": None,
            "phase": "READY" if accepted is not None else "EMPTY",
            "last_summary": None, "error": None, "outcome": None, "notice": None}


def _begin(workflow):
    state = deepcopy(workflow)
    state.update(error=None, notice=None, outcome=None)
    return state


def _error(state, code, message):
    state.update(phase="ERROR", outcome="error", error={"code": code, "message": message})
    return state


def _ask(state, issues, kind):
    issue = next(issue for issue in issues if issue["blocking"])
    state["pending"].update(issues=deepcopy(issues), question={
        "fields": issue.get("fields", [issue.get("field")]),
        "reason": issue["reason"], "question": issue["question"], "kind": kind,
    })
    state.update(phase="NEEDS_CLARIFICATION", outcome="clarification")
    return state


def _revision(state):
    return state["accepted"]["revision"] if state["accepted"] is not None else None


def _advance_initial(client, model, state, message, progress):
    pending = state["pending"]
    notify_progress(progress, "requirements")
    extraction = extract_requirements_result(client=client, model=model, user_message=message,
                                             current_state=deepcopy(pending["requirements"]))
    if extraction["status"] != "success":
        return _error(state, "extraction_failed", "Requirements extraction failed. Please try again; your event is unchanged.")
    requirements = extraction["requirements"]
    previous_question = pending.get("question")
    if (previous_question and previous_question["kind"] != "strategy"
            and requirements == pending["requirements"]):
        state["notice"] = "I couldn't understand that answer. Please try again."
    pending["requirements"] = deepcopy(requirements)
    notify_progress(progress, "validation")
    missing = get_missing_fields(requirements)
    if missing:
        field = missing[0]
        return _ask(state, [{"field": field, "reason": "Required information is missing.",
                            "question": QUESTIONS[field], "blocking": True}], "requirements")
    issues = [issue for issue in validate_requirements(requirements) if issue["blocking"]]
    if issues:
        return _ask(state, issues, "validation")
    notify_progress(progress, "strategy")
    assessment = assess_event_strategy(client=client, model=model, event_state=deepcopy(requirements),
                                       conversation_context=pending["context"])
    if assessment is None:
        return _error(state, "strategy_failed", "Planning stopped because event strategy could not be assessed.")
    if assessment["status"] == "needs_clarification":
        return _ask(state, assessment["issues"], "strategy")
    if assessment["status"] != "ready" or not assessment["strategy"]:
        return _error(state, "strategy_failed", "Event strategy returned an unusable assessment.")
    session = generate_revision(client, model, deepcopy(requirements), deepcopy(assessment["strategy"]),
                                previous=deepcopy(state["accepted"]), progress=progress)
    if session is None:
        return _error(state, "planning_failed", "Unable to complete the event plan and budget. Your accepted event is unchanged.")
    state.update(accepted=session, pending=None, phase="READY", outcome="ready", last_summary=None)
    return state


def _advance_change(client, model, state, progress):
    pending = state["pending"]
    notify_progress(progress, "change_extraction")
    patch = extract_requirement_changes(client, model, deepcopy(state["accepted"]["requirements"]),
                                        pending["request"], pending["context"])
    if patch is None:
        return _error(state, "extraction_failed", "Change interpretation failed. The accepted revision is unchanged.")
    pending["patch"] = deepcopy(patch)
    result = replan_event(client, model, deepcopy(state["accepted"]), patch,
                         conversation_context=pending["request"] + "\n" + pending["context"], progress=progress)
    pending["draft"] = deepcopy(result["draft"])
    if result["status"] == "needs_clarification":
        return _ask(state, result["draft"]["issues"], "change")
    if result["status"] not in ("ready", "noop"):
        return _error(state, "replanning_failed", "Replanning failed. The accepted revision is unchanged; you can retry the change.")
    summary = build_change_summary(result)
    if result["status"] == "ready":
        state["accepted"] = result["session"]
    state.update(pending=None, phase="READY", outcome=result["status"], last_summary=summary)
    return state


def start_plan(client, model, request, workflow=None, *, progress=None):
    """Interpret an initial request and return either a question or a plan.

    Passing an existing workflow also permits proposing a replacement event;
    its accepted revision is retained unless the new core succeeds.
    """
    state = _begin(workflow if workflow is not None else new_workflow())
    if not isinstance(request, str) or not request.strip():
        return _error(state, "empty_request", "Please describe the event you want to plan.")
    state["pending"] = {"kind": "initial", "request": request, "context": request,
                        "requirements": {**dict.fromkeys(REQUIRED_FIELDS), "preferences": []},
                        "base_revision": _revision(state), "question": None, "issues": []}
    try:
        return _advance_initial(client, model, state, request, progress)
    except Exception:
        return _error(state, "planning_failed", "Planning could not be completed. Your accepted event is unchanged.")


def submit_change(client, model, workflow, request, *, progress=None):
    """Interpret an explicit update using existing transactional replanning."""
    state = _begin(workflow)
    if state["accepted"] is None:
        return _error(state, "no_accepted_event", "Create an event plan before requesting changes.")
    if not isinstance(request, str) or not request.strip():
        return _error(state, "empty_request", "Please describe the change you want to make.")
    state["pending"] = {"kind": "change", "request": request, "context": "",
                        "base_revision": _revision(state), "question": None, "issues": []}
    try:
        return _advance_change(client, model, state, progress)
    except Exception:
        return _error(state, "replanning_failed", "Replanning could not be completed. The accepted revision is unchanged.")


def submit_clarification(client, model, workflow, answer, *, progress=None):
    """Consume one answer; never block for input or publish a partial update."""
    state = _begin(workflow)
    pending = state["pending"]
    if not pending or not pending.get("question"):
        return _error(state, "no_question", "There is no clarification question to answer.")
    if pending["base_revision"] != _revision(state):
        return _error(state, "stale_draft", "The event has changed. Please submit your update again.")
    if not isinstance(answer, str) or not answer.strip():
        return _error(state, "empty_answer", "Please answer the clarification question.")
    question = pending["question"]
    message = (f"The planner asked: {question['question']}\nThe user answered: {answer}\n"
               f"Related fields: {', '.join(question['fields'])}")
    pending["context"] += "\n" + message
    try:
        if pending["kind"] == "initial":
            return _advance_initial(client, model, state, message, progress)
        return _advance_change(client, model, state, progress)
    except Exception:
        return _error(state, "clarification_failed", "The answer could not be processed. Your accepted event is unchanged.")


def cancel_pending(workflow):
    """Discard only the proposed work; this action never calls a provider."""
    state = _begin(workflow)
    state.update(pending=None, phase="READY" if state["accepted"] is not None else "EMPTY",
                 outcome="cancelled")
    return state
