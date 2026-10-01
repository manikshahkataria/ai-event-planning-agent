"""In-memory event revisions: draft, validate, regenerate, then publish."""

from copy import deepcopy

from planner.requirements import REQUIRED_FIELDS, validate_change_patch
from planner.validation import parse_event_date, validate_requirements
from planner.strategy import assess_event_strategy
from planner.event_plan import generate_event_plan
from planner.budget import optimize_budget
from planner.timeline import generate_timeline
from planner.vendors import recommend_vendors


FIELDS = (*REQUIRED_FIELDS, "preferences")
_LABELS = {"event_type": "Event type", "location": "Location", "date": "Date",
           "guest_count": "Guest count", "budget": "Budget", "preferences": "Preferences"}


def _preference(value):
    return " ".join(value.split())


def _issue(field, reason, question):
    return {"field": field, "reason": reason, "question": question, "blocking": True}


def apply_change_patch(accepted_requirements, patch):
    """Build a detached draft. All invalid operations remain pending as issues."""
    proposed = deepcopy(accepted_requirements)
    draft = {"requirements": proposed, "patch": deepcopy(patch), "issues": []}
    try:
        validate_change_patch(patch)
    except (ValueError, TypeError):
        draft["issues"].append(_issue("requirements", "Invalid change response.", "Please restate the change."))
        return draft
    draft["issues"] = deepcopy(patch["issues"])
    seen = set()
    for update in patch["updates"]:
        field, value = update["field"], update["value"]
        if field in seen:
            draft["issues"].append(_issue(field, "Multiple updates target the same field.", f"What should the final {field} be?"))
        seen.add(field)
        proposed[field] = deepcopy(value)
        if field == "date":
            # Validate the supplied date wording too: an LLM must not repair an
            # impossible/ambiguous source date into a valid-looking proposal.
            try:
                source_date = parse_event_date(update["source"])
                if source_date != parse_event_date(value):
                    raise ValueError("The proposed date differs from the supplied date.")
            except ValueError:
                draft["issues"].append(_issue("date", "The supplied date is invalid, incomplete or ambiguous.",
                    "Please provide the full intended date, such as 2026-11-15."))

    preferences = list(proposed.get("preferences", []))
    for operation in patch["preference_changes"]:
        action, target, value = operation["action"], operation["target"], operation["value"]
        if action == "clear":
            preferences = []
            continue
        if action in ("remove", "replace"):
            matches = [i for i, item in enumerate(preferences) if _preference(item) == _preference(target)]
            if len(matches) != 1:
                draft["issues"].append(_issue("preferences", "Preference target is absent or ambiguous.",
                    f"Which existing preference should be changed instead of '{target}'?"))
                continue
            index = matches[0]
            if action == "remove":
                preferences.pop(index)
            else:
                preferences[index] = _preference(value)
        elif action == "add":
            preferences.append(_preference(value))
    proposed["preferences"] = list(dict.fromkeys(_preference(item) for item in preferences))
    draft["issues"].extend(validate_requirements(proposed))
    return draft


def compare_requirements(old, new):
    """Compare validated values; never infer semantic equivalence."""
    changes = {}
    for field in FIELDS:
        before, after = old.get(field), new.get(field)
        if field == "preferences":
            equal = {_preference(v) for v in before or []} == {_preference(v) for v in after or []}
        elif field == "date":
            equal = parse_event_date(before) == parse_event_date(after)
        elif isinstance(before, str) and isinstance(after, str):
            equal = before.strip() == after.strip()
        else:
            equal = before == after
        if not equal:
            changes[field] = {"old": deepcopy(before), "new": deepcopy(after)}
    return changes


def determine_impact(changes):
    """MVP: all effective requirement changes regenerate the core and timeline."""
    if not changes:
        return {}
    return {"strategy": "reassess", "event_plan": "regenerate", "budget_plan": "regenerate",
            "timeline": "regenerate", "vendors": "refresh" if "location" in changes else "check_search_inputs"}


def compare_strategy(old, new):
    changed = [key for key in new if old.get(key) != new[key]]
    def intent(strategy):
        return {tuple(sorted(set(item.get("vendor_types", [])))) for item in strategy.get("categories", [])}
    return {"changed_fields": changed,
            "material": bool(set(changed) & {"approach", "priorities", "categories", "omissions", "assumptions"}),
            "vendor_intent_changed": intent(old) != intent(new)}


def _call(function, **kwargs):
    # Isolate unexpected stage errors as well as the existing None failure path.
    try:
        return function(**deepcopy({k: v for k, v in kwargs.items() if k != "client"}),
                        **({"client": kwargs["client"]} if "client" in kwargs else {}))
    except Exception:
        return None


def notify_progress(callback, stage):
    """Report a public stage only; presentation failures cannot fail planning."""
    if callback is not None:
        try:
            callback(stage)
        except Exception:
            pass


def generate_revision(client, model, requirements, strategy, previous=None, progress=None):
    """Generate a detached revision; optional failures never resurrect old output."""
    if validate_requirements(requirements) or not isinstance(strategy, dict) or not strategy:
        return None
    session = {"revision": previous["revision"] + 1 if previous else 1,
               "requirements": deepcopy(requirements), "strategy": deepcopy(strategy),
               "event_plan": None, "budget_plan": None, "timeline": None,
               "vendor_groups": [], "vendor_cache": deepcopy(previous.get("vendor_cache", {})) if previous else {},
               "component_statuses": {"strategy": "ready"}}
    notify_progress(progress, "event_plan")
    session["event_plan"] = _call(generate_event_plan, client=client, model=model,
                                  event_state=requirements, strategy=strategy)
    if not session["event_plan"]:
        return None
    notify_progress(progress, "budget")
    session["budget_plan"] = _call(optimize_budget, client=client, model=model,
        event_state=requirements, event_plan=session["event_plan"], strategy=strategy)
    if not session["budget_plan"]:
        return None
    session["component_statuses"].update(event_plan="ready", budget_plan="ready")
    notify_progress(progress, "timeline")
    session["timeline"] = _call(generate_timeline, client=client, model=model,
        event_state=requirements, event_plan=session["event_plan"], budget_plan=session["budget_plan"])
    session["component_statuses"]["timeline"] = "ready" if session["timeline"] else "unavailable"
    try:
        # The cache belongs to the detached session and may be populated in place.
        notify_progress(progress, "vendors")
        session["vendor_groups"] = recommend_vendors(requirements["location"], deepcopy(strategy),
                                                     cache=session["vendor_cache"])
        if (not isinstance(session["vendor_groups"], list)
                or any(not isinstance(group, dict) for group in session["vendor_groups"])):
            raise ValueError("Malformed vendor response")
        statuses = [group.get("status") for group in session["vendor_groups"]]
    except Exception:
        session["vendor_groups"] = []
        session["component_statuses"]["vendors"] = "unavailable"
    else:
        session["component_statuses"]["vendors"] = (
            "partial" if "unavailable" in statuses and any(s != "unavailable" for s in statuses)
            else "unavailable" if "unavailable" in statuses else "ready")
    return session


def replan_event(client, model, accepted, patch, conversation_context="", progress=None):
    """Return an outcome; the caller publishes result['session'] only on ready."""
    notify_progress(progress, "validation")
    draft = apply_change_patch(accepted["requirements"], patch)
    result = {"status": "needs_clarification", "draft": draft, "session": None,
              "changes": {}, "impact": {}, "strategy_changes": {}}
    if any(issue["blocking"] for issue in draft["issues"]):
        return result
    changes = compare_requirements(accepted["requirements"], draft["requirements"])
    result.update(changes=changes, impact=determine_impact(changes))
    if not changes:
        result["status"] = "noop"
        return result
    # Preserve exact accepted representations for fields that did not change.
    for field in FIELDS:
        if field not in changes:
            draft["requirements"][field] = deepcopy(accepted["requirements"][field])
    notify_progress(progress, "strategy")
    assessment = _call(assess_event_strategy, client=client, model=model,
        event_state=draft["requirements"], conversation_context=conversation_context,
        previous_strategy=accepted["strategy"], changes=changes)
    if not assessment:
        result["status"] = "failed"
        return result
    if assessment["status"] != "ready":
        draft["issues"] = assessment["issues"]
        return result
    result["strategy_changes"] = compare_strategy(accepted["strategy"], assessment["strategy"])
    result["session"] = generate_revision(client, model, draft["requirements"], assessment["strategy"], accepted,
                                          progress=progress)
    result["status"] = "ready" if result["session"] else "failed"
    return result


def build_change_summary(result):
    """Report only published work; no LLM required."""
    if result["status"] == "noop":
        return "No effective changes. All requirements and outputs preserved."
    if result["status"] != "ready":
        return "Update pending. The accepted event revision is unchanged."
    def display(field, value):
        if field == "budget":
            return f"₹{value:,.0f}" if value == int(value) else f"₹{value:,}"
        return ", ".join(value) if isinstance(value, list) else str(value)
    lines = ["UPDATED"]
    for field, change in result["changes"].items():
        lines.append(f"{_LABELS[field]}: {display(field, change['old'])} → {display(field, change['new'])}")
    lines += ["", "REASSESSED", "Strategy", "", "REPLANNED", "Event plan", "Budget"]
    session = result["session"]
    if session["component_statuses"]["timeline"] == "ready":
        lines.append("Timeline")
    else:
        lines.append("Timeline unavailable — previous timeline is not current")
    lines += ["", "PRESERVED"]
    lines.extend(_LABELS[field] for field in FIELDS if field not in result["changes"])
    groups = session["vendor_groups"]
    searched = sum(group.get("search_action") == "searched" for group in groups)
    reused = sum(group.get("search_action") == "reused" for group in groups)
    if reused and not searched:
        lines.append("Vendor search results — search inputs unchanged; regrouped for this strategy")
    if searched:
        lines += ["", "VENDOR DISCOVERY", f"New searches: {searched}; reused groups: {reused}"]
    if session["component_statuses"]["vendors"] in ("unavailable", "partial"):
        lines.append("Some vendor results are unavailable; no stale results substituted")
    elif not searched and not reused:
        lines.append("No supported vendor searches required")
    return "\n".join(lines)
