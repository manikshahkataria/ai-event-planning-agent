"""Streamlit presentation for the shared event-planning controller.

Only explicit submission callbacks invoke workflow actions. Rendering reads saved
results and never contacts a provider, including on ordinary Streamlit reruns.
"""

from copy import deepcopy
from urllib.parse import urlsplit

import streamlit as st

from planner import controller
from planner.llm_client import create_llm_client, get_llm_model


STAGES = {
    "requirements": "Understanding your event…",
    "change_extraction": "Understanding your changes…",
    "validation": "Checking your requirements…",
    "strategy": "Checking feasibility and building your strategy…",
    "event_plan": "Creating your event plan…",
    "budget": "Creating your budget…",
    "timeline": "Building your timeline…",
    "vendors": "Finding real places or reusing saved results…",
}
VENDOR_DISCLAIMER = (
    "Place results do not verify capacity, availability, price, suitability, or budget fit. "
    "Strategy coverage describes planning intent, not confirmed services from a business."
)


def _initialize_session():
    if "workflow" not in st.session_state:
        st.session_state.workflow = controller.new_workflow()
    for key, value in (("action_generation", 0), ("busy", False), ("ui_error", None), ("failed_input", None)):
        if key not in st.session_state:
            st.session_state[key] = value


def _submit_action(action, text_key, generation):
    """The sole provider boundary; called only by an explicit form/button action."""
    if st.session_state.busy or generation != st.session_state.action_generation:
        return
    text = st.session_state.get(text_key, "").strip() if text_key else ""
    if action != "cancel" and not text:
        st.session_state.ui_error = "Please enter a description or answer before submitting."
        return
    st.session_state.action_generation += 1
    st.session_state.ui_error = None
    st.session_state.failed_input = None
    st.session_state.busy = True
    try:
        workflow = deepcopy(st.session_state.workflow)
        if action == "cancel":
            st.session_state.workflow = controller.cancel_pending(workflow)
            return
        with st.status("Working on your event…", expanded=True) as status:
            def progress(stage):
                status.update(label=STAGES.get(stage, "Working on your event…"))

            # Client lifetime belongs to this explicit action, never a rerender.
            with create_llm_client() as client:
                model = get_llm_model(client)
                if action == "create":
                    result = controller.start_plan(client, model, text, workflow, progress=progress)
                elif action == "clarify":
                    result = controller.submit_clarification(client, model, workflow, text, progress=progress)
                elif action == "change":
                    result = controller.submit_change(client, model, workflow, text, progress=progress)
                else:
                    raise ValueError("Unknown UI action")
            if result["error"]:
                status.update(label="This request could not be completed.", state="error", expanded=False)
            elif result["phase"] == "NEEDS_CLARIFICATION":
                status.update(label="One more detail is needed.", state="complete", expanded=False)
            else:
                status.update(label="Your event is ready.", state="complete", expanded=False)
        st.session_state.workflow = result
    except Exception:
        # Never render exception bodies, request URLs, headers, or credentials.
        st.session_state.ui_error = (
            "We couldn't complete that request. Your saved plan is unchanged. "
            "Please try again. If the problem continues, check the provider configuration."
        )
        st.session_state.failed_input = {"action": action, "text": text}
    finally:
        st.session_state.busy = False


def _money(value):
    return f"₹{value:,.2f}" if isinstance(value, (int, float)) else "Not available"


def _website_url(value):
    """Accept supplied web URLs only; never construct a booking/business URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if any(char.isspace() or ord(char) < 32 for char in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme.lower() in {"http", "https"} and parsed.hostname
                and not parsed.username and not parsed.password):
            return value
    except ValueError:
        pass
    return None


def _items(values):
    for item in values or []:
        st.text(f"• {item}")


def render_overview(session):
    requirements = session["requirements"]
    st.subheader("Event overview")
    st.caption(f"Current plan · Version {session['revision']}")
    columns = st.columns(3)
    for column, label, field in zip(columns, ("Event", "Location", "Date"), ("event_type", "location", "date")):
        with column:
            st.caption(label)
            st.text(requirements[field])
    guests, budget = st.columns(2)
    guests.metric("Guests", requirements["guest_count"])
    budget.metric("Total budget", _money(requirements["budget"]))
    st.caption("Your preferences")
    st.text(", ".join(requirements.get("preferences", [])) or "None specified")


def render_strategy(strategy):
    st.subheader("Your event strategy")
    st.text(strategy["approach"])
    st.caption("Why this approach")
    st.text(strategy["rationale"])
    st.markdown("**Your priorities**")
    _items(strategy.get("priorities"))
    st.markdown("**Services and bundles**")
    for category in strategy.get("categories", []):
        with st.container(border=True):
            st.text(category["category"])
            st.caption("Intended coverage")
            st.text(", ".join(category["covers"]))
            st.text(category["reason"])
    with st.expander("Assumptions, omissions & alternatives"):
        st.markdown("**Assumptions**")
        _items(strategy.get("assumptions"))
        st.markdown("**Services left out**")
        for omission in strategy.get("omissions", []):
            st.text(f"{omission['service']}: {omission['reason']}")
        st.markdown("**Other approaches considered**")
        for alternative in strategy.get("alternatives", []):
            st.text(f"{alternative['approach']}: {alternative['tradeoff']}")


def render_plan_budget(session):
    plan, budget = session["event_plan"], session["budget_plan"]
    st.subheader("The plan")
    st.text(plan["event_summary"])
    for category in plan.get("categories", []):
        with st.container(border=True):
            st.text(category["category"])
            st.text(category["recommendation"])
            st.caption(f"Initial planning estimate: {_money(category['estimated_budget'])}")
    _items(plan.get("planning_notes"))
    st.subheader("Budget allocation")
    st.text(budget["budget_strategy"])
    st.caption("Planning estimates, not vendor quotes. Final allocations are shown below.")
    st.dataframe([
        {"Category": item["category"], "Priority": item["priority"].title(),
         "Allocation": _money(item["allocated_budget"]), "Reason": item["reason"]}
        for item in budget.get("allocations", [])
    ], hide_index=True, width="stretch")
    allocated, remaining = st.columns(2)
    allocated.metric("Total allocated", _money(budget["total_allocated"]))
    remaining.metric("Remaining budget", _money(budget["remaining_budget"]))
    st.markdown("**Budget trade-offs**")
    _items(budget.get("tradeoffs"))


def render_timeline(session):
    timeline = session.get("timeline")
    if not timeline or session["component_statuses"].get("timeline") == "unavailable":
        st.warning("The timeline is unavailable for this version. Your event plan and budget are saved.")
        return
    st.subheader("Execution timeline")
    st.text(timeline["timeline_summary"])
    tasks = timeline.get("tasks", [])
    names = {task["task_id"]: task["task"] for task in tasks}
    critical = timeline.get("critical_tasks", [])
    st.dataframe([
        {"Task": task["task"], "Category": task["category"], "Due": task["due_date"],
         "Priority": task["priority"].title(), "Critical": "Yes" if task["task_id"] in critical else "",
         "Depends on": ", ".join(names.get(key, f"Task {key}") for key in task["dependencies"]) or "None",
         "Why": task["reason"]} for task in tasks
    ], hide_index=True, width="stretch")
    if critical:
        st.markdown("**Critical tasks**")
        _items([names[key] for key in critical if key in names])
    if not tasks:
        st.info("No tasks were returned for this version.")
    _items(timeline.get("execution_notes"))


def render_places(session):
    st.subheader("Real places to consider")
    st.info(VENDOR_DISCLAIMER)
    st.caption("Geoapify results · Within 10 km of the resolved location · No booking or availability checks")
    groups = session.get("vendor_groups", [])
    if not groups:
        if session["component_statuses"].get("vendors") == "unavailable":
            st.warning("Place discovery is unavailable for this version. Your event plan is saved.")
        else:
            st.info("No supported place searches were requested by this strategy.")
    for group in groups:
        with st.container(border=True):
            st.text(group["category"])
            st.caption("Intended coverage: " + (", ".join(group.get("covers", [])) or "Not specified"))
            if group.get("search_action") == "reused":
                st.caption("Saved place results reused for the same search area and type.")
            state = group.get("status")
            if state == "ok":
                for place in group.get("places", []):
                    st.divider()
                    st.text(place.get("name") or "Name not supplied")
                    st.text(place.get("address") or "Address not supplied")
                    website = _website_url(place.get("website"))
                    if website:
                        st.link_button("Visit website", website)
                    if place.get("phone"):
                        st.text(f"Phone: {place['phone']}")
                    st.caption(f"Source: {place.get('source') or 'Not supplied'}")
            elif state == "unavailable":
                st.warning("Place discovery is temporarily unavailable for this category.")
            elif state == "empty":
                st.info("No places were returned in the search area.")
            elif state == "unsupported":
                st.info("Place discovery does not support this service yet.")
            else:
                st.info("No supported external place search was requested for this category.")
            if group.get("unsupported_types") and state != "unsupported":
                st.caption("Some requested services are not supported by place discovery.")


def _render_section(renderer, value):
    try:
        renderer(value)
    except Exception:
        st.warning("This section could not be displayed. Your saved plan is unchanged.")


def _request_form(action, label, button, placeholder, default=""):
    generation = st.session_state.action_generation
    text_key = f"{action}_text_{generation}"
    failed_input = st.session_state.failed_input
    if failed_input and failed_input["action"] == action:
        default = failed_input["text"]
    with st.form(f"{action}_form_{generation}"):
        st.text_area(label, value=default, placeholder=placeholder, height=130, key=text_key)
        st.form_submit_button(button, type="primary", disabled=st.session_state.busy,
                              on_click=_submit_action, args=(action, text_key, generation))


def main():
    st.set_page_config(page_title="AI Event Planner", page_icon="📅", layout="wide")
    _initialize_session()
    st.title("AI Event Planner")
    st.write("Plan smarter. Adapt when things change.")
    workflow = st.session_state.workflow
    accepted, pending = workflow["accepted"], workflow["pending"]
    if st.session_state.ui_error:
        st.error(st.session_state.ui_error)
    if workflow["error"]:
        st.error(workflow["error"]["message"])
    if workflow["notice"]:
        st.info(workflow["notice"])
    question = pending.get("question") if pending else None
    if question:
        st.subheader("One more detail")
        st.text(question["reason"])
        st.text(question["question"])
        if accepted:
            st.caption("Your current plan stays saved while we clarify this change.")
        _request_form("clarify", "Your answer", "Submit Answer", "Add the detail requested above…")
    elif not accepted:
        st.caption("Tell us the occasion, where and when, guest count, budget, and what matters most.")
        _request_form("create", "Describe your event", "Create Plan",
                      "A casual birthday in Delhi on 15 November 2026 for 20 people, with a ₹25,000 budget…",
                      pending.get("request", "") if pending else "")
    if pending:
        st.button("Discard pending request", on_click=_submit_action,
                  args=("cancel", None, st.session_state.action_generation), disabled=st.session_state.busy)
    if accepted:
        if workflow["last_summary"]:
            with st.expander("What changed", expanded=workflow["outcome"] in {"ready", "noop"}):
                st.text(workflow["last_summary"])
        st.divider()
        _render_section(render_overview, accepted)
        strategy, budget, timeline, places = st.tabs(["Strategy", "Plan & Budget", "Timeline", "Real Places"])
        with strategy:
            _render_section(render_strategy, accepted["strategy"])
        with budget:
            _render_section(render_plan_budget, accepted)
        with timeline:
            _render_section(render_timeline, accepted)
        with places:
            _render_section(render_places, accepted)
        if not question:
            st.divider()
            st.subheader("Change your plan")
            _request_form("change", "What would you like to change?", "Update Plan",
                          "Increase the guest count to 35 and reduce the budget to ₹20,000.",
                          pending.get("request", "") if pending else "")
    st.caption("Your plan stays in this browser session. Closing or resetting the session may clear it.")


if __name__ == "__main__":
    main()
