"""Synchronous event-planning CLI with transactional in-memory revisions."""

from planner.llm_client import create_llm_client, get_llm_model
from planner import controller


def display_requirements(event_state):
    print()
    print('=' * 50)
    print('✅ EVENT REQUIREMENTS COMPLETE')
    print('=' * 50)
    print(f"Event Type : {event_state['event_type']}")
    print(f"Location   : {event_state['location']}")
    print(f"Date       : {event_state['date']}")
    print(f"Guests     : {event_state['guest_count']}")
    print(f"Budget     : ₹{event_state['budget']}")
    preferences = event_state.get('preferences', [])
    if preferences:
        print(f"Preferences: {', '.join(map(str, preferences))}")
    else:
        print('Preferences: None specified')


def display_event_plan(event_plan):
    print()
    print('=' * 50)
    print('🎯 YOUR EVENT PLAN')
    print('=' * 50)
    print(f"\n{event_plan['event_summary']}")
    print('\nPLAN')
    print('-' * 50)
    for item in event_plan['categories']:
        print(f"\n{item['category']}")
        print(f"  Recommendation: {item['recommendation']}")
        print(f"  Estimated Budget: ₹{item['estimated_budget']:,.0f}")
    print()
    print('-' * 50)
    print(f"Estimated Cost : ₹{event_plan['total_estimated_cost']:,.0f}")
    print(f"Remaining Budget: ₹{event_plan['remaining_budget']:,.0f}")
    notes = event_plan.get('planning_notes', [])
    if notes:
        print('\nPlanning Notes:')
        for note in notes:
            print(f'  • {note}')


def display_budget(budget_plan):
    print()
    print('=' * 50)
    print('💰 INTELLIGENT BUDGET PLAN')
    print('=' * 50)
    print(f"\nStrategy: {budget_plan['budget_strategy']}")
    print('\nBUDGET ALLOCATION')
    print('-' * 50)
    for item in budget_plan['allocations']:
        print(f"\n{item['category']} [{item['priority'].upper()} PRIORITY]")
        print(f"  Allocation: ₹{item['allocated_budget']:,.0f}")
        print(f"  Why: {item['reason']}")
    print()
    print('-' * 50)
    print(f"Total Allocated : ₹{budget_plan['total_allocated']:,.0f}")
    print(f"Remaining Budget: ₹{budget_plan['remaining_budget']:,.0f}")
    tradeoffs = budget_plan.get('tradeoffs', [])
    if tradeoffs:
        print('\n⚖️ TRADE-OFFS')
        for tradeoff in tradeoffs:
            print(f'  • {tradeoff}')


def display_timeline(timeline):
    print()
    print('=' * 50)
    print('📋 EVENT EXECUTION PLAN')
    print('=' * 50)
    print(f"\n{timeline['timeline_summary']}")
    print('\nTASKS')
    print('-' * 50)
    tasks = timeline['tasks']
    task_lookup = {task['task_id']: task['task'] for task in tasks}
    for task in tasks:
        print(f"\nTask {task['task_id']}: {task['task']}")
        print(f"  Category: {task['category']}")
        print(f"  Priority: {task['priority'].upper()}")
        print(f"  Due: {task['due_date']}")
        dependencies = task['dependencies']
        if dependencies:
            dependency_names = [task_lookup.get(dependency, f'Task {dependency}') for dependency in dependencies]
            print('  Depends on: ' + ', '.join(dependency_names))
        else:
            print('  Depends on: None')
        print(f"  Why: {task['reason']}")
    critical_tasks = timeline.get('critical_tasks', [])
    if critical_tasks:
        print()
        print('-' * 50)
        print('🚨 CRITICAL TASKS')
        for task_id in critical_tasks:
            task_name = task_lookup.get(task_id, f'Task {task_id}')
            print(f'  • Task {task_id}: {task_name}')
    execution_notes = timeline.get('execution_notes', [])
    if execution_notes:
        print()
        print('📝 EXECUTION NOTES')
        for note in execution_notes:
            print(f'  • {note}')


def display_vendors(vendor_groups):
    print("\nREAL VENDOR / PLACE RECOMMENDATIONS")
    print("Search area: within 10 km of the resolved location.")
    print("Strategy coverage is planning intent; places are not verified for capacity, availability or budget fit.")
    for group in vendor_groups:
        print(f"\n{group['category']}")
        print("Intended strategy coverage: " + (", ".join(group["covers"]) or "Not specified"))
        if group["unsupported_types"]:
            print("Unsupported search types skipped: " + ", ".join(group["unsupported_types"]))
        if group["status"] == "ok":
            print("Places to consider:")
            for place in group["places"]:
                print()
                for field, label in (("name", "Name"), ("address", "Address"),
                                     ("website", "Website"), ("phone", "Phone"), ("source", "Source")):
                    if place.get(field):
                        print(f"  {label}: {place[field]}")
        if group["message"]:
            print(group["message"])


def display_session(session):
    print(f"\nEVENT REVISION {session['revision']}")
    display_requirements(session["requirements"])
    display_event_plan(session["event_plan"])
    display_budget(session["budget_plan"])
    if session["timeline"]:
        display_timeline(session["timeline"])
    else:
        print("\nExecution timeline is unavailable for this revision.")
    if session["vendor_groups"]:
        display_vendors(session["vendor_groups"])
    elif session["component_statuses"]["vendors"] == "unavailable":
        print("\nVendor discovery is unavailable for this revision.")


def _exit_requested(message):
    return message.strip().lower() in {"exit", "quit", "done"}


def resolve_clarifications(client, model, workflow):
    """CLI input/output only; the controller chooses and processes questions."""
    while workflow["phase"] == "NEEDS_CLARIFICATION":
        question = workflow["pending"]["question"]
        changing = workflow["pending"]["kind"] == "change"
        if workflow["notice"]:
            print(workflow["notice"])
        if changing:
            print(f"\nUpdate pending: {question['reason']}")
            print(question["question"])
        else:
            if question["kind"] == "strategy":
                print(question["reason"])
            print(f"\nAI Planner: {question['question']}")
        answer = input("You (or cancel to discard this update): " if changing else "You: ")
        if _exit_requested(answer):
            return workflow, True
        if changing and answer.strip().lower() == "cancel":
            print("Proposed update discarded. Accepted event unchanged.")
            return controller.cancel_pending(workflow), False
        workflow = controller.submit_clarification(client, model, workflow, answer)
    return workflow, False


def handle_changes(client, model, session):
    """Render and submit CLI actions; all planning decisions live in controller."""
    workflow = controller.new_workflow(session)
    while True:
        request = input("\nDescribe a change, or enter done to finish:\n> ").strip()
        if _exit_requested(request):
            return session
        if not request:
            continue
        workflow = controller.submit_change(client, model, workflow, request)
        workflow, exiting = resolve_clarifications(client, model, workflow)
        if exiting:
            return session
        if workflow["outcome"] == "ready":
            session = workflow["accepted"]
            print(workflow["last_summary"])
            display_session(session)
        elif workflow["outcome"] == "noop":
            print(workflow["last_summary"])
        elif workflow["error"]:
            print(workflow["error"]["message"])


def main():
    try:
        client = create_llm_client()
    except ValueError as error:
        print(error)
        return
    with client:
        model = get_llm_model(client)
        print("\nAI EVENT PLANNING AGENT")
        try:
            message = input("\nDescribe the event you want to plan:\n> ")
            if _exit_requested(message):
                return
            workflow = controller.start_plan(client, model, message)
            workflow, exiting = resolve_clarifications(client, model, workflow)
            if exiting:
                return
            if workflow["error"]:
                print(workflow["error"]["message"])
                return
            session = workflow["accepted"]
            display_session(session)
            handle_changes(client, model, session)
        except (EOFError, KeyboardInterrupt):
            print("\nPlanning session ended.")


if __name__ == "__main__":
    main()
