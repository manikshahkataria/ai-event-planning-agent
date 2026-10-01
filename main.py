"""Synchronous event-planning CLI with transactional in-memory revisions."""

from copy import deepcopy

from planner.llm_client import create_llm_client, get_llm_model
from planner.requirements import extract_requirements, extract_requirement_changes, get_missing_fields
from planner.validation import validate_requirements
from planner.strategy import assess_event_strategy
from planner.replanning import generate_revision, replan_event, build_change_summary


QUESTIONS = {'event_type': 'What type of event are you planning?', 'location': 'Where would you like to organize the event?', 'date': 'What date are you planning the event for?', 'guest_count': 'Approximately how many guests are you expecting?', 'budget': 'What is your total budget for the event?'}


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


def collect_requirements(client, model, event_state, conversation_context):
    """Initial collection gate; returns validated requirements and ready strategy."""
    while True:
        missing_fields = get_missing_fields(event_state)
        blocking_issues = [issue for issue in validate_requirements(event_state) if issue["blocking"]]
        contextual_clarification = False
        if missing_fields:
            field = missing_fields[0]
            question = QUESTIONS[field]
        elif blocking_issues:
            issue = blocking_issues[0]
            field, question = issue["field"], issue["question"]
        else:
            assessment = assess_event_strategy(client=client, model=model, event_state=event_state,
                                               conversation_context=conversation_context)
            if assessment is None:
                print("\nPlanning stopped because event strategy could not be assessed.")
                return None
            if assessment["status"] == "ready":
                return event_state, assessment["strategy"], conversation_context
            issue = next(issue for issue in assessment["issues"] if issue["blocking"])
            field, question = ", ".join(issue["fields"]), issue["question"]
            contextual_clarification = True
            print(issue["reason"])
        print(f"\nAI Planner: {question}")
        user_answer = input("You: ")
        if _exit_requested(user_answer):
            return None
        contextual_message = f"The planner asked: {question}\nThe user answered: {user_answer}\nRelated fields: {field}"
        conversation_context += "\n" + contextual_message
        previous_state = event_state
        event_state = extract_requirements(client=client, model=model, user_message=contextual_message,
                                           current_state=event_state)
        if event_state == previous_state and not contextual_clarification:
            print("I couldn't understand that answer. Please try again.")


def handle_changes(client, model, session):
    """Keep accepted state separate from every pending change and clarification."""
    while True:
        request = input("\nDescribe a change, or enter done to finish:\n> ").strip()
        if _exit_requested(request):
            return session
        if not request:
            continue
        context = ""
        while True:
            patch = extract_requirement_changes(client, model, deepcopy(session["requirements"]), request, context)
            if patch is None:
                print("Change interpretation failed. The accepted revision is unchanged.")
                break
            result = replan_event(client, model, session, patch,
                                  conversation_context=request + "\n" + context)
            if result["status"] == "needs_clarification":
                issue = next(issue for issue in result["draft"]["issues"] if issue["blocking"])
                print(f"\nUpdate pending: {issue['reason']}")
                print(issue["question"])
                answer = input("You (or cancel to discard this update): ")
                if _exit_requested(answer):
                    return session
                if answer.strip().lower() == "cancel":
                    print("Proposed update discarded. Accepted event unchanged.")
                    break
                context += f"\nQuestion: {issue['question']}\nAnswer: {answer}"
                continue
            if result["status"] == "ready":
                session = result["session"]
                print(build_change_summary(result))
                display_session(session)
            elif result["status"] == "noop":
                print(build_change_summary(result))
            else:
                print("Replanning failed. The accepted revision is unchanged; you can retry the change.")
            break


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
            event_state = {"event_type": None, "location": None, "date": None,
                           "guest_count": None, "budget": None, "preferences": []}
            event_state = extract_requirements(client, model, message, event_state)
            collected = collect_requirements(client, model, event_state, message)
            if collected is None:
                return
            requirements, strategy, _ = collected
            session = generate_revision(client, model, requirements, strategy)
            if session is None:
                print("Unable to complete the event plan and budget. No revision was accepted.")
                return
            display_session(session)
            handle_changes(client, model, session)
        except (EOFError, KeyboardInterrupt):
            print("\nPlanning session ended.")


if __name__ == "__main__":
    main()
