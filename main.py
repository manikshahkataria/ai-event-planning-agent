import os

from dotenv import load_dotenv
from openai import OpenAI

from planner.requirements import (
    extract_requirements,
    get_missing_fields,
)

from planner.event_plan import generate_event_plan

from planner.budget import optimize_budget

from planner.timeline import generate_timeline
from planner.validation import validate_requirements
from planner.strategy import assess_event_strategy
# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env"
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    max_retries=0,
)


MODEL = "inclusionai/ling-3.0-flash-vl"


# ==========================================
# INITIAL EVENT STATE
# ==========================================

event_state = {
    "event_type": None,
    "location": None,
    "date": None,
    "guest_count": None,
    "budget": None,
    "preferences": [],
}


# ==========================================
# FOLLOW-UP QUESTIONS
# ==========================================

QUESTIONS = {
    "event_type": "What type of event are you planning?",

    "location": "Where would you like to organize the event?",

    "date": "What date are you planning the event for?",

    "guest_count": "Approximately how many guests are you expecting?",

    "budget": "What is your total budget for the event?",
}


# ==========================================
# START APPLICATION
# ==========================================

print()
print("=" * 50)
print("🎉 AI EVENT PLANNING AGENT")
print("=" * 50)


user_message = input(
    "\nDescribe the event you want to plan:\n> "
)


# ==========================================
# EXTRACT INITIAL REQUIREMENTS
# ==========================================

event_state = extract_requirements(
    client=client,
    model=MODEL,
    user_message=user_message,
    current_state=event_state,
)
validation_issues = validate_requirements(event_state)
conversation_context = user_message


# ==========================================
# COLLECT MISSING INFORMATION
# ==========================================

while True:

    missing_fields = get_missing_fields(event_state)

    blocking_issues = [
        issue for issue in validation_issues if issue["blocking"]
    ]

    contextual_clarification = False
    if missing_fields:
        field = missing_fields[0]
        question = QUESTIONS[field]
    elif blocking_issues:
        issue = blocking_issues[0]
        field = issue["field"]
        question = issue["question"]
    else:
        assessment = assess_event_strategy(
            client=client,
            model=MODEL,
            event_state=event_state,
            conversation_context=conversation_context,
        )
        if assessment is None:
            print("\nPlanning stopped because event strategy could not be assessed.")
            raise SystemExit(0)
        if assessment["status"] == "ready":
            strategy = assessment["strategy"]
            break
        issue = next(issue for issue in assessment["issues"] if issue["blocking"])
        field = ", ".join(issue["fields"])
        question = issue["question"]
        contextual_clarification = True
        print(f"\n{issue['reason']}")

    print(f"\nAI Planner: {question}")

    user_answer = input("You: ")

    contextual_message = f"""
The planner asked the user:

{question}

The user answered:

{user_answer}

The answer is related to the following field(s):
{field}
"""
    conversation_context += "\n" + contextual_message

    previous_state = event_state.copy()

    event_state = extract_requirements(
        client=client,
        model=MODEL,
        user_message=contextual_message,
        current_state=event_state,
    )
    validation_issues = validate_requirements(event_state)

    # Prevent endless loop if extraction fails
    if event_state == previous_state and not contextual_clarification:
        print(
            "\n⚠️ I couldn't understand that answer."
            " Please try again."
        )


# ==========================================
# DISPLAY FINAL REQUIREMENTS
# ==========================================

print()
print("=" * 50)
print("✅ EVENT REQUIREMENTS COMPLETE")
print("=" * 50)

print(f"Event Type : {event_state['event_type']}")
print(f"Location   : {event_state['location']}")
print(f"Date       : {event_state['date']}")
print(f"Guests     : {event_state['guest_count']}")
print(f"Budget     : ₹{event_state['budget']}")


preferences = event_state.get(
    "preferences",
    []
)

if preferences:
    print(
        f"Preferences: {', '.join(map(str, preferences))}"
    )
else:
    print("Preferences: None specified")


print()
print("=" * 50)
print("🤖 GENERATING EVENT PLAN")
print("=" * 50)


event_plan = generate_event_plan(
    client=client,
    model=MODEL,
    event_state=event_state,
    strategy=strategy,
)


if event_plan:

    print()
    print("=" * 50)
    print("🎯 YOUR EVENT PLAN")
    print("=" * 50)

    print(
        f"\n{event_plan['event_summary']}"
    )

    print("\nPLAN")
    print("-" * 50)

    for item in event_plan["categories"]:

        print(
            f"\n{item['category']}"
        )

        print(
            f"  Recommendation: "
            f"{item['recommendation']}"
        )

        print(
            f"  Estimated Budget: "
            f"₹{item['estimated_budget']:,.0f}"
        )

    print()
    print("-" * 50)

    print(
        f"Estimated Cost : "
        f"₹{event_plan['total_estimated_cost']:,.0f}"
    )

    print(
        f"Remaining Budget: "
        f"₹{event_plan['remaining_budget']:,.0f}"
    )

    notes = event_plan.get(
        "planning_notes",
        []
    )

    if notes:

        print("\nPlanning Notes:")

        for note in notes:
            print(f"  • {note}")

        # ==========================================
    # MILESTONE 3 - BUDGET INTELLIGENCE
    # ==========================================

        # ==========================================
    # MILESTONE 3 - BUDGET INTELLIGENCE
    # ==========================================

    print()
    print("=" * 50)
    print("💰 OPTIMIZING EVENT BUDGET")
    print("=" * 50)

    budget_plan = optimize_budget(
        client=client,
        model=MODEL,
        event_state=event_state,
        event_plan=event_plan,
        strategy=strategy,
    )

    if budget_plan:

        print()
        print("=" * 50)
        print("💰 INTELLIGENT BUDGET PLAN")
        print("=" * 50)

        print(
            f"\nStrategy: "
            f"{budget_plan['budget_strategy']}"
        )

        print("\nBUDGET ALLOCATION")
        print("-" * 50)

        for item in budget_plan["allocations"]:

            print(
                f"\n{item['category']} "
                f"[{item['priority'].upper()} PRIORITY]"
            )

            print(
                f"  Allocation: "
                f"₹{item['allocated_budget']:,.0f}"
            )

            print(
                f"  Why: {item['reason']}"
            )

        print()
        print("-" * 50)

        print(
            f"Total Allocated : "
            f"₹{budget_plan['total_allocated']:,.0f}"
        )

        print(
            f"Remaining Budget: "
            f"₹{budget_plan['remaining_budget']:,.0f}"
        )

        tradeoffs = budget_plan.get(
            "tradeoffs",
            []
        )

        if tradeoffs:

            print("\n⚖️ TRADE-OFFS")

            for tradeoff in tradeoffs:
                print(f"  • {tradeoff}")

        # ==========================================
        # MILESTONE 4 - TASKS & TIMELINE
        # ==========================================

        print()
        print("=" * 50)
        print("📅 GENERATING EXECUTION TIMELINE")
        print("=" * 50)

        timeline = generate_timeline(
            client=client,
            model=MODEL,
            event_state=event_state,
            event_plan=event_plan,
            budget_plan=budget_plan,
        )

        if timeline:

            print()
            print("=" * 50)
            print("📋 EVENT EXECUTION PLAN")
            print("=" * 50)

            print(
                f"\n{timeline['timeline_summary']}"
            )

            print("\nTASKS")
            print("-" * 50)

            tasks = timeline["tasks"]

            task_lookup = {
                task["task_id"]: task["task"]
                for task in tasks
            }

            for task in tasks:

                print(
                    f"\nTask {task['task_id']}: "
                    f"{task['task']}"
                )

                print(
                    f"  Category: "
                    f"{task['category']}"
                )

                print(
                    f"  Priority: "
                    f"{task['priority'].upper()}"
                )

                print(
                    f"  Due: {task['due_date']}"
                )

                dependencies = task["dependencies"]

                if dependencies:

                    dependency_names = [
                        task_lookup.get(
                            dependency,
                            f"Task {dependency}"
                        )
                        for dependency in dependencies
                    ]

                    print(
                        "  Depends on: "
                        + ", ".join(dependency_names)
                    )

                else:
                    print("  Depends on: None")

                print(
                    f"  Why: {task['reason']}"
                )

            # ======================================
            # CRITICAL TASKS
            # ======================================

            critical_tasks = timeline.get(
                "critical_tasks",
                []
            )

            if critical_tasks:

                print()
                print("-" * 50)
                print("🚨 CRITICAL TASKS")

                for task_id in critical_tasks:

                    task_name = task_lookup.get(
                        task_id,
                        f"Task {task_id}"
                    )

                    print(
                        f"  • Task {task_id}: "
                        f"{task_name}"
                    )

            # ======================================
            # EXECUTION NOTES
            # ======================================

            execution_notes = timeline.get(
                "execution_notes",
                []
            )

            if execution_notes:

                print()
                print("📝 EXECUTION NOTES")

                for note in execution_notes:
                    print(f"  • {note}")

        else:

            print(
                "\n⚠️ Unable to generate "
                "execution timeline."
            )

    else:

        print(
            "\n⚠️ Unable to generate "
            "budget intelligence."
        )

else:

    print(
        "\n⚠️ Unable to generate event plan."
    )
