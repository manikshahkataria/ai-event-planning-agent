import json
from planner.llm_client import create_reliable_completion


TIMELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "timeline_summary": {
            "type": "string"
        },

        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer"
                    },
                    "task": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "high",
                            "medium",
                            "low"
                        ]
                    },
                    "due_date": {
                        "type": "string"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        }
                    },
                    "reason": {
                        "type": "string"
                    }
                },

                "required": [
                    "task_id",
                    "task",
                    "category",
                    "priority",
                    "due_date",
                    "dependencies",
                    "reason"
                ],

                "additionalProperties": False
            }
        },

        "critical_tasks": {
            "type": "array",
            "items": {
                "type": "integer"
            }
        },

        "execution_notes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "timeline_summary",
        "tasks",
        "critical_tasks",
        "execution_notes"
    ],

    "additionalProperties": False
}


def generate_timeline(
    client,
    model,
    event_state,
    event_plan,
    budget_plan
):
    """
    Convert the event plan into an executable
    task timeline with priorities and dependencies.
    """

    system_prompt = """
You are the execution-planning component of an
AI Event Planning Agent.

Your job is to convert an event plan into a
practical sequence of tasks.

The output must help the user understand:

- What needs to be done
- When it should be completed
- Which tasks are most important
- Which tasks depend on other tasks

RULES:

1. Create practical tasks required to execute
   the supplied event plan.

2. Do not create unnecessary tasks.

3. Every task must have a unique integer task_id.

4. task_id values must start at 1 and increase
   sequentially.

5. Dependencies must contain task_id values.

Example:

Task 1:
Finalize guest list

Task 2:
Shortlist venue
dependencies = [1]

This means Task 2 depends on Task 1.

6. Never create a dependency on a task that
   occurs later in the task list.

7. If a task has no dependency, return:

dependencies = []

8. Assign each task a priority:
   high, medium, or low.

9. High-priority tasks should include activities
   that could block other important work or cause
   major event problems if delayed.

10. Use the event date to create realistic
    deadlines.

11. due_date should use this format:

YYYY-MM-DD

12. Every due date must be before or on the
    event date.

13. Consider the user's priorities and the
    supplied budget plan.

14. Tasks related to high-priority budget
    categories should receive appropriate
    attention in the timeline.

15. Do not recommend specific vendors.

16. Do not invent additional user preferences.

17. critical_tasks must contain task_id values
    for the tasks that are especially important
    to successful event execution.

18. Explain briefly why each task is necessary.

19. Keep the timeline practical for a real user.

20. Do not include tasks that should already
    have happened before the planning period
    unless they are still necessary.
"""

    user_prompt = f"""
EVENT REQUIREMENTS:

{json.dumps(event_state, indent=2)}


EVENT PLAN:

{json.dumps(event_plan, indent=2)}


BUDGET PLAN:

{json.dumps(budget_plan, indent=2)}


Create an executable task timeline for this event.

The event date is:

{event_state["date"]}
"""

    try:

        response = create_reliable_completion(
            client=client,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event_timeline",
                    "strict": True,
                    "schema": TIMELINE_SCHEMA
                }
            },
        )

        if response is None:
            print(
                "\nTimeline generation "
                "could not reach the AI service."
            )
            return None

        raw_response = (
            response.choices[0].message.content
        )


        if not raw_response:

            print(
                "\n❌ Empty timeline response."
            )

            return None

        timeline = json.loads(
            raw_response
        )

        tasks = timeline.get(
            "tasks",
            []
        )

        # ----------------------------------
        # BASIC DEPENDENCY VALIDATION
        # ----------------------------------

        valid_task_ids = {
            task["task_id"]
            for task in tasks
        }

        for task in tasks:

            valid_dependencies = []

            for dependency in task[
                "dependencies"
            ]:

                # Dependency must exist.
                if dependency not in valid_task_ids:
                    continue

                # A task cannot depend on itself.
                if dependency == task["task_id"]:
                    continue

                # Dependencies should point only
                # to earlier tasks.
                if dependency > task["task_id"]:
                    continue

                valid_dependencies.append(
                    dependency
                )

            task["dependencies"] = (
                valid_dependencies
            )

        # ----------------------------------
        # VALIDATE CRITICAL TASK IDS
        # ----------------------------------

        timeline["critical_tasks"] = [
            task_id
            for task_id in timeline.get(
                "critical_tasks",
                []
            )
            if task_id in valid_task_ids
        ]

        return timeline

    except json.JSONDecodeError:

        print(
            "\n❌ Timeline response was not valid JSON."
        )

        return None

    except Exception as error:

        print(
            "\n❌ Error generating event timeline:"
        )


        return None
