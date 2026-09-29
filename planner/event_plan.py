import json
from planner.llm_client import create_reliable_completion

EVENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "event_summary": {
            "type": "string"
        },
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string"
                    },
                    "recommendation": {
                        "type": "string"
                    },
                    "estimated_budget": {
                        "type": "number"
                    }
                },
                "required": [
                    "category",
                    "recommendation",
                    "estimated_budget"
                ],
                "additionalProperties": False
            }
        },
        "total_estimated_cost": {
            "type": "number"
        },
        "remaining_budget": {
            "type": "number"
        },
        "planning_notes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "event_summary",
        "categories",
        "total_estimated_cost",
        "remaining_budget",
        "planning_notes"
    ],
    "additionalProperties": False
}


def generate_event_plan(
    client,
    model,
    event_state,
    strategy=None,
):
    """
    Generate a structured event plan using
    the completed event requirements.
    """

    system_prompt = """
You are the planning component of an AI Event Planning Agent.

Convert the completed event requirements into a
practical high-level event plan.

Create only categories that make sense for the event.

Possible categories include:
- Venue
- Food and catering
- Decoration
- Entertainment
- Cake
- Photography
- Invitations
- Transportation
- Miscellaneous

Rules:

1. Respect the user's total budget.
2. Estimated costs should not exceed the total budget.
3. Keep recommendations realistic and practical.
4. Consider event type, location, guest count,
   date and preferences.
5. estimated_budget must be numeric.
6. Do not recommend specific vendors yet.
7. Do not invent user preferences.
8. Keep some money for contingency when practical.
"""

    user_prompt = f"""
EVENT REQUIREMENTS:

{json.dumps(event_state, indent=2)}

Create a high-level event plan.
"""

    try:

        if strategy is not None:
            system_prompt += """
Follow the supplied event strategy's approach, priorities, omissions and selected
categories. Use exactly its category names. Explain bundled services in each
recommendation; do not add separate categories for services already covered.
Do not reinstate omitted services or silently remove explicitly requested ones.
Treat alternatives as alternatives, not additional purchases. Include relevant
assumptions and trade-offs in planning_notes for downstream execution planning.
"""
            user_prompt += "\nEVENT STRATEGY:\n" + json.dumps(strategy, indent=2)

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
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event_plan",
                    "strict": True,
                    "schema": EVENT_PLAN_SCHEMA,
                },
            },
        )

        if response is None:
            print(
                "\n⚠️ Event plan generation "
                "could not reach the AI service."
            )
            return None

        raw_response = (
            response.choices[0].message.content
        )

        print("\nRaw Event Plan Response:")
        print(raw_response)

        if not raw_response:
            print(
                "\n❌ Empty event plan response."
            )
            return None

        event_plan = json.loads(raw_response)

        # ----------------------------------
        # CALCULATE ORIGINAL TOTAL
        # ----------------------------------

        categories = event_plan.get(
            "categories",
            []
        )

        if strategy is not None:
            selected = {item["category"] for item in strategy["categories"]}
            actual = [item["category"] for item in categories]
            if set(actual) != selected or len(actual) != len(selected):
                print("\nEvent plan categories did not match the selected strategy.")
                return None

        total_cost = sum(
            item["estimated_budget"]
            for item in categories
        )

        user_budget = event_state["budget"]

        # ----------------------------------
        # BUDGET SAFETY CHECK
        # ----------------------------------

        if total_cost > user_budget:

            print(
                "\n⚠️ Generated plan exceeded budget."
            )

            print(
                "Automatically adjusting allocations..."
            )

            scale_factor = (
                user_budget / total_cost
            )

            for item in categories:

                adjusted_budget = (
                    item["estimated_budget"]
                    * scale_factor
                )

                # Round to nearest ₹100
                adjusted_budget = round(
                    adjusted_budget / 100
                ) * 100

                item["estimated_budget"] = (
                    adjusted_budget
                )

        # ----------------------------------
        # CALCULATE FINAL TOTAL
        # ----------------------------------

        final_total = sum(
            item["estimated_budget"]
            for item in categories
        )

        # Rounding may leave the plan
        # slightly over budget.
        if final_total > user_budget:

            difference = (
                final_total - user_budget
            )

            largest_category = max(
                categories,
                key=lambda item:
                    item["estimated_budget"]
            )

            largest_category[
                "estimated_budget"
            ] -= difference

        # Calculate one final time
        final_total = sum(
            item["estimated_budget"]
            for item in categories
        )

        event_plan[
            "total_estimated_cost"
        ] = final_total

        event_plan[
            "remaining_budget"
        ] = (
            user_budget - final_total
        )

        return event_plan

    except json.JSONDecodeError:

        print(
            "\n❌ Event plan was not valid JSON."
        )

        return None

    except Exception as error:

        print(
            "\n❌ Error generating event plan:"
        )

        print(error)

        return None
