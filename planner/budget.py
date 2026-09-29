import json
from planner.llm_client import create_reliable_completion


BUDGET_SCHEMA = {
    "type": "object",
    "properties": {
        "budget_strategy": {
            "type": "string"
        },

        "allocations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
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
                    "allocated_budget": {
                        "type": "number"
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "category",
                    "priority",
                    "allocated_budget",
                    "reason"
                ],
                "additionalProperties": False
            }
        },

        "total_allocated": {
            "type": "number"
        },

        "remaining_budget": {
            "type": "number"
        },

        "tradeoffs": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "budget_strategy",
        "allocations",
        "total_allocated",
        "remaining_budget",
        "tradeoffs"
    ],

    "additionalProperties": False
}


def optimize_budget(
    client,
    model,
    event_state,
    event_plan,
    strategy=None,
):
    """
    Analyze the event plan and create an intelligent
    budget allocation based on user requirements
    and preferences.
    """

    system_prompt = """
You are the Budget Intelligence component of an
AI Event Planning Agent.

Your job is to allocate the user's available budget
across the event-plan categories.

You must reason about priorities and explain
trade-offs.

IMPORTANT RULES:

1. Never exceed the user's total budget.

2. Use ONLY categories already present in the
   supplied event plan.

3. Consider:
   - event type
   - number of guests
   - location
   - user preferences
   - practical event needs

4. Assign every category one priority:
   high, medium, or low.

5. User preferences should influence priorities.

Example:

If the user says:
"food and music are important"

Food and entertainment should normally receive
higher priority.

If the user says:
"decoration can be basic"

Decoration should normally receive lower priority.

6. Do not blindly distribute money equally.

7. Explain why each category receives its
   allocation.

8. Explain meaningful trade-offs.

Example:
"Decoration was kept basic so more budget could
be allocated to food."

9. Keep some budget unallocated for contingency
when practical.

10. allocated_budget must always be numeric.

11. total_allocated must not exceed the user's
total budget.

12. Do not recommend specific vendors.

13. Do not invent preferences the user did not
provide.

The result should represent a practical budget,
not just mathematical distribution.
"""

    user_prompt = f"""
EVENT REQUIREMENTS:

{json.dumps(event_state, indent=2)}


CURRENT EVENT PLAN:

{json.dumps(event_plan, indent=2)}


TOTAL AVAILABLE BUDGET:

{event_state["budget"]}


Create an intelligent budget allocation for this
event.

Prioritize categories according to the user's
requirements and preferences.

Explain important trade-offs.
"""

    try:

        if strategy is not None:
            system_prompt += """
Respect the supplied strategy's approach, priorities, bundles and omissions.
Allocate once per event-plan category, using its exact name. A bundle's allocation
covers all its included services; never budget those services again separately.
Do not fund omitted services or treat alternative approaches as extra purchases.
Explain trade-offs without silently dropping an explicit user priority.
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
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "budget_plan",
                    "strict": True,
                    "schema": BUDGET_SCHEMA
                }
            },

        )

        if response is None:
            print(
                "\nBudget Intelligence "
                "could not reach the AI service."
            )
            return None

        raw_response = (
            response.choices[0].message.content
        )

        print("\nRaw Budget Intelligence Response:")
        print(raw_response)

        if not raw_response:

            print(
                "\n❌ Empty budget response."
            )

            return None

        budget_plan = json.loads(
            raw_response
        )

        allocations = budget_plan.get(
            "allocations",
            []
        )

        if strategy is not None:
            selected = {item["category"] for item in strategy["categories"]}
            planned = {item["category"] for item in event_plan["categories"]}
            actual = [item["category"] for item in allocations]
            if planned != selected or set(actual) != selected or len(actual) != len(selected):
                print("\nBudget categories did not match the selected strategy.")
                return None

        # ----------------------------------
        # SAFETY VALIDATION
        # ----------------------------------

        user_budget = float(
            event_state["budget"]
        )

        calculated_total = sum(
            float(item["allocated_budget"])
            for item in allocations
        )

        # Never trust LLM arithmetic blindly.
        budget_plan["total_allocated"] = (
            calculated_total
        )

        budget_plan["remaining_budget"] = (
            user_budget - calculated_total
        )

        # ----------------------------------
        # OVER-BUDGET REPAIR
        # ----------------------------------

        if calculated_total > user_budget:

            print(
                "\n⚠️ Budget intelligence exceeded "
                "the available budget."
            )

            print(
                "Automatically correcting allocations..."
            )

            scale_factor = (
                user_budget / calculated_total
            )

            for item in allocations:

                adjusted = (
                    float(
                        item["allocated_budget"]
                    )
                    * scale_factor
                )

                # Round to nearest ₹100
                item["allocated_budget"] = (
                    round(adjusted / 100) * 100
                )

            # Recalculate after scaling
            calculated_total = sum(
                float(item["allocated_budget"])
                for item in allocations
            )

            # Rounding may leave us slightly
            # above the total budget.
            if calculated_total > user_budget:

                difference = (
                    calculated_total
                    - user_budget
                )

                largest_category = max(
                    allocations,
                    key=lambda item:
                        item["allocated_budget"]
                )

                largest_category[
                    "allocated_budget"
                ] -= difference

            calculated_total = sum(
                float(item["allocated_budget"])
                for item in allocations
            )

            budget_plan[
                "total_allocated"
            ] = calculated_total

            budget_plan[
                "remaining_budget"
            ] = (
                user_budget
                - calculated_total
            )

        return budget_plan

    except json.JSONDecodeError:

        print(
            "\n❌ Budget response was not valid JSON."
        )

        return None

    except Exception as error:

        print(
            "\n❌ Error generating budget intelligence:"
        )

        print(error)

        return None
