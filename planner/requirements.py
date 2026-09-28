import json


REQUIRED_FIELDS = [
    "event_type",
    "location",
    "date",
    "guest_count",
    "budget",
]


EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": ["string", "null"]
        },
        "location": {
            "type": ["string", "null"]
        },
        "date": {
            "type": ["string", "null"]
        },
        "guest_count": {
            "type": ["integer", "null"]
        },
        "budget": {
            "type": ["number", "null"]
        },
        "preferences": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "event_type",
        "location",
        "date",
        "guest_count",
        "budget",
        "preferences"
    ],
    "additionalProperties": False
}


def extract_requirements(
    client,
    model,
    user_message,
    current_state
):
    """
    Extract event requirements from a user message
    while preserving the existing event state.
    """

    system_prompt = """
You are a structured-data extraction component
inside an AI Event Planning Agent.

Extract event information from the user's message.

Never invent information.

Preserve information that already exists in the
current event state unless the user changes it.

Examples:

User:
birthday party in Delhi for 20 people

Result:
event_type = birthday party
location = Delhi
guest_count = 20

User:
my budget is 40000

Result:
budget = 40000

User:
I want something casual with music

Result:
preferences = ["casual", "music"]
"""

    user_prompt = f"""
CURRENT EVENT STATE:

{json.dumps(current_state, indent=2)}

NEW USER MESSAGE:

{user_message}

Update the event state using the new information.
"""

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event_requirements",
                    "strict": True,
                    "schema": EVENT_SCHEMA,
                },
            },

            extra_body={
                "provider": {
                    "require_parameters": True
                }
            },

            # Important for the OpenRouter router:
            # use providers that support our parameters.
            
        )

        raw_response = (
            response.choices[0].message.content
        )

        print("\nRaw AI response:")
        print(raw_response)

        if not raw_response:
            print("\n❌ Empty AI response.")
            return current_state

        extracted = json.loads(raw_response)

        # -----------------------------------
        # MERGE WITH CURRENT STATE
        # -----------------------------------

        updated_state = current_state.copy()

        for field in REQUIRED_FIELDS:

            new_value = extracted.get(field)

            # Only overwrite if the model
            # actually found new information
            if new_value is not None:
                updated_state[field] = new_value

        # Handle preferences separately
        new_preferences = extracted.get(
            "preferences",
            []
        )

        if new_preferences:

            old_preferences = updated_state.get(
                "preferences",
                []
            )

            # Prevent duplicate preferences
            updated_state["preferences"] = list(
                dict.fromkeys(
                    old_preferences + new_preferences
                )
            )

        return updated_state

    except json.JSONDecodeError:

        print(
            "\n❌ The model did not return valid JSON."
        )

        print(
            "Keeping previous event state."
        )

        return current_state

    except Exception as error:

        print("\n❌ API/extraction error:")
        print(error)

        return current_state


def get_missing_fields(event_state):
    """
    Find required information that is
    still missing.
    """

    missing = []

    for field in REQUIRED_FIELDS:

        value = event_state.get(field)

        if value is None or value == "":
            missing.append(field)

    return missing