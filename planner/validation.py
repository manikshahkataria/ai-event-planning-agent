"""Deterministic checks for extracted event requirements."""

import math
import re
from datetime import date


MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}
MONTHS.update({name[:3]: number for name, number in list(MONTHS.items())})


def _date_problem(value):
    """Accept explicit English dates, ISO dates, and unambiguous numeric dates."""
    value = value.strip()
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    named = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    month_first = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", value
    )
    numeric = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{4})", value)

    try:
        if iso:
            date(*map(int, iso.groups()))
        elif named:
            day, month, year = named.groups()
            date(int(year), MONTHS[month.lower()], int(day))
        elif month_first:
            month, day, year = month_first.groups()
            date(int(year), MONTHS[month.lower()], int(day))
        elif numeric:
            first, _, second, year = numeric.groups()
            first, second, year = int(first), int(second), int(year)
            possibilities = set()
            for month, day in ((first, second), (second, first)):
                try:
                    possibilities.add(date(year, month, day))
                except ValueError:
                    pass
            if len(possibilities) > 1:
                return "The numeric date is ambiguous between day/month and month/day."
            if not possibilities:
                return "The date does not exist in the calendar."
        else:
            return "The date needs an explicit day, month, and four-digit year."
    except (ValueError, KeyError):
        return "The date does not exist in the calendar."
    return None


def validate_requirements(event_state):
    """Return blocking issues without changing the supplied event state."""
    issues = []

    def add_issue(field, reason, question):
        issues.append({
            "field": field,
            "reason": reason,
            "question": question,
            "blocking": True,
        })

    for field in ("event_type", "location", "date"):
        value = event_state.get(field)
        if not isinstance(value, str) or not value.strip():
            add_issue(
                field,
                "A nonempty text value is required.",
                f"Please provide the {field.replace('_', ' ')}.",
            )
        elif field == "date":
            reason = _date_problem(value)
            if reason:
                add_issue(
                    field,
                    reason,
                    f"{reason} What date did you intend? "
                    "Please use YYYY-MM-DD or a month name, such as 15 November 2026.",
                )

    guests = event_state.get("guest_count")
    if type(guests) is not int or guests <= 0:
        add_issue(
            "guest_count",
            "Guest count must be a positive whole number.",
            "How many guests are attending? Please provide a positive whole number.",
        )

    budget = event_state.get("budget")
    if (
        type(budget) not in (int, float)
        or (isinstance(budget, float) and not math.isfinite(budget))
        or budget < 0
    ):
        add_issue(
            "budget",
            "Budget must be a finite, nonnegative number.",
            "What is your total budget? Please provide a finite number of zero or more.",
        )

    return issues
