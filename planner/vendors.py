"""Geoapify-backed place discovery; no LLM or inferred business attributes."""

import json
from copy import deepcopy
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv


_GEOCODING_URL = "https://api.geoapify.com/v1/geocode/search"
_PLACES_URL = "https://api.geoapify.com/v2/places"
_REQUEST_TIMEOUT = 10
_VENDOR_CATEGORIES = {
    # Documented categories: https://apidocs.geoapify.com/docs/places/
    "restaurant": "catering.restaurant",
    "cafe": "catering.cafe",
    "bar": "catering.bar",
    "pub": "catering.pub",
}


def _get_json(url, params, api_key):
    """Return an API object or None, without logging credential-bearing URLs."""
    try:
        query = urlencode({**params, "apiKey": api_key})
        with urlopen(f"{url}?{query}", timeout=_REQUEST_TIMEOUT) as response:
            data = json.load(response)
        if isinstance(data, dict):
            return data
        print("Geoapify returned an unexpected response format.")
    except HTTPError as error:
        # Exception text and response bodies may include the request/API key.
        print(f"Geoapify request failed (HTTP {error.code}).")
        error.close()
    except (URLError, OSError, TimeoutError):
        print("Geoapify could not be reached or the request timed out.")
    except (ValueError, UnicodeError):
        print("Geoapify returned an unreadable response.")
    return None


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _coordinate(value, bound):
    if type(value) in (int, float) and -bound <= value <= bound:
        return value
    return None


def _geocode_location(location, api_key):
    """Resolve the top matching locality to (longitude, latitude), or None."""
    data = _get_json(_GEOCODING_URL, {
        "text": location, "type": "locality", "format": "json", "limit": 1,
    }, api_key)
    results = data.get("results") if data else None
    if not isinstance(results, list) or not results:
        return None
    place = results[0]
    if not isinstance(place, dict):
        return None
    lon = _coordinate(place.get("lon"), 180)
    lat = _coordinate(place.get("lat"), 90)
    return (lon, lat) if lon is not None and lat is not None else None


def _search_places(longitude, latitude, categories, limit, radius_m, api_key):
    """Fetch provider features within a radius, preserving provider order."""
    data = _get_json(_PLACES_URL, {
        "categories": ",".join(categories),
        "filter": f"circle:{longitude},{latitude},{radius_m}",
        "limit": limit,
    }, api_key)
    features = data.get("features") if data else None
    # None denotes failure/malformed output; [] is a successful empty search.
    return features[:limit] if isinstance(features, list) else None


def _normalize_place(feature):
    """Keep only supplied business information; skip malformed features."""
    if not isinstance(feature, dict):
        return None
    properties = feature.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    contact = properties.get("contact")
    contact = contact if isinstance(contact, dict) else {}
    datasource = properties.get("datasource")
    datasource = datasource if isinstance(datasource, dict) else {}
    raw = datasource.get("raw")
    raw = raw if isinstance(raw, dict) else {}
    categories = properties.get("categories")
    categories = categories if isinstance(categories, list) else []
    lon = _coordinate(properties.get("lon"), 180)
    lat = _coordinate(properties.get("lat"), 90)
    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type") == "Point":
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            if lon is None:
                lon = _coordinate(coordinates[0], 180)
            if lat is None:
                lat = _coordinate(coordinates[1], 90)
    return {
        "name": _text(properties.get("name")),
        "address": _text(properties.get("formatted")),
        "city": _text(properties.get("city")),
        "categories": [value.strip() for value in categories if _text(value)],
        "latitude": lat,
        "longitude": lon,
        "website": (_text(properties.get("website")) or _text(contact.get("website"))
                    or _text(raw.get("website")) or _text(raw.get("contact:website"))),
        "phone": (_text(properties.get("phone")) or _text(contact.get("phone"))
                  or _text(raw.get("phone")) or _text(raw.get("contact:phone"))),
        "source": "Geoapify",
    }


def _normalize_places(features):
    """Normalize a search and deduplicate stable provider IDs within its group."""
    if features is None:
        return {"status": "unavailable", "places": [],
                "message": "Place search unavailable or returned a malformed response."}
    places, seen = [], set()
    malformed = False
    for feature in features:
        place = _normalize_place(feature)
        if place is None:
            malformed = True
            continue
        place_id = _text(feature["properties"].get("place_id"))
        if place_id is not None:
            if place_id in seen:
                continue
            seen.add(place_id)
        places.append(place)
    if places:
        return {"status": "ok", "places": places,
                "message": "Some malformed records were skipped." if malformed else ""}
    if malformed:
        return {"status": "unavailable", "places": [],
                "message": "Place search returned no usable records."}
    return {"status": "empty", "places": [],
            "message": "No places returned within the search radius."}


def recommend_vendors(location, strategy, limit_per_category=5, *, cache=None,
                      radius_m=10000):
    """Return one status/result group per selected strategy category.

    Search only explicit supported vendor_types. Geocode once and reuse identical
    searches within this call. Coverage is intent, never a claim about a place.
    Empty vendor_types means no supported search was requested; it does not prove
    that the event needs no external service. Existing public search_vendors is
    unchanged. All provider failures leave other groups/results intact.
    An optional session-owned cache reuses facts across revisions. Failed searches
    remain unavailable until the cache is cleared or the search inputs change.
    """
    categories = strategy.get("categories") if isinstance(strategy, dict) else None
    if not isinstance(categories, list):
        return [{"category": "Vendor search", "covers": [], "vendor_types": [],
                 "unsupported_types": [], "status": "unavailable", "places": [],
                 "message": "The selected strategy is unavailable or malformed.",
                 "search_action": "not_searched"}]
    groups = []
    searches = []
    for category in categories:
        if not isinstance(category, dict):
            category = {}
        types = category.get("vendor_types")
        covers = category.get("covers")
        group = {
            "category": _text(category.get("category")) or "Unnamed strategy category",
            "covers": [item for item in covers if _text(item)] if isinstance(covers, list) else [],
            "vendor_types": types if isinstance(types, list) else [],
            "unsupported_types": [], "status": "not_required", "places": [],
            "search_action": "not_searched",
            "message": "No supported external place search requested for this category.",
        }
        groups.append(group)
        if not isinstance(types, list) or any(not isinstance(item, str) for item in types):
            group.update(status="unsupported", message="Missing or invalid vendor search types; no search guessed.")
            continue
        group["unsupported_types"] = [item for item in types if item not in _VENDOR_CATEGORIES]
        mapped = tuple(sorted({_VENDOR_CATEGORIES[item] for item in types if item in _VENDOR_CATEGORIES}))
        if not mapped:
            if types:
                group.update(status="unsupported", message="This provider does not support the requested vendor types.")
            continue
        group.update(status="unavailable", message="Vendor discovery is unavailable.")
        searches.append((group, mapped))
    if not searches:
        return groups

    try:
        if (not _text(location) or type(limit_per_category) is not int or not 1 <= limit_per_category <= 500
                or type(radius_m) not in (int, float) or radius_m <= 0
                or (type(radius_m) is float and not math.isfinite(radius_m))):
            return groups
        cache = {} if cache is None else cache
        saved_searches = cache.setdefault("searches", {})
        saved_locations = cache.setdefault("locations", {})
        location_key = ("Geoapify", location.strip())
        pending = []
        for group, mapped in searches:
            search_key = (*location_key, mapped, radius_m, limit_per_category)
            if search_key in saved_searches:
                group.update(deepcopy(saved_searches[search_key]))
                group["search_action"] = "reused"
            else:
                pending.append((group, mapped, search_key))
        if not pending:
            return groups
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        api_key = os.getenv("GEOAPIFY_API_KEY", "").strip()
        if not api_key:
            for group, _, _ in pending:
                group["message"] = "Vendor discovery is unavailable: API key is not configured."
            return groups
        if location_key not in saved_locations:
            saved_locations[location_key] = _geocode_location(location.strip(), api_key)
        coordinates = saved_locations[location_key]
        if coordinates is None:
            for group, _, search_key in pending:
                group["message"] = "Vendor discovery could not resolve the location."
                saved_searches[search_key] = {key: deepcopy(group[key]) for key in ("status", "places", "message")}
            return groups
        for group, mapped, search_key in pending:
            if search_key not in saved_searches:
                try:
                    features = _search_places(*coordinates, mapped, limit_per_category, radius_m, api_key)
                    saved_searches[search_key] = _normalize_places(features)
                except Exception:
                    saved_searches[search_key] = _normalize_places(None)
                group["search_action"] = "searched"
            else:
                group["search_action"] = "reused"
            group.update(deepcopy(saved_searches[search_key]))
        return groups
    except Exception:
        # No exception text: it may contain a URL with credentials.
        return groups


def search_vendors(location: str, categories: list[str], limit: int = 10,
                   *, radius_m: int = 10000) -> list[dict]:
    """Find real places near a locality; return [] on failure or no matches.

    Categories are Geoapify keys, e.g. ['catering.restaurant', 'catering.cafe'].
    The radius defaults to 10 km around the top geocoding match, not the city
    boundary. Qualify ambiguous names with state/country. Limit must be 1..500.
    Environment credentials take precedence over the project's .env file.
    No prices, ratings, availability or budget suitability are inferred.
    The HTTP timeout is per network wait, not an overall operation deadline.
    """
    if (
        not _text(location) or not isinstance(categories, list) or not categories
        or len(categories) > 100
        or any(not isinstance(item, str) or not re.fullmatch(r"[a-z]+(?:[._][a-z0-9]+)*", item)
               for item in categories)
        or type(limit) is not int or not 1 <= limit <= 500
        or type(radius_m) not in (int, float) or radius_m <= 0
        or (type(radius_m) is float and not math.isfinite(radius_m))
    ):
        print("Vendor search requires a location, category keys, limit 1..500 and a positive radius.")
        return []
    try:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        api_key = os.getenv("GEOAPIFY_API_KEY", "").strip()
        if not api_key:
            print("Vendor search is unavailable: GEOAPIFY_API_KEY is not configured.")
            return []
        coordinates = _geocode_location(location.strip(), api_key)
        if coordinates is None:
            print("Vendor search could not resolve the location.")
            return []
        features = _search_places(*coordinates, categories, limit, radius_m, api_key)
        return _normalize_places(features)["places"]
    except Exception:
        # Last-resort provider isolation: never expose URLs, keys or raw errors.
        print("Vendor search is currently unavailable.")
        return []
