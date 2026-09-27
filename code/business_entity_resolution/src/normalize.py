"""
Business Entity Resolution - Normalization Module

This module provides normalization functions for business names and addresses
to enable consistent entity resolution across multiple data sources.
"""

import re
import string
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import pandas as pd

import config


# ─── Constants ────────────────────────────────────────────────────────────────

# Legal suffix canonicalization mapping
# Maps variations to a single canonical form
SUFFIX_MAP: Dict[str, str] = {
    # Private / Pvt variations
    "pvt": "private",
    "pvt.": "private",
    "private": "private",
    "privat": "private",
    # Limited / Ltd variations
    "ltd": "limited",
    "ltd.": "limited",
    "limited": "limited",
    "ltda": "limited",
    # Corporation / Corp variations
    "corp": "corporation",
    "corp.": "corporation",
    "corporation": "corporation",
    "corporated": "corporation",
    # Incorporated / Inc variations
    "inc": "incorporated",
    "inc.": "incorporated",
    "incorporated": "incorporated",
    "incorp": "incorporated",
    # Limited Liability Company
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "llc.": "llc",
    # Limited Liability Partnership
    "llp": "llp",
    "l.l.p.": "llp",
    "l.l.p": "llp",
    # Company
    "co": "company",
    "co.": "company",
    "company": "company",
    # And / & variations
    "&": "and",
    "and": "and",
    # Group
    "grp": "group",
    "grp.": "group",
    "group": "group",
    # Holdings
    "hldgs": "holdings",
    "holdings": "holdings",
    # International
    "intl": "international",
    "intl.": "international",
    "international": "international",
    # Technologies / Tech
    "tech": "technologies",
    "tech.": "technologies",
    "technologies": "technologies",
    "technology": "technologies",
    # Solutions
    "sol": "solutions",
    "sol.": "solutions",
    "solutions": "solutions",
    # Services
    "svc": "services",
    "svc.": "services",
    "services": "services",
    # Systems
    "sys": "systems",
    "sys.": "systems",
    "systems": "systems",
    # Enterprises
    "ent": "enterprises",
    "ent.": "enterprises",
    "enterprises": "enterprises",
    # Associates
    "assoc": "associates",
    "assoc.": "associates",
    "associates": "associates",
    # Partners / Partnership
    "ptr": "partners",
    "partners": "partners",
    "partnership": "partners",
    # Corporation (French)
    "s.a.": "sa",
    "sa": "sa",
    "s.a.s.": "sas",
    "sas": "sas",
    "s.a.r.l.": "sarl",
    "sarl": "sarl",
    # Corporation (German)
    "gmbh": "gmbh",
    "g.m.b.h.": "gmbh",
    "ag": "ag",
    # Corporation (Indian)
    "pvt ltd": "private limited",
    "private limited": "private limited",
}


# Address abbreviation canonicalization mapping
ADDRESS_ABBREV_MAP: Dict[str, str] = {
    # Street types
    "rd": "road",
    "rd.": "road",
    "road": "road",
    "st": "street",
    "st.": "street",
    "street": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "av.": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "boulevard": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "drive": "drive",
    "ln": "lane",
    "ln.": "lane",
    "lane": "lane",
    "ct": "court",
    "ct.": "court",
    "court": "court",
    "pl": "place",
    "pl.": "place",
    "place": "place",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "parkway": "parkway",
    "cir": "circle",
    "cir.": "circle",
    "circle": "circle",
    "trl": "trail",
    "trl.": "trail",
    "trail": "trail",
    "hwy": "highway",
    "hwy.": "highway",
    "highway": "highway",
    "expy": "expressway",
    "expy.": "expressway",
    "expressway": "expressway",
    # Unit types
    "apt": "apartment",
    "apt.": "apartment",
    "apartment": "apartment",
    "unit": "unit",
    "ste": "suite",
    "ste.": "suite",
    "suite": "suite",
    "fl": "floor",
    "fl.": "floor",
    "floor": "floor",
    "bldg": "building",
    "bldg.": "building",
    "building": "building",
    # Directions
    "n": "north",
    "n.": "north",
    "north": "north",
    "s": "south",
    "s.": "south",
    "south": "south",
    "e": "east",
    "e.": "east",
    "east": "east",
    "w": "west",
    "w.": "west",
    "west": "west",
    "ne": "northeast",
    "ne.": "northeast",
    "northeast": "northeast",
    "nw": "northwest",
    "nw.": "northwest",
    "northwest": "northwest",
    "se": "southeast",
    "se.": "southeast",
    "southeast": "southeast",
    "sw": "southwest",
    "sw.": "southwest",
    "southwest": "southwest",
    # Common abbreviations
    "nr": "near",
    "nr.": "near",
    "near": "near",
    "opp": "opposite",
    "opp.": "opposite",
    "opposite": "opposite",
    "behind": "behind",
    "adj": "adjacent",
    "adj.": "adjacent",
    "adjacent": "adjacent",
    "beside": "beside",
    # Note: no "next" -> "next to" entry - it doubled to "next to to" on
    # addresses that already read "next to ..."
    "nr to": "near to",
    # Postal / region
    "po": "post office",
    "po box": "post office box",
    "p.o.": "post office",
    "p.o. box": "post office box",
    "zip": "zip code",
    # French address terms
    "r": "rue",
    "r.": "rue",
    "rue": "rue",
    "av": "avenue",
    "bd": "boulevard",
    "bd.": "boulevard",
    "ch": "chemin",
    "ch.": "chemin",
    "chemin": "chemin",
    "pl": "place",
    "rte": "route",
    "rte.": "route",
    "route": "route",
    "all": "allee",
    "all.": "allee",
    "allee": "allee",
    "imp": "impasse",
    "imp.": "impasse",
    "impasse": "impasse",
    # Indian address terms
    "kh": "khasra",
    "kh.": "khasra",
    "khasra": "khasra",
    "no": "number",
    "no.": "number",
    "number": "number",
    "col": "colony",
    "col.": "colony",
    "colony": "colony",
    "sec": "sector",
    "sec.": "sector",
    "sector": "sector",
    "ph": "phase",
    "ph.": "phase",
    "phase": "phase",
    "blk": "block",
    "blk.": "block",
    "block": "block",
    "flr": "floor",
    "flr.": "floor",
}


# Landmark detection keywords (case-insensitive)
# No duplicates: _extract_landmark resolves overlaps positionally, but repeated
# entries used to cause double removal and duplicated landmark text.
LANDMARK_KEYWORDS: List[str] = [
    "near",
    "opposite",
    "opp",
    "behind",
    "adjacent to",
    "adjacent",
    "beside",
    "next to",
    "next",
    "close to",
    "close",
    "across from",
    "across",
    "facing",
    "by",
    "at",
    "landmark",
    "reference",
    "in front of",
    "nearby",
]


# Regex patterns for extracting address components
# Country-agnostic patterns. Bare digit runs are only accepted at the end of a
# comma-separated segment so 5-digit house numbers are not mistaken for ZIPs.
POSTAL_CODE_PATTERNS: List[str] = [
    r"\b\d{5}(?:-\d{4})?\b(?=\s*(?:,|$))",   # US ZIP / ZIP+4 at segment end
    r"\b\d{6}\b(?=\s*(?:,|$))",               # India PIN at segment end
]

# Country-specific patterns, tried first when the record's country is known.
# Matched case-insensitively (input is lowercased before extraction).
COUNTRY_POSTAL_CODE_PATTERNS: Dict[str, List[str]] = {
    "france": [r"\b\d{5}\b(?=\s)"],                       # 75016 paris
    "india": [r"\b\d{6}\b"],                               # 400001 mumbai
    "canada": [r"\b[a-z]\d[a-z]\s?\d[a-z]\d\b"],           # m5h 2n2
    "netherlands": [r"\b\d{4}\s?[a-z]{2}\b"],              # 1012 ab
    "poland": [r"\b\d{2}-\d{3}\b"],                        # 00-001
}


# State/region patterns (generic)
# US state codes (2 letters) + full names
US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia"
]
# US state codes (2 letters) - EXCLUDE those that conflict with street abbreviations
# Conflicts: ct (court), fl (floor) - use full names "connecticut", "florida" instead
US_STATE_CODES = [
    "al", "ak", "az", "ar", "ca", "co", "de", "ga", "hi", "id", "il", "in",
    "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv",
    "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn",
    "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"
]

# Indian states and union territories (full names only)
INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "jammu", "kashmir",
    "ladakh", "chandigarh", "puducherry", "lakshadweep", "dadra", "nagar haveli",
    "daman", "diu",
]
# Indian state codes - matched only in the same positional slots as US codes
# (after/before a comma or at the end of the string) to avoid false positives
# such as "and/or" matching "or".
INDIAN_STATE_CODES = [
    "an", "ap", "ar", "as", "br", "cg", "ch", "dd", "dl", "dn", "ga", "gj",
    "hr", "hp", "jh", "jk", "ka", "kl", "ld", "mh", "ml", "mn", "mp", "mz",
    "nl", "or", "pb", "py", "rj", "sk", "tg", "tn", "tr", "up", "ut", "wb",
]

# Two-letter codes used by the positional pass of _extract_state
STATE_CODES: List[str] = US_STATE_CODES + INDIAN_STATE_CODES

# Build combined state patterns (case-insensitive matching)
ALL_STATES = US_STATES + US_STATE_CODES + INDIAN_STATES + INDIAN_STATE_CODES
STATE_PATTERNS: List[str] = [
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(ALL_STATES, key=len, reverse=True)) + r")\b",
]

# Last-token vocabulary used to reject street lines as city candidates
STREET_TYPES: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "av", "boulevard", "blvd",
    "drive", "dr", "lane", "ln", "court", "ct", "place", "pl", "parkway",
    "pkwy", "circle", "cir", "trail", "trl", "highway", "hwy", "expressway",
    "expy", "way", "close", "crescent", "terrace", "square", "row", "route",
    # French street types. Needed because _extract_city runs on the text BEFORE
    # abbreviation expansion, so both the abbreviation and the full word must be
    # listed. Measured on 209k France rows: 'rue' appears in 65.8% of addresses
    # and was being returned as the city for 17,108 of them.
    "rue", "bd", "impasse", "chemin", "allee", "quai", "cours", "passage",
    "ruelle", "sentier", "voie", "esplanade", "faubourg", "hameau", "villa",
    "lotissement", "rond", "parvis", "promenade", "traverse",
}


# French administrative divisions (regions + departments) must never be taken as
# a city. On this dataset they dominate the trailing comma-part of a French
# address, so before this list existed ~52% of the 1.69M French rows stored
# 'hauts-de-france' / 'nord' / 'gironde' as their city - a near-constant value
# that carries no discriminative signal for blocking.
#
# Matching is on the whole candidate string, so a city that merely contains a
# division name is unaffected: 'calais' stays a city even though the department
# is 'pas-de-calais'.
#
# 'paris' is deliberately absent: it is both the department (75) and France's
# largest city, and rejecting it erased the city from every Paris address.
# Every other entry is a division with no major city of the same name.
FRENCH_REGIONS: Set[str] = {
    "auvergne-rhone-alpes", "auvergne rhone alpes", "borgogne-franche-comte",
    "bretagne", "centre-val de loire", "centre val de loire", "corse",
    "grand-est", "grand est", "hauts-de-france", "ile-de-france", "ile de france",
    "normandie", "nouvelle-aquitaine", "occitanie", "pays de la loire",
    "provence-alpes-cote d'azur", "provence alpes cote d azur",
    # pre-2016 region names, still present in older address data
    "aquitaine", "auvergne", "basse-normandie", "champagne-ardenne",
    "franche-comte", "haute-normandie", "languedoc-roussillon", "lorraine",
    "midi-pyrenees", "picardie", "poitou-charentes", "rhone-alpes",
    # overseas regions / departments
    "guadeloupe", "guyane", "la reunion", "martinique", "mayotte",
}

FRENCH_DEPARTMENTS: Set[str] = {
    "ain", "aisne", "allier", "alpes-de-haute-provence", "hautes-alpes",
    "alpes-maritimes", "ardèche", "ardennes", "ariège", "aube", "aude", "aveyron",
    "bouches-du-rhone", "calvados", "cantal", "charente", "charente-maritime",
    "cher", "corrèze", "corse-du-sud", "haute-corse", "côte-d'or", "cotes-d'armor",
    "creuse", "dordogne", "doubs", "drôme", "eure", "eure-et-loir", "finistère",
    "gard", "haute-garonne", "gers", "gironde", "hérault", "ille-et-vilaine",
    "indre", "indre-et-loire", "isère", "jura", "landes", "loir-et-cher", "loire",
    "haute-loire", "loire-atlantique", "loiret", "lot", "lot-et-garonne", "lozère",
    "maine-et-loire", "manche", "marne", "haute-marne", "mayenne",
    "meurthe-et-moselle", "meuse", "morbihan", "moselle", "nièvre", "nord", "oise",
    "orne", "pas-de-calais", "puy-de-dôme", "pyrénees-atlantiques",
    "hautes-pyrénées", "pyrénees-orientales", "bas-rhin", "haut-rhin", "rhône",
    "haute-saône", "saône-et-loire", "sarthe", "savoie", "haute-savoie",
    "seine-maritime", "seine-et-marne", "yvelines", "deux-sèvres", "somme",
    "tarn", "tarn-et-garonne", "var", "vaucluse", "vendée", "vienne",
    "haute-vienne", "vosges", "yonne", "territoire de belfort", "essonne",
    "hauts-de-seine", "seine-saint-denis", "val-de-marne", "val-d'oise",
}

_ADMIN_DIVISIONS: Set[str] = FRENCH_REGIONS | FRENCH_DEPARTMENTS


# ─── Precompiled Patterns ─────────────────────────────────────────────────────
# Built once so per-record calls never rebuild or re-sort pattern strings.
# Single combined patterns are behavior-equivalent to the per-key/per-keyword
# loops they replace (longest alternative first, same boundary rules).

_WS_RE = re.compile(r"\s+")
_DOTTED_INITIALISM_RE = re.compile(r"\b(?:[a-z]\.){2,}", re.IGNORECASE)
_STRIP_COMMA_RE = re.compile(r"[^\w\s\-,]")
_STRIP_RE = re.compile(r"[^\w\s\-]")
_DIGITS_RE = re.compile(r"\d+")
_ORPHAN_HYPHEN_RE = re.compile(r"(?:(?<=\s)-|-(?=\s))")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


@lru_cache(maxsize=1)
def _unicode_mark_class() -> str:
    """Regex character-class body covering every Unicode mark (category M).

    Python's ``\\w`` does not match category-M codepoints, so a negated class
    like ``[^\\w\\s\\-]`` silently deletes Indic matras, viramas and Arabic
    harakat. ``re`` has no ``\\p{M}``, so the mark ranges are derived once from
    the ``unicodedata`` tables and cached for the process lifetime.
    """
    ranges: List[tuple] = []
    start: Optional[int] = None
    for cp in range(0x110000):
        if unicodedata.category(chr(cp)).startswith("M"):
            if start is None:
                start = cp
        elif start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    body = []
    for lo, hi in ranges:
        body.append(re.escape(chr(lo)) if lo == hi
                    else f"{re.escape(chr(lo))}-{re.escape(chr(hi))}")
    return "".join(body)


@lru_cache(maxsize=2)
def _strip_pattern(preserve_commas: bool):
    """Punctuation-strip pattern that also preserves Unicode marks.

    Compiled lazily because building the mark class walks the whole code space
    once; ASCII-only input never triggers it (see ``_strip_punctuation``).
    """
    keep = "\\w\\s\\-" + ("," if preserve_commas else "")
    # No separator is inserted before the mark body on purpose: every keep char
    # above is either a class escape (\w, \s) or an escaped literal (\-), so the
    # body's leading range cannot be absorbed into one. A bare ',' here would be
    # a literal inside the class and would silently keep every comma.
    return re.compile("[^" + keep + _unicode_mark_class() + "]")

_LANDMARK_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(LANDMARK_KEYWORDS, key=len, reverse=True))
    + r")\b\s+([^,;.]+)",
    re.IGNORECASE,
)

_FULL_STATE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(US_STATES + INDIAN_STATES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_STATE_CODE_JOINED = "|".join(re.escape(s) for s in sorted(STATE_CODES, key=len, reverse=True))
_STATE_CODE_RE = re.compile(r"\b(?:" + _STATE_CODE_JOINED + r")\b", re.IGNORECASE)
_STATE_AFTER_COMMA_RE = re.compile(r",\s*(\b(?:" + _STATE_CODE_JOINED + r")\b)\b", re.IGNORECASE)
_STATE_AT_END_RE = re.compile(r"\b(?:" + _STATE_CODE_JOINED + r")\b\s*$", re.IGNORECASE)
_STATE_BEFORE_COMMA_RE = re.compile(r"\b(\b(?:" + _STATE_CODE_JOINED + r")\b)\s*,", re.IGNORECASE)

_POSTAL_PATTERNS_COMPILED: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in POSTAL_CODE_PATTERNS]
_COUNTRY_POSTAL_PATTERNS_COMPILED: Dict[str, List[re.Pattern]] = {
    name: [re.compile(p, re.IGNORECASE) for p in pats] for name, pats in COUNTRY_POSTAL_CODE_PATTERNS.items()
}

# Per-mapping single-pass substitution: one combined regex instead of one
# re.sub per key (100+ passes over the string per record)
_MAPPING_COMBINED_CACHE: Dict[tuple, tuple] = {}

# Compiled state-inside-part patterns, cached by state string
_STATE_IN_PART_CACHE: Dict[str, re.Pattern] = {}


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return _WS_RE.sub(" ", text).strip()


def _strip_punctuation(text: str, preserve_commas: bool = False) -> str:
    """Remove punctuation while preserving alphanumeric and spaces.

    Canonicalizes constructs that plain stripping would destroy:
    - '&' becomes 'and' (otherwise it is deleted entirely)
    - dotted single-letter initialisms ('l.l.c.', 's.a.', 'p.o.') are merged
      so they stay matchable by the suffix/abbreviation maps

    Unicode marks (category M - Indic matras and viramas, Arabic harakat) are
    kept, because Python's ``\\w`` excludes them and a plain ``[^\\w\\s]`` class
    deletes them outright, shattering 'प्राइवेट' into 'प र इव ट'. Input is
    NFC-normalized first so canonically equivalent spellings agree.
    """
    text = text.replace("&", " and ")
    text = _DOTTED_INITIALISM_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    if text.isascii():
        # Fast path: no marks exist in pure ASCII, so the class is identical.
        return (_STRIP_COMMA_RE if preserve_commas else _STRIP_RE).sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    return _strip_pattern(preserve_commas).sub(" ", text)


def _tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return text.split()


def _apply_mapping(text: str, mapping: Dict[str, str], word_boundary: bool = True) -> str:
    """
    Apply a mapping dictionary to text, replacing keys with values in a
    single pass over a combined regex (longest key first), instead of one
    re.sub per key.

    Boundaries: lookbehind (?<!\\w) and lookahead (?![\\w-]). The right-hand
    side also blocks hyphens so 'co-op' is not turned into 'company-op',
    while keys like '&' or 'pvt.' (which cannot satisfy \\b...\\b) still match.
    """
    cache_key = (id(mapping), word_boundary)
    entry = _MAPPING_COMBINED_CACHE.get(cache_key)
    if entry is None:
        keys = sorted(mapping.keys(), key=len, reverse=True)
        body = "|".join(re.escape(k) for k in keys)
        if word_boundary:
            pattern = re.compile(r"(?<!\w)(?:" + body + r")(?![\w-])", re.IGNORECASE)
        else:
            pattern = re.compile(body, re.IGNORECASE)
        replacements = {k.lower(): v for k, v in mapping.items()}
        entry = (pattern, replacements, mapping)
        _MAPPING_COMBINED_CACHE[cache_key] = entry
    pattern, replacements, _mapping_ref = entry
    return pattern.sub(lambda m: replacements.get(m.group(0).lower(), m.group(0)), text)


def _extract_landmark(address: str) -> tuple[str, str]:
    """
    Detect and extract landmark phrases from address.
    Returns (core_address, landmark_text).

    One combined pattern (alternatives ordered longest-first) finds all
    non-overlapping matches left-to-right, so removal offsets always stay
    valid and no span is ever extracted twice.
    """
    selected = list(_LANDMARK_RE.finditer(address))

    if not selected:
        return _normalize_whitespace(address), ""

    core_parts: List[str] = []
    prev_end = 0
    for match in selected:
        core_parts.append(address[prev_end:match.start()])
        prev_end = match.end()
    core_parts.append(address[prev_end:])

    core_address = _normalize_whitespace("".join(core_parts))
    landmark_text = "; ".join(_normalize_whitespace(m.group(0)) for m in selected)
    return core_address, landmark_text


def _extract_postal_code(text: str, country: Optional[str] = None) -> Optional[str]:
    """Extract postal code from text.

    When the country is known only its patterns plus the country-agnostic
    segment-boundary patterns are tried (keeps 5-digit US house numbers from
    being returned as ZIPs). When unknown, every known pattern is tried.
    """
    country_l = (country or "").strip().lower()
    patterns: List[re.Pattern] = []
    if country_l:
        for name, compiled in _COUNTRY_POSTAL_PATTERNS_COMPILED.items():
            if country_l.startswith(name):
                patterns.extend(compiled)
    else:
        for compiled in _COUNTRY_POSTAL_PATTERNS_COMPILED.values():
            patterns.extend(compiled)
    patterns.extend(_POSTAL_PATTERNS_COMPILED)

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _extract_french_region(text: str) -> Optional[str]:
    """Return the French region/department acting as the state, if present.

    French rows have no US-style state code, so the code-based passes must not
    run for them: 'la' in "La Teste-de-Buch" matched Louisiana, and 'or'/'ca'/
    'in' match ordinary French words. Returning the real division instead gives
    _extract_city the anchor it needs to prefer the city over the region.
    """
    for part in text.split(","):
        candidate = part.strip()
        if candidate and _is_admin_division(candidate):
            return candidate.lower()
    return None


def _extract_state(text: str, country: Optional[str] = None) -> Optional[str]:
    """Extract state/region from text."""
    if (country or "").strip().lower().startswith("france"):
        return _extract_french_region(text)

    # First try full state names (more reliable), skipping names that are
    # really street names ("2213 Michigan Street" -> not the state of Michigan)
    for match in _FULL_STATE_RE.finditer(text):
        remainder = text[match.end():].strip()
        token_match = re.match(r"\w+", remainder)
        next_token = token_match.group(0).lower() if token_match else ""
        if next_token in STREET_TYPES:
            continue
        return match.group(0).lower()

    # Then try state codes (US + India), but only in typical positions:
    # 1. After a comma (", tx" or ", texas")
    # 2. At the end of the string
    # 3. Before a comma (less common but possible)
    # Positional matching avoids false positives like "and/or" -> "or".
    match = _STATE_AFTER_COMMA_RE.search(text)
    if match:
        return match.group(1).lower()

    match = _STATE_AT_END_RE.search(text)
    if match:
        return match.group(0).lower()

    match = _STATE_BEFORE_COMMA_RE.search(text)
    if match:
        return match.group(1).lower()

    return None


# Tokens that mark a part as an address fragment rather than a city name
JUNK_PART_TOKENS: Set[str] = {
    "hno", "sno", "srno", "no", "opp", "behind", "near", "ward", "floor",
    "block", "phase", "sector", "colony", "typ", "kh", "blk", "sec", "ph",
    "cross", "main", "house", "building", "gali", "mohalla", "nagar", "chowk",
    "tower", "circle", "complex", "market", "centre", "center",
    # French unit / building descriptors, which otherwise become the city
    "residence", "appartement", "batiment", "etage", "rez", "za", "zi", "zac",
    "lieu", "dit", "immUBLE", "local", "atelier",
}


def _is_meaningful_part(text: str) -> bool:
    """True if a comma-separated part looks like a name rather than a number."""
    if len(text) <= 2 or text.isdigit():
        return False
    tokens = text.split()
    if any(tok in JUNK_PART_TOKENS for tok in tokens):
        return False
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin_chars == 0:
        return False
    return latin_chars / max(len(text), 1) >= 0.3


def _looks_like_street(text: str) -> bool:
    """True if any token is a street-type word, i.e. the part is a street line.

    A single whole-part check covers both placements seen in the data: the
    street word trailing the name ('rue de foo', 'avenue de dunkerque',
    '5 bis rue pierre dignac') and leading it ('bd du president wilson'). An
    earlier version inspected only the last two or three tokens, which let
    'rue' through for 17,108 of 60k French rows.
    """
    tokens = text.split()
    if not tokens:
        return False
    return any(tok in STREET_TYPES for tok in tokens)


@lru_cache(maxsize=512)
def _trailing_state_code_re(state: str) -> re.Pattern:
    """Matches ' <state-code>' at the end of a candidate, cached per state."""
    return re.compile(r"\s+\b" + re.escape(state) + r"$", re.IGNORECASE)


def _clean_city_candidate(
    part: str,
    state: Optional[str] = None,
    postal: Optional[str] = None,
    country: Optional[str] = None,
) -> str:
    """Strip postal code, digits, dangling hyphens and a trailing state code.

    A full state/country name is only dropped when the candidate IS exactly it,
    never when embedded, since cities like "central delhi" or "michigan city"
    legitimately contain those words. A trailing two-letter state code is the
    exception: the code never belongs to a city name, so "austin tx" is trimmed
    to "austin" rather than being kept as a second spelling of the same city.
    """
    out = part
    if postal:
        out = out.replace(postal, " ")
    out = _DIGITS_RE.sub(" ", out)
    out = _ORPHAN_HYPHEN_RE.sub(" ", out)
    out = _normalize_whitespace(out).strip(" -")
    if state and len(state) == 2:
        match = _trailing_state_code_re(state).search(out)
        if match and match.start() > 0:
            out = out[:match.start()].strip()
    if state and out.lower() == state.lower():
        return ""
    if country and out.lower() == country.lower():
        return ""
    return out


def _fold_admin_key(text: str) -> str:
    """Accent- and punctuation-insensitive key for admin-division comparison.

    Lets 'côte-d'or', "cote d'or" and 'Cote d Or' all resolve to one entry.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM_RE.sub(" ", stripped).strip()


@lru_cache(maxsize=300_000)
def _is_admin_division(candidate: str) -> bool:
    """True if the candidate is a French region or department, not a city."""
    return _fold_admin_key(candidate) in _ADMIN_DIVISION_KEYS


_ADMIN_DIVISION_KEYS: Set[str] = {_fold_admin_key(a) for a in _ADMIN_DIVISIONS}


def _extract_city(
    text: str,
    state: Optional[str] = None,
    postal: Optional[str] = None,
    country: Optional[str] = None,
) -> Optional[str]:
    """
    Extract city from address text.

    Strategy:
    1. If the state is inside a longer part, that part is the city
       ("new delhi" contains "delhi").
    2. Otherwise take the closest valid part before the state part
       ("Kolkata, Howrah, West Bengal").
    3. If nothing valid precedes it, scan from the end of the address, which
       covers reversed formats ("OH, Columbus, 5559 Orville Avenue") and
       state-first prefixes ("West Bengal, ... Calcutta, Kolkata").
    4. With no state at all, take the last valid part.

    Street lines, bare numbers, address fragments, the state/country itself and
    French regions/departments are never returned; skipping an admin division
    keeps scanning, so "... Lille, Hauts-de-France" still yields "lille".
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]

    def valid(candidate: str) -> bool:
        if not candidate or not _is_meaningful_part(candidate) or _looks_like_street(candidate):
            return False
        return not _is_admin_division(candidate)

    def state_re(s: str) -> re.Pattern:
        compiled = _STATE_IN_PART_CACHE.get(s)
        if compiled is None:
            compiled = re.compile(r"\b" + re.escape(s.lower()) + r"\b", re.IGNORECASE)
            _STATE_IN_PART_CACHE[s] = compiled
        return compiled

    state_indices: List[int] = []
    if state:
        pattern = state_re(state)
        state_indices = [i for i, part in enumerate(parts) if pattern.search(part)]

    if state_indices:
        last_state_idx = state_indices[-1]
        last_state_part = parts[last_state_idx].strip()

        # "new delhi" / "austin tx" - the part is city (+state). Cleaned like
        # every other candidate so a trailing postal code or state code cannot
        # survive into the city field.
        if last_state_part.lower() != state.lower() and state_re(state).search(last_state_part):
            candidate = _clean_city_candidate(last_state_part, state, postal, country)
            if valid(candidate):
                return candidate

        for j in range(last_state_idx - 1, -1, -1):
            candidate = _clean_city_candidate(parts[j], state, postal, country)
            if valid(candidate):
                return candidate

        for j in range(len(parts) - 1, -1, -1):
            if j == last_state_idx:
                continue
            candidate = _clean_city_candidate(parts[j], state, postal, country)
            if valid(candidate):
                return candidate
        return None

    for j in range(len(parts) - 1, -1, -1):
        candidate = _clean_city_candidate(parts[j], state, postal, country)
        if valid(candidate):
            return candidate
    return None


# ─── Main Normalization Functions ─────────────────────────────────────────────


def normalize_name(raw_name: str) -> Dict[str, Any]:
    """
    Normalize a business name.

    Steps:
    1. Lowercase the input
    2. Strip punctuation (keeping hyphens; '&' becomes 'and', dotted
       initialisms such as 'l.l.c.' are merged so they stay matchable)
    3. Normalize whitespace
    4. Apply legal suffix canonicalization mapping
    5. Tokenize the result

    Args:
        raw_name: Original business name string

    Returns:
        Dictionary with keys:
        - 'raw_name': Original unchanged string
        - 'normalized_name': Fully normalized string
        - 'name_tokens': List of normalized tokens
    """
    if not raw_name or not isinstance(raw_name, str):
        return {
            "raw_name": raw_name or "",
            "normalized_name": "",
            "name_tokens": [],
        }

    # Preserve original
    original = raw_name

    # Step 1: Lowercase
    normalized = raw_name.lower()

    # Step 2: Strip punctuation (keep alphanumeric, spaces, hyphens)
    normalized = _strip_punctuation(normalized)

    # Step 3: Normalize whitespace
    normalized = _normalize_whitespace(normalized)

    # Step 4: Apply suffix mapping
    normalized = _apply_mapping(normalized, SUFFIX_MAP, word_boundary=True)

    # Step 5: Final whitespace normalization
    normalized = _normalize_whitespace(normalized)

    # Step 6: Tokenize
    tokens = _tokenize(normalized)

    return {
        "raw_name": original,
        "normalized_name": normalized,
        "name_tokens": tokens,
    }


def normalize_address(raw_address: str, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize a business address.

    Steps:
    1. Lowercase the input
    2. Strip punctuation (preserving alphanumeric, hyphens, commas for extraction)
    3. Detect and extract landmark phrases ONCE from the comma-preserved text
    4. Extract postal code, state/region, and city from the landmark-free core
    5. Derive normalized_address from that SAME core (comma-free), so the
       removed landmark text always matches the returned 'landmark' field
    6. Apply address abbreviation canonicalization

    Args:
        raw_address: Original business address string
        country: Record's country, used to pick postal-code patterns and to
            keep the country name out of the extracted city

    Returns:
        Dictionary with keys:
        - 'raw_address': Original unchanged string
        - 'normalized_address': Core address (landmark removed, normalized)
        - 'landmark': Extracted landmark text (empty if none)
        - 'city': Extracted city name or None
        - 'state': Extracted state/region or None
        - 'postal_code': Extracted postal code or None
    """
    if not raw_address or not isinstance(raw_address, str):
        return {
            "raw_address": raw_address or "",
            "normalized_address": "",
            "landmark": "",
            "city": None,
            "state": None,
            "postal_code": None,
        }

    # Preserve original
    original = raw_address

    # Step 1: Lowercase
    normalized = raw_address.lower()

    # Step 2: Strip punctuation but KEEP commas for city/state/postal extraction
    normalized_for_extraction = _strip_punctuation(normalized, preserve_commas=True)
    normalized_for_extraction = _normalize_whitespace(normalized_for_extraction)

    # Step 3: Extract landmark phrases (single pass, offsets stay valid)
    core_for_extraction, landmark = _extract_landmark(normalized_for_extraction)

    # Step 4: Extract components from the landmark-free core
    postal_code = _extract_postal_code(core_for_extraction, country)
    state = _extract_state(core_for_extraction, country)
    city = _extract_city(core_for_extraction, state, postal_code, country)

    # Step 5: Build the normalized address from the same core text
    core_address = _normalize_whitespace(core_for_extraction.replace(",", " "))

    # Step 6: Apply address abbreviation mapping
    core_address = _apply_mapping(core_address, ADDRESS_ABBREV_MAP, word_boundary=True)

    # Step 7: Final whitespace normalization
    core_address = _normalize_whitespace(core_address)

    return {
        "raw_address": original,
        "normalized_address": core_address,
        "landmark": landmark,
        "city": city,
        "state": state,
        "postal_code": postal_code,
    }


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single record combining name and address normalization.

    Args:
        row: Dictionary with keys 'entity_id', 'business_name',
             'business_address', 'country'

    Returns:
        Dictionary with original fields plus all normalized fields:
        - entity_id (passed through)
        - country (passed through)
        - raw_business_name, normalized_business_name, business_name_tokens
        - raw_business_address, normalized_business_address, landmark,
          city, state, postal_code
    """
    entity_id = row.get("entity_id", "")
    country = row.get("country", "")
    business_name = row.get("business_name", "")
    business_address = row.get("business_address", "")

    name_result = normalize_name(business_name)
    address_result = normalize_address(business_address, country or None)

    return {
        "entity_id": entity_id,
        "country": country,
        # Name fields
        "raw_business_name": name_result["raw_name"],
        "normalized_business_name": name_result["normalized_name"],
        "business_name_tokens": " ".join(name_result["name_tokens"]),
        # Address fields
        "raw_business_address": address_result["raw_address"],
        "normalized_business_address": address_result["normalized_address"],
        "landmark": address_result["landmark"],
        "city": address_result["city"],
        "state": address_result["state"],
        "postal_code": address_result["postal_code"],
    }


# ─── Script Entry Point ───────────────────────────────────────────────────────


def process_file(input_path: Path, output_path: Path, chunksize: int = 50_000) -> int:
    """
    Process a single TSV file through normalization.

    Reads and writes in chunks so multi-hundred-MB files never have to fit
    in memory together with their normalized copies.

    Args:
        input_path: Path to input TSV file
        output_path: Path to write normalized TSV file
        chunksize: Rows per read/write chunk

    Returns:
        Number of rows written
    """
    print(f"Processing: {input_path}", flush=True)
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    reader = pd.read_csv(input_path, sep="\t", dtype=str, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk.fillna("")
        rows = [normalize_record(row) for row in chunk.to_dict("records")]
        pd.DataFrame(rows).to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if total == 0 else "a",
            header=(total == 0),
        )
        total += len(rows)
        rate = total / max(time.time() - t0, 1e-9)
        print(f"  {total} rows ({rate:.0f} rows/s)", flush=True)

    print(f"  Written: {output_path} ({total} rows, {time.time() - t0:.0f}s)", flush=True)
    return total


def _read_head(path: Path, n: int = 3) -> pd.DataFrame:
    """Read the first n rows of a normalized output file for display."""
    return pd.read_csv(path, sep="\t", dtype=str, nrows=n).fillna("")


def _first_france_rows(path: Path, n: int = 2, chunksize: int = 100_000) -> pd.DataFrame:
    """Stream an output file until the first n France rows are found."""
    found: List[dict] = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
        for record in chunk[chunk["country"] == "France"].to_dict("records"):
            found.append(record)
            if len(found) >= n:
                return pd.DataFrame(found).fillna("")
    return pd.DataFrame(found).fillna("")


def _print_rows(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for _, row in df.iterrows():
        def shown(field: str) -> str:
            value = row[field]
            return value if isinstance(value, str) and value else "(none)"

        print(f"\nEntity: {row['entity_id']} | Country: {row['country']}")
        print(f"  Raw Name:      {row['raw_business_name']}")
        print(f"  Norm Name:     {row['normalized_business_name']}")
        print(f"  Name Tokens:   {row['business_name_tokens']}")
        print(f"  Raw Address:   {row['raw_business_address']}")
        print(f"  Norm Address:  {row['normalized_business_address']}")
        print(f"  Landmark:      {shown('landmark')}")
        print(f"  City:          {shown('city')}")
        print(f"  State:         {shown('state')}")
        print(f"  Postal:        {shown('postal_code')}")


def main():
    """Main entry point: normalize all six source files (ground truth excluded)."""
    base_input = config.DATA_RAW
    base_output = config.NORM_DIR

    files_to_process = [
        ("train/train_source1.tsv", "train_source1_normalized.tsv"),
        ("train/train_source2.tsv", "train_source2_normalized.tsv"),
        ("train/train_source3.tsv", "train_source3_normalized.tsv"),
        ("test/test_source1.tsv", "test_source1_normalized.tsv"),
        ("test/test_source2.tsv", "test_source2_normalized.tsv"),
        ("test/test_source3.tsv", "test_source3_normalized.tsv"),
    ]

    for rel_input, rel_output in files_to_process:
        process_file(base_input / rel_input, base_output / rel_output)

    # Print sample rows from train and test (loaded from disk, not retained)
    _print_rows(
        "SAMPLE OUTPUT - TRAIN (train_source1_normalized.tsv)",
        _read_head(base_output / "train_source1_normalized.tsv"),
    )
    _print_rows(
        "SAMPLE OUTPUT - TEST (test_source1_normalized.tsv) - includes France",
        _read_head(base_output / "test_source1_normalized.tsv", n=5),
    )

    # Also show France records specifically
    print("\n" + "=" * 80)
    print("FRANCE RECORDS FROM TEST FILES")
    print("=" * 80)
    for fname in ["test_source1_normalized.tsv", "test_source2_normalized.tsv", "test_source3_normalized.tsv"]:
        france_rows = _first_france_rows(base_output / fname)
        if len(france_rows) > 0:
            _print_rows(f"--- {fname} ---", france_rows)


if __name__ == "__main__":
    main()