"""
Qafiya rule registry.

Each rule lives in its own file `rNNN_<slug>.py` (e.g. `r001_alif_tanween.py`)
and defines a module-level `RULE = Rule(...)`.

A rule's `match(poem)` receives a dict with keys:
    id, title, poet_name, verses (list[str] — search-normalized, no diacritics),
    verses_diacritized (list[str|None] — with tashkeel when available),
    meter, rhyme_letter
and returns either:
    * a dict of qafiya fields the rule is confident about — any subset of
      {rawiy, radf, wasl, harakah, type, taassis, pattern}, OR
    * None if the rule does not apply.

Rules are registered at import time by being discovered in this package.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Shared Arabic normalization helpers — use these in every rule
# ---------------------------------------------------------------------------

# All Unicode forms of hamza that are the same rawiy
_HAMZA_FORMS = {"ء", "ئ", "ؤ", "أ", "إ", "آ"}

def norm_char(c: str) -> str:
    """Normalize a single Arabic character for rawiy/radf comparison.

    - Collapses all hamza carrier forms (ئ, ؤ, أ, إ, آ) → bare hamza (ء).
    - Collapses ta marbuta (ة) → ta (ت): they are the same rawiy in classical
      prosody — a verse ending in ة and one ending in ت rhyme together.
    """
    if c in _HAMZA_FORMS:
        return "ء"
    if c == "ة":
        return "ت"
    return c


def norm_rawiy(chars: list[str]) -> list[str]:
    """Apply norm_char to a list of rawiy candidate characters."""
    return [norm_char(c) for c in chars]


@dataclass(frozen=True)
class Rule:
    code: str
    title_ar: str
    description_ar: str
    match: Callable[[dict], dict | None]


def all_rules() -> list[Rule]:
    """Discover every rNNN_*.py module in this package and collect its RULE."""
    rules: list[Rule] = []
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("r"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        rule = getattr(mod, "RULE", None)
        if isinstance(rule, Rule):
            rules.append(rule)
    rules.sort(key=lambda r: r.code)
    return rules


def get_rule(code: str) -> Rule | None:
    for r in all_rules():
        if r.code == code:
            return r
    return None
