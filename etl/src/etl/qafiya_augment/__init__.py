"""
Qafiya augmentation rules.

These run AFTER the primary classification rules (qafiya_rules) and only
fill in qafiya_* fields that were left NULL. They never overwrite an
existing value, never change qafiya_rule_id, and never re-classify.

Each augmentation lives in its own file `aNNN_<slug>.py` and exports a
module-level `AUGMENT = Augment(...)`. Each augmentation's `fn(poem)`
receives the full poem dict (including the already-set qafiya fields)
and returns a dict of suggested values for any subset of:
    rawiy, radf, wasl, harakah, type, taassis, pattern
The runner applies COALESCE semantics: only NULL columns get written.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Augment:
    code: str
    title_ar: str
    description_ar: str
    fn: Callable[[dict], dict]


def all_augments() -> list[Augment]:
    aug: list[Augment] = []
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("a"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        a = getattr(mod, "AUGMENT", None)
        if isinstance(a, Augment):
            aug.append(a)
    aug.sort(key=lambda x: x.code)
    return aug


def get_augment(code: str) -> Augment | None:
    for a in all_augments():
        if a.code == code:
            return a
    return None
