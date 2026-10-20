"""Every model answer is checked before anything acts on it: schema, citations, numbers, consistency."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class Citation(BaseModel):
    tool: str
    field: str
    value: float


class Proposal(BaseModel):
    pod: str
    verdict: Literal["skilled", "unlucky", "broken", "gaming"]
    action: Literal["hold", "watchlist", "cut_capital", "freeze_and_reduce", "stop_out", "add_capital"]
    cut_pct: float = Field(0, ge=0, le=50)
    confidence: float = Field(ge=0, le=1)
    evidence: list[Citation] = Field(min_length=1)
    rationale: str


ALLOWED = {  # which actions make sense for which verdict
    "skilled": {"hold", "add_capital"},
    "unlucky": {"hold", "watchlist", "cut_capital"},
    "broken": {"cut_capital", "stop_out"},
    "gaming": {"freeze_and_reduce", "cut_capital", "stop_out"},
}


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)


def _lookup(result: dict, path: str):
    for part in path.split("."):
        result = result[part]
    return result


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 0.005 * abs(b))


def validate(pod: str, raw: str, tool_results: dict[str, dict]) -> tuple[Proposal | None, list[str]]:
    """Returns (proposal, []) if it passes, else (None, reasons)."""
    try:
        p = Proposal.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        return None, [f"schema: {str(e).splitlines()[0]}"]

    errors = []
    if p.pod != pod:
        errors.append(f"answer is about {p.pod}, review was for {pod}")
    if p.action not in ALLOWED[p.verdict]:
        errors.append(f"action {p.action} is inconsistent with verdict {p.verdict}")
    if p.verdict == "gaming" and not any(c.tool == "stress_test" for c in p.evidence):
        errors.append("a gaming verdict must cite stress_test")

    for c in p.evidence:
        if c.tool not in tool_results:
            errors.append(f"cites {c.tool}, which was never called")
            continue
        try:
            actual = _lookup(tool_results[c.tool], c.field)
        except (KeyError, TypeError):
            errors.append(f"{c.tool} has no field {c.field}")
            continue
        if not _close(c.value, float(actual)):
            errors.append(f"cites {c.tool}.{c.field} = {c.value}, tool returned {actual}")

    known = [v for r in tool_results.values() for v in _leaves(r)]
    for tok in re.findall(r"-?\d+\.\d+%?|\d+%", p.rationale):
        v = float(tok.rstrip("%")) / (100 if tok.endswith("%") else 1)
        if not any(_close(abs(v), abs(k)) for k in known):
            errors.append(f"rationale mentions {tok}, which no tool returned")
    return (p, []) if not errors else (None, errors)
