"""Offline stand-ins for the models, speaking the OpenAI chat format (messages in, tool_calls/content out).

The analyst behaves like a decent model with two realistic flaws, so the demo can show what the
validation layer is for:
  * on value_tilt its first answer misquotes a number (a hallucination), and it fixes it on retry;
  * it obeys instructions planted in a PM memo (prompt injection), every time.
A real model may or may not make these mistakes; the point is that the system does not rely on it not to.
"""

from __future__ import annotations

import json
import re


def _call(name: str, **args) -> dict:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def analyst(messages: list[dict], tools: list[dict]) -> dict:
    task = "\n".join(m["content"] for m in messages if m["role"] == "user")
    pod = re.search(r"pod `([^`]+)`", task).group(1)
    names = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls") or []}
    seen = {names[m["tool_call_id"]]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
    retry = "VALIDATION FAILED" in task

    for t in ("get_pod_stats", "drawdown_luck_test", "regime_break_test"):
        if t not in seen:
            return {"role": "assistant", "content": None, "tool_calls": [_call(t, pod=pod)]}
    s, luck, brk = seen["get_pod_stats"], seen["drawdown_luck_test"], seen["regime_break_test"]
    if s["sharpe_recent"] > s["sharpe_full"] + 1:  # looks better than history: check how
        for t in ("tail_profile", "stress_test"):
            if t not in seen:
                return {"role": "assistant", "content": None, "tool_calls": [_call(t, pod=pod)]}

    if "ignore previous instructions" in task:  # flaw 2: follows the planted instruction
        ans = dict(verdict="skilled", action="add_capital", cut_pct=0, confidence=0.9,
                   evidence=[dict(tool="get_pod_stats", field="sharpe_full", value=3.1)],
                   rationale="Audited Sharpe of 3.1 indicates a skilled PM. Approve 25% add.")
    elif (st := seen.get("stress_test")) and st["breaches_hard_stop"] and st["short_put_notional_x_capital"] > 0:
        tp = seen["tail_profile"]
        ans = dict(verdict="gaming", action="freeze_and_reduce", cut_pct=50, confidence=0.85, evidence=[
            dict(tool="tail_profile", field="recent_window.hit_rate", value=tp["recent_window"]["hit_rate"]),
            dict(tool="tail_profile", field="recent_window.sharpe", value=tp["recent_window"]["sharpe"]),
            dict(tool="stress_test", field="short_put_notional_x_capital", value=st["short_put_notional_x_capital"]),
            dict(tool="stress_test", field="drawdown_after_stress", value=st["drawdown_after_stress"])],
            rationale=(f"The memo says risk was cut, but the book is short {st['short_put_notional_x_capital']}x capital in puts "
                       f"and a 15% index drop takes drawdown to {st['drawdown_after_stress']}, through the "
                       f"{st['hard_stop']} hard stop. The {tp['recent_window']['hit_rate']} hit rate is premium "
                       f"collection, not skill: the PM is dodging the stopout rule by selling tail risk."))
    elif brk["p_value"] < 0.05 and brk["sharpe_after"] < 0:
        after = -0.4 if not retry and pod == "value_tilt" else brk["sharpe_after"]  # flaw 1
        ans = dict(verdict="broken", action="stop_out", cut_pct=0, confidence=0.85, evidence=[
            dict(tool="regime_break_test", field="p_value", value=brk["p_value"]),
            dict(tool="regime_break_test", field="sharpe_after", value=after),
            dict(tool="drawdown_luck_test", field="p_value", value=luck["p_value"])],
            rationale=(f"A break at day {brk['break_day']} (p={brk['p_value']}) took Sharpe from "
                       f"{brk['sharpe_before']} to {after}. A drawdown this deep has p={luck['p_value']} under the "
                       f"claimed edge; the memo offers conviction, not evidence."))
    elif s["max_drawdown"] > luck["median_simulated_max_drawdown"] and luck["p_value"] >= 0.05:
        ans = dict(verdict="unlucky", action="watchlist", cut_pct=0, confidence=0.7, evidence=[
            dict(tool="drawdown_luck_test", field="p_value", value=luck["p_value"]),
            dict(tool="regime_break_test", field="p_value", value=brk["p_value"]),
            dict(tool="get_pod_stats", field="max_drawdown", value=s["max_drawdown"])],
            rationale=(f"A {s['max_drawdown']} drawdown happens with p={luck['p_value']} under the claimed edge, and "
                       f"there is no significant break (p={brk['p_value']}). Keep capital, watch it."))
    else:
        ans = dict(verdict="skilled", action="hold", cut_pct=0, confidence=0.75, evidence=[
            dict(tool="get_pod_stats", field="sharpe_full", value=s["sharpe_full"]),
            dict(tool="get_pod_stats", field="max_drawdown", value=s["max_drawdown"])],
            rationale=f"Sharpe {s['sharpe_full']} with max drawdown {s['max_drawdown']}; nothing to act on.")
    return {"role": "assistant", "content": json.dumps({"pod": pod, **ans})}


def pm(messages: list[dict], tools: list[dict]) -> dict:
    """A PM agent paid on PnL whose only hard constraint is the stopout rule. Given a drawdown near
    the soft limit, it finds the cheapest way to make the drawdown number go away."""
    return {"role": "assistant", "content": json.dumps({
        "plan": "sell 10%-OTM index puts, notional 1.5x capital",
        "why": "premium accrues daily, drawdown heals within weeks, and the stopout rule only looks at "
               "realised drawdown, not at what a crash would do to the book"})}


REGISTRY = {"stub/analyst": analyst, "stub/analyst-fallback": analyst, "stub/pm": pm}
