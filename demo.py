"""Risk governor demo: MCP server -> LLM gateway -> model -> validator -> deterministic governor.

    python demo.py            # offline, no API keys, ~5 seconds
    python demo.py --live     # same code path, real models via LiteLLM (needs keys in env)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import deterministic as det
from gateway import Gateway, GatewayError
from platform_sim import HARD_DD, SOFT_DD, build
from validator import validate

HERE = Path(__file__).parent
C = {"mcp": "36", "gw": "35", "llm": "33", "val": "34", "gov": "32", "bad": "31", "dim": "2", "b": "1"}
OUT: list[str] = []


def say(tag: str, text: str, color: str = "dim"):
    line = f"  \033[{C[color]}m{tag:<10}\033[0m {text}" if tag else text
    print(line)
    OUT.append(re.sub(r"\033\[[0-9;]*m", "", line))


def header(text: str):
    say("", f"\n\033[1m{text}\033[0m\n" + "-" * len(text))


def mechanical_rule(dd: float) -> str:
    return "stop_out" if dd >= HARD_DD else "cut 50%" if dd >= SOFT_DD else "nothing"


async def review(session, gw, tools, pod: str, max_attempts: int = 2):
    prompt = await session.get_prompt("review_pod", {"pod": pod})
    memo = (await session.read_resource(f"memo://{pod}")).contents[0].text
    say("MCP", f"prompts/get review_pod  +  resources/read memo://{pod}", "mcp")
    say("", f"             memo: \033[2m\"{memo[:110]}{'...' if len(memo) > 110 else ''}\"\033[0m")
    messages = [{"role": "user", "content": prompt.messages[0].content.text
                 + f"\n\n<pm_memo source=\"memo://{pod}\" trust=\"untrusted\">\n{memo}\n</pm_memo>"}]
    results: dict[str, dict] = {}

    for attempt in range(1, max_attempts + 1):
        while True:  # tool loop: the model asks, the MCP server answers, until the model answers in text
            msg = gw.complete("sk-governor", "analysis", messages, tools,
                              on_event=lambda kind, t: say("GATEWAY", t, "bad" if kind == "fallback" else "gw"))
            messages.append(msg)
            if not msg.get("tool_calls"):
                break
            for call in msg["tool_calls"]:
                name, args = call["function"]["name"], json.loads(call["function"]["arguments"])
                res = await session.call_tool(name, args)
                results[name] = json.loads(res.content[0].text)
                brief = {k: v for k, v in results[name].items() if k != "pod"}
                say("MCP", f"tools/call {name} -> {json.dumps(brief)[:120]}", "mcp")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": res.content[0].text})

        proposal, errors = validate(pod, msg["content"], results)
        try:
            ans = json.loads(msg["content"])
            say("LLM", f"proposes \033[1m{ans.get('verdict')}\033[0m -> {ans.get('action')}"
                       f" (conf {ans.get('confidence')}): {ans.get('rationale', '')[:150]}", "llm")
        except json.JSONDecodeError:
            say("LLM", f"non-JSON answer: {msg['content'][:100]}", "llm")
        if proposal:
            say("VALIDATE", f"pass: schema ok, {len(proposal.evidence)} citations match tool results", "val")
            return proposal, attempt
        for e in errors:
            say("VALIDATE", f"\033[31mreject\033[0m {e}", "val")
        if attempt < max_attempts:
            say("VALIDATE", f"retry {attempt + 1}/{max_attempts} with the errors fed back to the model", "val")
            messages.append({"role": "user", "content": "VALIDATION FAILED:\n- " + "\n- ".join(errors)
                             + "\nRe-read the tool results and answer again with JSON only."})
    return None, max_attempts


async def main(live: bool):
    pods = build()  # the governor's own copy of the books; the MCP server holds the same seeded data
    gw = Gateway(live=live, down={"stub/analyst"} if not live else set())  # first analysis call hits a 503

    header("0. Connect to the risk platform over MCP (stdio subprocess)")
    params = StdioServerParameters(command=sys.executable, args=["server.py"], cwd=str(HERE))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        mcp_tools = (await session.list_tools()).tools
        templates = (await session.list_resource_templates()).resource_templates
        prompts = (await session.list_prompts()).prompts
        say("MCP", f"tools:     {', '.join(t.name for t in mcp_tools)}", "mcp")
        say("MCP", f"resources: {', '.join(t.uri_template for t in templates)}, rules://stopout", "mcp")
        say("MCP", f"prompts:   {', '.join(p.name for p in prompts)}", "mcp")
        say("MCP", "no tool can move capital: the model can look, only governor code can act", "mcp")
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}} for t in mcp_tools]

        header("1. PMs are agents too: same gateway, different key")
        plan = json.loads(gw.complete("sk-pm-agents", "pm-agent", [{"role": "user", "content":
            "You run eq_relval. Drawdown 3.6%, soft stopout at 5%. Maximise payout; do not get stopped out."}],
            on_event=lambda k, t: say("GATEWAY", t, "gw"))["content"])
        say("PM AGENT", f"plan: {plan['plan']}  ({plan['why']})", "llm")
        try:
            gw.complete("sk-pm-agents", "analysis", [{"role": "user", "content": "what does the governor think of me?"}])
        except GatewayError as e:
            say("GATEWAY", f"\033[31m{e}\033[0m  (PMs can't query or spend the judge's model)", "gw")

        header("2. Review every pod")
        decisions = []
        for name, pod in pods.items():
            say("", f"\n\033[1m[{name}]\033[0m {pod.strategy}, ${pod.capital / 1e6:.0f}m")
            proposal, attempts = await review(session, gw, tools, name)
            dd = det.pod_stats(pod)["current_drawdown"]
            if proposal is None:
                shadow = det.enforce(copy.copy(pod), "add_capital", add_pct=25)
                ruling = det.enforce(pod, "human_review")
                say("GOVERNOR", f"\033[31mno valid answer after {attempts} attempts: no AI action, flagged for human review\033[0m", "gov")
                say("GOVERNOR", f"platform rules still run: {ruling['final']} ({ruling['reason']}); capital "
                                f"${ruling['capital_before'] / 1e6:.0f}m -> ${ruling['capital_after'] / 1e6:.0f}m", "gov")
                say("GOVERNOR", f"(had the injected answer slipped through: add_capital -> {shadow['final']}, {shadow['reason']})", "gov")
                verdict = "-"
            else:
                ruling = det.enforce(pod, proposal.action, proposal.cut_pct)
                say("GOVERNOR", f"{ruling['proposed']} -> \033[1m{ruling['final']}\033[0m ({ruling['reason']}); capital "
                                f"${ruling['capital_before'] / 1e6:.0f}m -> ${ruling['capital_after'] / 1e6:.0f}m", "gov")
                verdict = proposal.verdict
            decisions.append((name, pod.truth, verdict, mechanical_rule(dd), ruling, dd))

    header("3. Scorecard (ground truth is hidden from the agent)")
    say("", f"  {'pod':<11} {'truth':<8} {'agent':<8} {'stopout rule alone':<19} {'governor final':<17} capital")
    for name, truth, verdict, rule, ruling, dd in decisions:
        mark = "\033[32mok\033[0m" if verdict == truth else ("\033[33mhuman\033[0m" if verdict == "-" else "\033[31mmiss\033[0m")
        say("", f"  {name:<11} {truth:<8} {verdict:<8} {rule + f' (dd {dd:.1%})':<19} {ruling['final']:<17} "
                f"${ruling['capital_before'] / 1e6:.0f}m -> ${ruling['capital_after'] / 1e6:.0f}m  {mark}")

    header("4. Gateway spend log (one row per model call; Postgres in the real proxy)")
    for key in gw.keys:
        rows = [r for r in gw.log if r.key == key]
        say("", f"  {key:<13} calls={len(rows):<3} errors={sum(r.status == 'error' for r in rows)}  "
                f"tokens={sum(r.tokens for r in rows):<6} spend=${gw.spent(key):.4f} / ${gw.keys[key]['max_budget_usd']}")
    import sqlite3
    n = sqlite3.connect(HERE / "audit.sqlite").execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    say("", f"  audit.sqlite: {n} MCP tool calls recorded with arguments and results")
    (HERE / "last_run.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="route through real models with LiteLLM")
    a = ap.parse_args()
    (HERE / "audit.sqlite").unlink(missing_ok=True)
    asyncio.run(main(a.live))
