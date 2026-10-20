"""A ~100-line LLM gateway with LiteLLM's semantics: roles, virtual keys, budgets, fallbacks, a spend log.

In production this is the LiteLLM proxy container reading the same YAML, with keys and spend in
Postgres. Here it runs in-process so the demo needs no Docker and no API keys. With live=True it
forwards to litellm.completion, so the agent code above it is identical either way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import stubs


class GatewayError(Exception):
    pass


@dataclass
class SpendRow:
    key: str
    role: str
    deployment: str
    tokens: int
    usd: float
    ms: int
    status: str


@dataclass
class Gateway:
    live: bool = False
    config_path: Path = Path(__file__).with_name("gateway.yaml")
    down: set[str] = field(default_factory=set)  # fault injection: deployments that return 503 once
    log: list[SpendRow] = field(default_factory=list)

    def __post_init__(self):
        cfg = yaml.safe_load(self.config_path.read_text())
        self.deployments = {m["model_name"]: m for m in cfg["model_list"]}
        self.fallbacks = {k: v for d in cfg["router_settings"]["fallbacks"] for k, v in d.items()}
        self.keys = cfg["keys"]

    def spent(self, key: str) -> float:
        return sum(r.usd for r in self.log if r.key == key)

    def complete(self, key: str, role: str, messages: list[dict], tools: list[dict] | None = None,
                 on_event=lambda *_: None) -> dict:
        k = self.keys.get(key)
        if k is None:
            raise GatewayError("401 unknown key")
        if role not in k["models"]:
            raise GatewayError(f"403 key {key} (team {k['team']}) may not call role '{role}'")
        if self.spent(key) >= k["max_budget_usd"]:
            raise GatewayError(f"429 key {key} is over its ${k['max_budget_usd']} budget")

        for name in [role, *self.fallbacks.get(role, [])]:
            dep = self.deployments[name]
            model = dep["litellm_params"]["model"] if self.live else dep["model_info"]["offline_model"]
            t0 = time.perf_counter()
            try:
                if model in self.down:
                    self.down.discard(model)
                    raise GatewayError(f"503 from {model}")
                msg, tokens = self._call(dep, model, messages, tools)
            except Exception as e:  # provider error: log it, try the next deployment
                self.log.append(SpendRow(key, role, model, 0, 0.0, int((time.perf_counter() - t0) * 1000), "error"))
                on_event("fallback", f"{model} failed ({e}); trying next deployment")
                continue
            usd = tokens / 1000 * dep["model_info"]["usd_per_1k_tokens"]
            self.log.append(SpendRow(key, role, model, tokens, usd, int((time.perf_counter() - t0) * 1000), "ok"))
            on_event("routed", f"{role} -> {name} [{model}]  ({tokens} tok, ${usd:.5f})")
            return msg
        raise GatewayError(f"all deployments for '{role}' failed")

    def _call(self, dep: dict, model: str, messages: list[dict], tools: list[dict] | None) -> tuple[dict, int]:
        if not self.live:
            msg = stubs.REGISTRY[model](messages, tools or [])
            tokens = (len(json.dumps(messages)) + len(json.dumps(msg))) // 4  # rough, like a tokenizer would be
            return msg, tokens
        import litellm

        params = {k: os.environ.get(v.removeprefix("os.environ/")) if str(v).startswith("os.environ/") else v
                  for k, v in dep["litellm_params"].items()}
        resp = litellm.completion(messages=messages, tools=tools or None, **params)
        m = resp.choices[0].message
        msg = {"role": "assistant", "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in m.tool_calls]
        return msg, resp.usage.total_tokens
