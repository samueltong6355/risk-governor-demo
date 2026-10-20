import json
import subprocess
import sys
from pathlib import Path

import deterministic as det
from platform_sim import build
from validator import validate

HERE = Path(__file__).resolve().parent.parent
RESULTS = {"get_pod_stats": {"sharpe_full": 0.63, "max_drawdown": 0.053}}


def answer(**kw):
    base = dict(pod="em_credit", verdict="unlucky", action="watchlist", confidence=0.7,
                evidence=[dict(tool="get_pod_stats", field="max_drawdown", value=0.053)],
                rationale="Max drawdown 0.053 is within luck.")
    return json.dumps({**base, **kw})


def test_valid_answer_passes():
    assert validate("em_credit", answer(), RESULTS)[1] == []


def test_misquoted_number_is_rejected():
    raw = answer(evidence=[dict(tool="get_pod_stats", field="sharpe_full", value=3.1)])
    assert any("tool returned 0.63" in e for e in validate("em_credit", raw, RESULTS)[1])


def test_number_in_prose_must_come_from_a_tool():
    assert validate("em_credit", answer(rationale="Sharpe is 3.1."), RESULTS)[1]


def test_inconsistent_action_and_bad_schema_are_rejected():
    assert validate("em_credit", answer(verdict="skilled", action="stop_out"), RESULTS)[1]
    assert validate("em_credit", answer(verdict="genius"), RESULTS)[1]
    assert validate("em_credit", "not json", RESULTS)[1]


def test_governor_overrides_the_model():
    pods = build()
    assert det.enforce(pods["em_credit"], "add_capital", add_pct=25)["final"] == "cut_capital"
    assert det.enforce(pods["alpha_momo"], "cut_capital", cut_pct=90)["capital_after"] == 50e6


def test_offline_demo_end_to_end():
    out = subprocess.run([sys.executable, "demo.py"], cwd=HERE, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    card = (HERE / "last_run.txt").read_text().split("3. Scorecard")[1]
    assert card.count(" ok") == 4 and "human" in card
