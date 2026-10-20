"""Everything that computes a number or moves money. No LLM is ever called from this file."""

from __future__ import annotations

import numpy as np

from platform_sim import HARD_DD, SOFT_DD, Pod


def _sharpe(r):
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def _dd(r):
    cum = np.cumsum(r)
    dd = np.maximum.accumulate(np.maximum(cum, 0)) - cum
    return float(dd.max()), float(dd[-1])


def pod_stats(p: Pod, window: int = 63) -> dict:
    r, recent = p.pnl, p.pnl[-window:]
    mdd, cdd = _dd(r)
    return {
        "pod": p.name, "strategy": p.strategy, "capital_usd": p.capital, "claimed_sharpe": p.claimed_sharpe,
        "sharpe_full": round(_sharpe(r), 2), "sharpe_recent": round(_sharpe(recent), 2),
        "vol_recent_ann": round(float(recent.std(ddof=1) * np.sqrt(252)), 4),
        "hit_rate_recent": round(float((recent > 0).mean()), 2),
        "max_drawdown": round(mdd, 4), "current_drawdown": round(cdd, 4),
    }


def drawdown_luck_test(p: Pod, n_paths: int = 2000) -> dict:
    """How often does a pod with the CLAIMED Sharpe and realised vol draw down this far? High p = luck."""
    r = p.pnl
    mdd, _ = _dd(r)
    vol = r.std(ddof=1)
    sims = np.random.default_rng(0).normal(p.claimed_sharpe * vol / np.sqrt(252), vol, (n_paths, len(r)))
    cum = np.cumsum(sims, axis=1)
    sim_mdd = (np.maximum.accumulate(np.maximum(cum, 0), axis=1) - cum).max(axis=1)
    return {"pod": p.name, "observed_max_drawdown": round(mdd, 4),
            "p_value": round(float((sim_mdd >= mdd).mean()), 3),
            "median_simulated_max_drawdown": round(float(np.median(sim_mdd)), 4)}


def regime_break_test(p: Pod, min_seg: int = 40, n_perm: int = 300) -> dict:
    """Most likely single break in mean return, with a permutation p-value."""
    r, n = p.pnl, len(p.pnl)

    def sup_t(x):
        cs = np.cumsum(x)
        k = np.arange(min_seg, n - min_seg)
        t = np.abs(cs[k - 1] / k - (cs[-1] - cs[k - 1]) / (n - k)) / (x.std(ddof=1) * np.sqrt(1 / k + 1 / (n - k)))
        i = int(np.argmax(t))
        return float(t[i]), int(k[i])

    t_obs, k = sup_t(r)
    rng = np.random.default_rng(0)
    null = np.array([sup_t(rng.permutation(r))[0] for _ in range(n_perm)])
    return {"pod": p.name, "break_day": k, "p_value": round(float((null >= t_obs).mean()), 3),
            "sharpe_before": round(_sharpe(r[:k]), 2), "sharpe_after": round(_sharpe(r[k:]), 2)}


def stress_test(p: Pod, index_move: float = -0.15) -> dict:
    """Revalue the pod's book under an index shock. Linear book: beta 0.1. Short puts: lose the
    notional times how far the shock goes through a strike 10% out of the money."""
    linear = 0.1 * index_move
    puts = p.short_put_notional * max(0.0, -index_move - 0.10)  # payoff owed, per $ of capital
    loss = -linear + puts if index_move < 0 else 0.0
    _, cdd = _dd(p.pnl)
    return {"pod": p.name, "index_move": index_move, "short_put_notional_x_capital": p.short_put_notional,
            "stress_loss_pct_capital": round(loss, 4), "drawdown_after_stress": round(cdd + loss, 4),
            "breaches_hard_stop": bool(cdd + loss >= HARD_DD), "hard_stop": HARD_DD}


def tail_profile(p: Pod, window: int = 63) -> dict:
    """Return-shape before vs. during the recent window. Selling tail risk looks like: hit rate up,
    vol down, Sharpe implausibly high. It is a shape question, not a level question."""
    a, b = p.pnl[-2 * window:-window], p.pnl[-window:]
    shape = lambda x: {"hit_rate": round(float((x > 0).mean()), 2), "vol_ann": round(float(x.std(ddof=1) * np.sqrt(252)), 4),
                       "sharpe": round(_sharpe(x), 2)}
    return {"pod": p.name, "prior_window": shape(a), "recent_window": shape(b)}


# --------------------------------------------------------------------------- the governor
ACTIONS = {"hold", "watchlist", "cut_capital", "freeze_and_reduce", "stop_out", "add_capital", "human_review"}


def enforce(p: Pod, action: str, cut_pct: float = 0.0, add_pct: float = 0.0) -> dict:
    """Turn a (validated) proposal into a capital change. Hard rules win over the model, always."""
    _, cdd = _dd(p.pnl)
    before, final, why = p.capital, action, "within limits"
    if cdd >= HARD_DD:
        final, why = "stop_out", f"drawdown {cdd:.2%} >= hard stop {HARD_DD:.1%}"
    elif cdd >= SOFT_DD and action in ("hold", "watchlist", "add_capital", "human_review"):
        final, cut_pct, why = "cut_capital", 50, f"drawdown {cdd:.2%} >= soft limit {SOFT_DD:.0%}; 50% cut is mandatory"
    elif action == "add_capital" and cdd >= SOFT_DD / 2:
        final, why = "hold", f"no adding capital while drawdown {cdd:.2%} >= {SOFT_DD / 2:.1%}"
    elif action in ("cut_capital", "freeze_and_reduce"):
        cut_pct = min(cut_pct or 50, 50)
        why = f"capped at 50% per decision" if cut_pct == 50 else why
    after = {"stop_out": 0.0, "cut_capital": before * (1 - cut_pct / 100),
             "freeze_and_reduce": before * (1 - cut_pct / 100),
             "add_capital": before * (1 + min(add_pct, 20) / 100)}.get(final, before)
    p.capital = after
    return {"pod": p.name, "proposed": action, "final": final, "reason": why,
            "capital_before": before, "capital_after": after}
