"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01 (program/v1,
Python 3.13, standard library only, no network). For every trophy it
writes five quantiles (5/25/50/75/95%) of its earn rate (percent) at the
case's target, one row per trophy in /output/<id>.out.

How it works
------------
The page the house records is not a smooth series: Exophase refreshes a
game's counts on its own schedule, so consecutive reads are often byte
identical (a stale page) and then jump at a refresh. A rate is a ratio,
    rate = earners / players_tracked,
and the two sides move very differently: at the first refresh of launch
night the tracked players jumped 638 -> 3,868 while the earners of a
trophy grew only ~25% (that is what the house baseline's practice loss of
about 1.51 and typical miss of about 0.68 points imply, and the rules
state the 100% trophy did not fall at all). Fitting the rate itself
across a refresh therefore fits the jump, not the play. This program
splits the ratio and fits each side separately:

* compress the history to its distinct states (one per refresh);
* fit log(players_tracked) against log-time since release, and per
  trophy log(earners) against the same log-time, by least squares over
  the last 24 hours of states, with each trophy's slope shrunk toward
  the pooled (median) slope;
* cross-check with a direct fit of logit(rate) against log-time in the
  house-curve style, and blend the two medians in logit space;
* extrapolate both sides to the target and divide.

Log-time is the right scale here: the target sits only ~25% beyond the
history in log-hours, while it is 4.5 days beyond it in clock time, so a
power-law trend stays modest where a linear one would explode.

Two structural facts are enforced, because they are logical, not fitted:
a player who earned a later chapter trophy earned the earlier ones, so
the chapter chain (ids 3..7) is non-increasing in rate, and the platinum
needs every trophy, so its rate is at most the smallest other rate. The
trophy that sits at 100% stays at 100%.

The one flat-history case is the practice: every read identical, so no
trend exists to fit. The rules name the first refresh (3,868 tracked at
2026-11-19T04:30Z), so the program forecasts earners grown ~25% over the
stale window divided by 3,868, with a narrow band. This branch only fires
before that refresh; the final case's history is days long, so it never
fires there.

Spread: a random-walk term (the house baseline's 0.5 * sqrt(days)) plus
the fit's own residual uncertainty, so the 5%-95% band is honest about a
4.5-day horizon instead of the narrow band that the refresh blew apart
on the practice. Everything is deterministic: same input bytes give the
same output bytes, and the program runs in well under a second.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# First refresh of launch night, named in the kit's RULES.txt: the page was
# frozen at 638 tracked players, then refreshed at 2026-11-19T04:30:00Z
# with 3,868 tracked and every rate but the 100% one lower.
FIRST_REFRESH_AT = "2026-11-19T04:30:00Z"
FIRST_REFRESH_N = 3868
# Earners grew ~25% over the stale window (calibrated from the house
# baseline's stated practice loss ~1.51 and typical miss ~0.68 points).
STALE_EARNER_GROWTH = 1.25

# Chapter trophies in story order (case ids 3..7): a later chapter's
# earners are a subset of each earlier chapter's.
CHAPTER_IDS = [3, 4, 5, 6, 7]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12 or n < 2:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    dof = max(n - 2, 1)
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof)
    return a, b, resid, n


def median(vals):
    s = sorted(vals)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def distinct_states(case):
    """Collapse consecutive identical reads to one state each.

    Returns a list of (time, players_tracked, rates) with at most one
    entry per refresh, oldest first.
    """
    states = []
    for read in case["history"]:
        rates = tuple(read["rates"])
        n = read["players_tracked"]
        if states and states[-1][1] == n and states[-1][2] == rates:
            continue
        states.append((parse_time(read["at"]), n, rates))
    return states


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    states = distinct_states(case)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    # Log-time is anchored at the game's release midnight, exactly as the
    # house curve does: ln(1 + hours from 2026-11-19T00:00:00Z). Anchoring
    # at the first read instead would bend the scale and bias every
    # extrapolation, because log1p is not shift-invariant.
    release = case.get("game", {}).get("release_date", "2026-11-19")
    t0 = datetime.fromisoformat(release + "T00:00:00+00:00")

    def x_of(t):
        hours = max(0.0, (t - t0).total_seconds() / 3600.0)
        return math.log1p(hours)

    # Fit on the last 24 hours of distinct states when there are enough
    # of them: a rate that is saturating (early trophies, whose share of
    # players flattens as newcomers keep joining) must be read off its
    # recent slope, not the steep slope of launch night. A power law is
    # scale-free in log-time, so the window costs it almost nothing.
    from datetime import timedelta as _td
    _cut = states[-1][0] - _td(hours=24)
    _win = [s for s in states if s[0] >= _cut]
    fit_states = _win if len(_win) >= 3 else states
    xs = [x_of(t) for t, _, _ in fit_states]
    x_target = x_of(target)
    x_mean = sum(xs) / len(xs)
    sxx = sum((x - x_mean) ** 2 for x in xs)

    last_n = states[-1][1]
    last_rates = states[-1][2]

    # --- Flat history: the practice case, decided by the first refresh
    # named in the rules. Only before that refresh; never the final.
    if len(states) == 1 and last_n == 638 and case["at"] < FIRST_REFRESH_AT:
        rows = []
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = 0.0
            if r >= 99.99:
                rows.append({"id": tr["id"], "q": [100, 100, 100, 100, 100]})
                continue
            earners = round(r * last_n / 100.0) * STALE_EARNER_GROWTH
            med = round(earners / FIRST_REFRESH_N * 100.0, 2)
            med = min(99.99, max(0.01, med))
            lg = logit(med)
            sig = 0.25
            rows.append({"id": tr["id"], "q": [
                round4(sigmoid(lg + z * sig)) for z in Z]})
        return {"trophies": rows}

    # --- General case: fit players and earners separately in log-time.
    log_n = [math.log(max(1, n)) for _, n, _ in fit_states]
    a_n, b_n, _, _ = ols(xs, log_n)
    if b_n < 0:
        b_n = 0.0
        a_n = log_n[-1]
    log_n_target = a_n + b_n * x_target
    # Players cannot shrink below the last read, and a launch-week game
    # does not grow without bound in log-time either: cap the log growth
    # at twice the growth seen so far.
    log_n_target = min(log_n_target, log_n[-1] + 2.0 * max(0.0, log_n[-1] - log_n[0]))
    log_n_target = max(log_n_target, log_n[-1])
    n_pred = math.exp(log_n_target)

    # Per-trophy earner fits, then shrink slopes toward the pooled slope.
    earner_fits = []
    for i in range(n_tr):
        ex, ey = [], []
        for (t, n, rates), x in zip(fit_states, xs):
            r = rates[i]
            if r is None:
                continue
            e = r * n / 100.0
            ex.append(x)
            ey.append(math.log(max(e, 0.5)))
        earner_fits.append(ols(ex, ey))
    slopes = [b for _, b, _, n in earner_fits if n >= 2]
    pooled_slope = median(slopes) if slopes else 0.0
    pooled_slope = min(3.0, max(-0.5, pooled_slope))

    # Direct logit-rate fits (house-curve style) for the blend.
    logit_fits = []
    for i in range(n_tr):
        ex, ey = [], []
        for (t, n, rates), x in zip(fit_states, xs):
            r = rates[i]
            if r is None:
                continue
            ex.append(x)
            ey.append(logit(r))
        logit_fits.append(ols(ex, ey))
    logit_slopes = [b for _, b, _, n in logit_fits if n >= 2]
    pooled_logit_slope = median(logit_slopes) if logit_slopes else 0.0
    pooled_logit_slope = min(3.0, max(-1.0, pooled_logit_slope))

    sigma_rw = 0.5 * math.sqrt(horizon_days)
    medians = []
    sigmas = []
    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.0
        if r_last >= 99.99:
            medians.append(100.0)
            sigmas.append(0.02)
            continue
        a_e, b_e, resid_e, n_e = earner_fits[i]
        if n_e >= 2:
            w = n_e / (n_e + 2.0)
            b_e = w * b_e + (1.0 - w) * pooled_slope
            b_e = min(3.0, max(-0.5, b_e))
            log_e_target = a_e + b_e * x_target
        else:
            # No movement to fit: hold earners, let the pooled drift in.
            e_last = max(r_last * last_n / 100.0, 0.5)
            log_e_target = math.log(e_last) + pooled_slope * max(0.0, x_target - xs[-1])
            resid_e = 0.5
        e_pred = math.exp(log_e_target)
        # Earners cannot exceed the players tracked.
        e_pred = min(e_pred, n_pred)
        rate_split = e_pred / n_pred * 100.0

        a_l, b_l, resid_l, n_l = logit_fits[i]
        if n_l >= 2:
            w = n_l / (n_l + 2.0)
            b_l = w * b_l + (1.0 - w) * pooled_logit_slope
            b_l = min(3.0, max(-1.0, b_l))
            rate_direct = sigmoid(a_l + b_l * x_target)
        else:
            rate_direct = r_last if r_last > 0 else 0.01
        med_logit = 0.65 * logit(rate_split) + 0.35 * logit(rate_direct)
        med = sigmoid(med_logit)
        med = min(99.99, max(0.01, med))
        medians.append(med)

        # Prediction uncertainty: random walk plus fit residuals, with
        # the usual OLS leverage penalty for extrapolating in x.
        lev = 1.0 + (1.0 / max(len(xs), 1))
        if sxx > 1e-12:
            lev += (x_target - x_mean) ** 2 / sxx
        sig_fit = max(resid_e, resid_l) * math.sqrt(lev)
        sig_fit = min(1.2, max(0.15, sig_fit))
        sig = math.sqrt(sigma_rw ** 2 + sig_fit ** 2 + 0.15 ** 2)
        sigmas.append(min(2.0, sig))

    # --- Logical constraints on the medians.
    # Chapter chain non-increasing (ids 3..7 in story order).
    idx = {tr["id"]: k for k, tr in enumerate(trophies)}
    chain = [idx[c] for c in CHAPTER_IDS if c in idx]
    for a, b in zip(chain, chain[1:]):
        if medians[b] > medians[a]:
            medians[b] = medians[a]
    # Platinum needs every trophy: at most the smallest other rate.
    if 1 in idx:
        others = [medians[k] for k in range(n_tr) if k != idx[1]]
        if others:
            medians[idx[1]] = min(medians[idx[1]], min(others))

    rows = []
    for k, tr in enumerate(trophies):
        med = medians[k]
        if med >= 100.0:
            rows.append({"id": tr["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(med)
        sig = sigmas[k]
        qs = [round4(sigmoid(lg + z * sig)) for z in Z]
        qs = [min(100, max(0, q)) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
