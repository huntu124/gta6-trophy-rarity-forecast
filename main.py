"""GTA VI trophy rarity forecaster.

Reads one case (makefun/gta6-trophies-case/v1) and writes five quantiles of
each trophy's earn rate at the case's target, minimising expected pinball
loss on the logit scale.

Model, per trophy:
  - logit-rate least-squares line on ln(1 + hours since 2026-11-19T00:00Z)
    over the case's whole history (shrink the slope toward 0 — the log-time
    curve flattens as the launch cohort saturates);
  - predicted logit at the target's ln(1 + hours);
  - spread = fit residual combined with a random-walk term that grows with
    sqrt(days of horizon), floored so a quiet history stays honest;
  - fewer than 3 usable reads: persistence with the same walk spread.
"""
import json
import math
import os
from datetime import datetime, timezone

Z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
LOGIT_FLOOR, LOGIT_CAP = 0.01, 99.99
SHRINK_N = 4.0          # slope shrinkage: slope * n / (n + SHRINK_N)
WALK_PER_DAY = 0.35     # logit spread added per sqrt(day of horizon)
RESID_FLOOR = 0.15      # minimum residual spread
GROWTH_RATIO = 0.35     # assumed rate ratio at target: earners grow far slower than the tracked pool
FLAT_SPREAD = 1.6       # logit spread when the history carries no trend


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since(ts):
    return (ts - EPOCH).total_seconds() / 3600.0


def logit(p):
    p = min(max(p, LOGIT_FLOOR), LOGIT_CAP)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    x = min(max(x, logit(LOGIT_FLOOR)), logit(LOGIT_CAP))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def fit_line(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1)
    return intercept, slope, math.sqrt(max(var, 0.0)), n


def forecast_trophy(reads, idx, target_hours, target_x):
    xs, ys = [], []
    for r in reads:
        rate = r["rates"][idx]
        if rate is None:
            continue
        xs.append(math.log1p(hours_since(parse_ts(r["at"]))))
        ys.append(logit(rate))
    mu_trend, sigma_fit = None, None
    if len(xs) >= 3 and max(xs) > min(xs):
        intercept, slope, resid, n = fit_line(xs, ys)
        if abs(slope) * (max(xs) - min(xs)) > 0.2 or resid > 0.2:
            slope_eff = slope * n / (n + SHRINK_N)
            mu_trend = intercept + slope_eff * target_x
            sigma_fit = max(resid, RESID_FLOOR)
    if mu_trend is not None:
        mu = mu_trend
        sigma = math.sqrt(sigma_fit ** 2
                          + (WALK_PER_DAY * math.sqrt(target_hours / 24.0)) ** 2)
    else:
        # History carries no trend: the launch cohort is tiny and hardcore,
        # so rates dilute as the tracked pool floods in over the horizon.
        # Scale the last rate down and hedge with a wide logit spread.
        base = inv_logit(ys[-1]) if ys else 1.0
        mu = logit(base * GROWTH_RATIO)
        sigma = math.sqrt(FLAT_SPREAD ** 2
                          + (WALK_PER_DAY * math.sqrt(target_hours / 24.0)) ** 2)
    return [round(min(max(inv_logit(mu + Z[t] * sigma), LOGIT_FLOOR), LOGIT_CAP), 4)
            for t in (0.05, 0.25, 0.5, 0.75, 0.95)]


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    case_id = None
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".in"):
            case_id = name[:-3]
            break
    if case_id is None:
        raise SystemExit("no case file in input dir")
    with open(os.path.join(input_dir, case_id + ".in")) as f:
        case = json.load(f)

    target = parse_ts(case["target"])
    target_hours = hours_since(target)
    target_x = math.log1p(target_hours)
    reads = [r for r in case["history"] if r.get("ok", True)]
    trophies = case["trophies"]
    rows = [forecast_trophy(reads, i, target_hours, target_x)
            for i in range(len(trophies))]
    out = {"trophies": [{"id": t["id"], "q": rows[i]}
                        for i, t in enumerate(trophies)]}
    with open(os.path.join(output_dir, case_id + ".out"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
