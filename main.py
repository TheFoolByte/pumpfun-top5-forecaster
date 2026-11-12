#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: Monte Carlo log-cap forecaster.

Reads the house's 10-minute pump.fun snapshots (history.ndjson + task.json),
simulates every candidate coin's log10 market cap forward with a jump-diffusion
(empirical step resampling, shrunk recent-slope drift that decays with age),
a graduation hazard for bonding-curve coins, and a newcomer cohort whose sizes
are drawn from the empirical distribution of newcomer sizes relative to the
current place-1 level. Averages place probabilities and place-cap
distributions over the seven answer reads from task.json.

Stdlib only. Deterministic: fixed seed, byte-identical output across runs.
"""
import heapq
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
# Monte Carlo budget: total coin-steps kept near this; trials adapt to horizon/size.
STEP_BUDGET = 18_000_000
MAX_TRIALS = 2000
MIN_TRIALS = 700
CAND_MAX = 220
# Drift: recent slope shrunk toward the global mean step; decays with time.
# Shrink 0.50 (was 0.30): rolling-cut proxy mean -32.281 vs -33.191 at 0.30,
# improving 4 of 5 cuts (A +0.55, B +0.70, C +1.60, L2 +1.69; L flat, it
# trains on 1 mark so there is no slope to shrink). Less shrinkage trusts
# recent momentum more, which the proxy says pays on memecoin caps.
DRIFT_SHRINK = 0.50
DRIFT_HALF_LIFE_MIN = 120.0
POOL_DRIFT_BONUS = 0.01  # per 10-min step, pool coins only
DRIFT_CLIP = (-0.05, 0.08)
# Newcomers: per read, this many contender draws. A contender is a newborn
# that reaches top-5 scale; the practice shows ~2.5 of them per 10 min, and
# drawing 8 from the contender-delta distribution lands ~half of the top-5
# slots on newcomers, matching the observed 50% newcomer share.
NEWCOMERS_PER_READ = int(os.environ.get("MF_NEWCOMERS", "6"))# Graduation: hazard per 10-min step for a curve coin, scaled by (qr/85)^2.
GRAD_BASE = 0.01
GRAD_QR_REF = 85.0
# Step-pool fallback: if the history yields fewer steps than this, pad the
# pool with N(0, STEP_FALLBACK_SD) draws (a 1-mark history has no steps at all).
STEP_POOL_MIN = 200
STEP_FALLBACK_SD = 0.12
# Size-only blend tau (v3 experiment): sizes are more persistent than
# identities — the top-5 sizes stay in a plausible band even as the coins
# turn over. A longer tau for the SIZE blend anchors the size distribution
# near the last mark's levels at long horizons. Default = BLEND_TAU_MIN.
SIZE_TAU_MIN = float(os.environ.get("MF_SIZE_TAU", "480.0"))  # v4: long-horizon size anchor (round 3, verified +23.5 nats on realistic 7-13h test)
# Long-horizon level control (v3): the pure Monte Carlo's top level grows
# without bound — random-walk max ~ sqrt(steps) plus a newcomer ratchet
# (newcomers anchor to the running max). On the 66-mark final-shaped input
# v2's place-1 mean cap is 9.42 dex (~$400B, absurd) with newcomer share
# 1.000 and ALL size mass in bin 81. Two env knobs, both default off:
# MF_REVERT_KAPPA: per-step pull of each coin's log-cap back toward its
#   last-mark size (OU mean reversion; bounds the level, keeps incumbents
#   competitive and sizes plausible at 7-13 h).
# MF_STEP_SHRINK: multiplicative shrink of the resampled step noise.
REVERT_KAPPA = float(os.environ.get("MF_REVERT_KAPPA", "0.0"))
STEP_SHRINK = float(os.environ.get("MF_STEP_SHRINK", "1.0"))
# v4 (realistic long-horizon test) knobs, all default to v2 behavior:
# MF_NEWCOMER_ANCHOR=1: newcomers anchor to the LAST-MARK place-1 level
#   (last_place1) instead of the running max — kills the newcomer ratchet
#   that carries the top level up with it at 7-13 h.
# MF_SIZE_CAP (dex): hard cap on a coin's simulated size above its last-mark
#   size (cand_size); bounds the random-walk diffusion explosion. 0 = off.
# MF_DRIFT_HL (min): override DRIFT_HALF_LIFE_MIN (faster drift decay).
NEWCOMER_ANCHOR = int(os.environ.get("MF_NEWCOMER_ANCHOR", "0"))
SIZE_CAP = float(os.environ.get("MF_SIZE_CAP", "1.0"))  # v4: cap simulated size growth above last-mark level (round 3, stacked with SIZE_TAU 480)
DRIFT_HL = float(os.environ.get("MF_DRIFT_HL", str(DRIFT_HALF_LIFE_MIN)))
# Output floors.
MIN_P = 0.002        # min probability to emit a probabilities row
MIN_SIZE_P = 0.001   # min probability to emit a sizes row
# Persistence blend: the Monte Carlo diffuses too far at short horizons — it
# gives place 1 to newcomers in ~95% of trials and extrapolates the rally, so
# its size distributions sit ~+0.4 dex above the last mark (real practice
# -46.79 vs baseline -40.41, while the proxy said the opposite). Blend each
# read's place probabilities and size histograms with the house baseline's
# persistence forecast (0.30 stay / 0.10 neighbours; sizes lognormal sd 0.35
# around the last mark's place level). The weight is regime-aware: it scales
# with the recent top-5 overlap (calm market -> trust persistence) and decays
# with the read's horizon, so the 7-13 h final reads stay pure Monte Carlo.
BLEND_TAU_MIN = float(os.environ.get("MF_BLEND_TAU", "120.0"))
BLEND_SCALE = float(os.environ.get("MF_BLEND_SCALE", "1.0"))
P_STAY = 0.30
P_NEIGHBOUR = 0.10
SIZE_SD = 0.35

# Empirical newcomer size deltas vs the previous mark's place-1 size
# (practice history, top-5 entrants only) — fallback when the history yields
# no top-100 newcomer deltas of its own (see build_newcomer_deltas).
NEWCOMER_DELTAS = [0.21, 0.15, 0.08, 0.01, -0.01, -0.12, -0.16, -0.40, -0.66, -0.88]
# Empirical graduation jumps (curve -> pool), practice history, n=6.
GRAD_JUMPS = [-0.88, -0.82, -0.74, -0.20, 0.26, 0.54]


def build_newcomer_deltas(marks):
    """Deltas (size - previous mark's place-1 size) for newborn coins that
    enter a mark's top 5 without being in the previous mark's rows: the
    contender cohort. Falls back to the practice's ten such deltas when the
    training history shows too few (a short history)."""
    deltas = []
    for prev, cur in zip(marks, marks[1:]):
        if not prev["rows"] or not cur["rows"]:
            continue
        prev_mints = {r["mint"] for r in prev["rows"]}
        p1_prev = size_of(prev["rows"][0]["market_cap_lamports"])
        for r in cur["rows"][:5]:
            if r["mint"] not in prev_mints:
                deltas.append(size_of(r["market_cap_lamports"]) - p1_prev)
    if len(deltas) < 10:
        return list(NEWCOMER_DELTAS)
    return deltas


def top5_persist_rate(marks):
    """Recency-weighted (2,1) mean top-5 overlap over the last <=2 transitions
    of the training history; 0.5 when there is no transition to learn from.
    The practice history shows the regime flip this captures: early transitions
    overlap 1/5 (wild churn) while the last two overlap 4/5 (calm)."""
    overlaps = []
    for prev, cur in zip(marks, marks[1:]):
        if not prev["rows"] or not cur["rows"]:
            continue
        prev5 = {r["mint"] for r in sorted(prev["rows"], key=lambda r: -int(r["market_cap_lamports"]))[:5]}
        cur5 = {r["mint"] for r in sorted(cur["rows"], key=lambda r: -int(r["market_cap_lamports"]))[:5]}
        overlaps.append(len(prev5 & cur5) / 5.0)
    if not overlaps:
        return 0.5
    if len(overlaps) == 1:
        return overlaps[0]
    return (2 * overlaps[-1] + overlaps[-2]) / 3.0


def lognormal_bin_probs(mean, sd, edges):
    """P(X in bin) for X ~ N(mean, sd) over the 82 bins from edges."""
    out = []
    for b in range(len(edges) + 1):
        lo = float("-inf") if b == 0 else edges[b - 1]
        hi = float("inf") if b == len(edges) else edges[b]
        plo = 0.0 if lo == float("-inf") else 0.5 * (1.0 + math.erf((lo - mean) / (sd * math.sqrt(2.0))))
        phi = 1.0 if hi == float("inf") else 0.5 * (1.0 + math.erf((hi - mean) / (sd * math.sqrt(2.0))))
        out.append(max(0.0, phi - plo))
    return out


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def size_of(lamports):
    return math.log10(int(lamports)) - 9


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and m.get("rows"):
                marks.append(m)
    marks.sort(key=lambda m: parse_time(m["at"]))
    return task, marks


def build_series(marks):
    """mint -> dict with per-mark observations (oldest first)."""
    series = {}
    for mi, m in enumerate(marks):
        t = parse_time(m["at"])
        for r in m["rows"]:
            s = series.get(r["mint"])
            if s is None:
                s = series[r["mint"]] = {
                    "mint": r["mint"], "obs": [], "created": parse_time(r["created_at"]),
                    "mayhem": bool(r.get("mayhem")),
                }
            s["obs"].append((t, size_of(r["market_cap_lamports"]), r["venue"],
                             int(r["rank"]), float(r["quote_reserves_lamports"]) / 1e9,
                             bool(r.get("complete"))))
    return series


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_input(input_dir)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    edges = task["size"]["edges"]
    places = int(task["places"])
    cadence = float(task.get("cadence_minutes", 10))
    cut_t = parse_time(task["history_until"]) if task.get("history_until") else parse_time(marks[-1]["at"])
    read_times = [parse_time(r) for r in task["reads"]]
    gaps_min = [max(0.0, (rt - cut_t) / 60.0) for rt in read_times]
    max_gap = max(gaps_min)
    n_steps = int(math.ceil(max_gap / cadence)) if max_gap > 0 else 0
    # step index (0-based count of cadence steps) at which each read lands
    read_steps = [max(0, int(round(g / cadence))) for g in gaps_min]

    series = build_series(marks)
    last_mark = marks[-1]
    last_t = parse_time(last_mark["at"])

    # ---- empirical step pool (centered 10-min log changes) ----------------
    steps = []
    for s in series.values():
        obs = s["obs"]
        for a, b in zip(obs, obs[1:]):
            dt_min = (b[0] - a[0]) / 60.0
            if dt_min <= 0:
                continue
            # rescale to one cadence step
            steps.append((b[1] - a[1]) * (cadence / dt_min))
    if not steps:
        steps = [0.0]
    mean_step = sum(steps) / len(steps)
    centered = [x - mean_step for x in steps]
    if len(centered) < STEP_POOL_MIN:
        # Degenerate history (e.g. a single mark): pad with a generic step law
        # so the simulation still has realistic volatility.
        pad = random.Random(SEED ^ 0x5EED)
        centered.extend(pad.gauss(0.0, STEP_FALLBACK_SD) for _ in range(STEP_POOL_MIN - len(centered)))
    n_steps_pool = len(centered)
    newcomer_deltas = build_newcomer_deltas(marks)

    # last mark's rows, ranked by size; keep pool coins, top sizes, jumpers.
    last_rows = sorted(last_mark["rows"], key=lambda r: int(r["market_cap_lamports"]), reverse=True)
    cands = []
    seen = set()
    # all pool coins first (they hold the top places), then by size
    for r in last_rows:
        if r["venue"] == "pool":
            cands.append(r["mint"])
            seen.add(r["mint"])
    for r in last_rows:
        if r["mint"] not in seen:
            cands.append(r["mint"])
            seen.add(r["mint"])
        if len(cands) >= CAND_MAX:
            break
    # jumpers: coins whose last observed 10-min step was >= +0.25, even if small
    for s in series.values():
        obs = s["obs"]
        if len(obs) >= 2 and obs[-1][1] - obs[-2][1] >= 0.25 and s["mint"] not in seen:
            cands.append(s["mint"])
            seen.add(s["mint"])
    cands = cands[:CAND_MAX]

    # per-candidate state
    cand_size = []
    cand_drift = []
    cand_venue = []  # 0 curve, 1 pool, 2 migrating
    cand_qr = []
    cand_age_min = []
    for mint in cands:
        s = series[mint]
        obs = s["obs"]
        last = obs[-1]
        cand_size.append(last[1])
        # drift: shrunk mean of last up-to-2 steps
        d = 0.0
        if len(obs) >= 2:
            seg = obs[-3:]
            vals = [(seg[i + 1][1] - seg[i][1]) for i in range(len(seg) - 1)]
            if vals:
                d = sum(vals) / len(vals)
        d = DRIFT_SHRINK * d + (1.0 - DRIFT_SHRINK) * mean_step
        if last[2] == "pool":
            d += POOL_DRIFT_BONUS
        d = max(DRIFT_CLIP[0], min(DRIFT_CLIP[1], d))
        cand_drift.append(d)
        cand_venue.append(0 if last[2] == "curve" else (1 if last[2] == "pool" else 2))
        cand_qr.append(last[4])
        cand_age_min.append(max(0.0, (last_t - s["created"]) / 60.0))
    n_cand = len(cands)
    last_place1 = max(cand_size) if cand_size else 3.0
    persist_rate = top5_persist_rate(marks)

    # ---- trials ------------------------------------------------------------
    trials = MAX_TRIALS
    if n_steps > 0 and n_cand > 0:
        trials = int(STEP_BUDGET // max(1, n_steps * n_cand))
        trials = max(MIN_TRIALS, min(MAX_TRIALS, trials))
    rng = random.Random(SEED)
    n_reads = len(read_times)
    place_counts = [[[0] * n_cand for _ in range(n_reads)] for _ in range(places)]
    newcomer_counts = [[0] * n_reads for _ in range(places)]
    place_cap_samples = [[[] for _ in range(n_reads)] for _ in range(places)]
    total_samples = trials * n_reads
    # ref_level: the place-1 size as of the previous read (cut level first).
    # Newcomers are anchored to it, so the top level can ratchet up when a
    # newborn exceeds it, as the practice place-1 did (+0.13 per 10 min).
    # (Reset to the cut level at the start of each trial, below.)

    # Precompute drift decay per step: drift * 0.5**(t / half_life)
    decay = [0.5 ** ((i * cadence) / DRIFT_HL) for i in range(n_steps + 1)]
    read_step_set = {}
    for ri, rs in enumerate(read_steps):
        read_step_set.setdefault(rs, []).append(ri)

    for _ in range(trials):
        sizes = list(cand_size)
        drifts = list(cand_drift)
        venues = list(cand_venue)
        qrs = list(cand_qr)
        # step through time; record at read steps
        for step in range(n_steps + 1):
            if step in read_step_set:
                # newcomers: sizes relative to the best incumbent size at this
                # read (the level the market has reached); the contender deltas
                # are measured against the previous mark's place-1, so this
                # anchor carries the level's growth with it.
                ref = last_place1 if NEWCOMER_ANCHOR else (max(sizes) if sizes else last_place1)
                entries = [(sizes[i], i) for i in range(n_cand)]
                for _ in range(NEWCOMERS_PER_READ):
                    delta = newcomer_deltas[rng.randrange(len(newcomer_deltas))]
                    entries.append((ref + delta, -1))
                top = heapq.nlargest(places, entries, key=lambda e: e[0])
                for ri in read_step_set[step]:
                    for k, (cap, idx) in enumerate(top):
                        place_cap_samples[k][ri].append(cap)
                        if idx >= 0:
                            place_counts[k][ri][idx] += 1
                        else:
                            newcomer_counts[k][ri] += 1
            if step == n_steps:
                break
            dec = decay[step]
            for i in range(n_cand):
                mv = drifts[i] * dec + STEP_SHRINK * centered[rng.randrange(n_steps_pool)]
                sizes[i] += mv
                if REVERT_KAPPA > 0.0:
                    sizes[i] -= REVERT_KAPPA * (sizes[i] - cand_size[i])
                if venues[i] == 0:  # curve: maybe graduate
                    qr = qrs[i] * (10.0 ** mv) if mv > 0 else qrs[i]
                    qrs[i] = qr
                    p_grad = GRAD_BASE + 0.15 * min(1.0, (qr / GRAD_QR_REF) ** 2)
                    if rng.random() < p_grad:
                        venues[i] = 1
                        sizes[i] += GRAD_JUMPS[rng.randrange(len(GRAD_JUMPS))]
                if SIZE_CAP > 0.0:
                    cap_i = cand_size[i] + SIZE_CAP
                    if sizes[i] > cap_i:
                        sizes[i] = cap_i
                # age-based drift fade is in `dec`; nothing else per coin

    # ---- assemble forecast --------------------------------------------------
    # Per-read blend weights: regime-aware persistence, decaying with horizon.
    weights = []
    weights_size = []
    for ri in range(n_reads):
        w = persist_rate * math.exp(-gaps_min[ri] / BLEND_TAU_MIN) * BLEND_SCALE
        weights.append(min(1.0, max(0.0, w)))
        ws = persist_rate * math.exp(-gaps_min[ri] / SIZE_TAU_MIN) * BLEND_SCALE
        weights_size.append(min(1.0, max(0.0, ws)))
    # Last mark's ranking (by size) for the persistence forecast.
    last_ranked = [r["mint"] for r in last_rows]
    last_sizes = [size_of(r["market_cap_lamports"]) for r in last_rows]

    def persist_place_probs(k):
        """Baseline persistence distribution for place k (0-based): 0.30 on
        the coin at #k, 0.10 on each neighbour."""
        d = {last_ranked[k]: P_STAY}
        if k > 0:
            d[last_ranked[k - 1]] = d.get(last_ranked[k - 1], 0.0) + P_NEIGHBOUR
        if k + 1 < len(last_ranked):
            d[last_ranked[k + 1]] = d.get(last_ranked[k + 1], 0.0) + P_NEIGHBOUR
        return d

    def bin_of(cap):
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) >> 1
            if edges[mid] <= cap:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # Blended per-place probabilities and size histograms, averaged over reads.
    blended = [dict() for _ in range(places)]       # place -> mint -> p
    blended_size = [dict() for _ in range(places)]  # place -> bin -> p
    for k in range(places):
        per = persist_place_probs(k)
        ln_probs = lognormal_bin_probs(last_sizes[k], SIZE_SD, edges)
        for ri in range(n_reads):
            w = weights[ri]
            for i in range(n_cand):
                c = place_counts[k][ri][i]
                if c:
                    pmc = c / trials
                    mint = cands[i]
                    blended[k][mint] = blended[k].get(mint, 0.0) + ((1.0 - w) * pmc) / n_reads
            for mint, pp in per.items():
                blended[k][mint] = blended[k].get(mint, 0.0) + (w * pp) / n_reads
            hist = [0] * (len(edges) + 1)
            for cap in place_cap_samples[k][ri]:
                hist[bin_of(cap)] += 1
            tot = len(place_cap_samples[k][ri])
            for b in range(len(edges) + 1):
                pmc = (hist[b] / tot) if tot else 0.0
                pb = ((1.0 - weights_size[ri]) * pmc + weights_size[ri] * ln_probs[b]) / n_reads
                if pb > 0:
                    blended_size[k][b] = blended_size[k].get(b, 0.0) + pb

    probabilities = []
    for k in range(places):
        rows = [(p, mint) for mint, p in blended[k].items() if p >= MIN_P]
        rows.sort(key=lambda x: (-x[0], x[1]))
        for p, mint in rows[:100]:
            probabilities.append({"mint": mint, "place": k + 1, "p": round(p, 6)})
    # top5: highest P(in top5), ordered by expected rank
    in_top5 = {}
    exp_rank = {}
    for k in range(places):
        for mint, p in blended[k].items():
            in_top5[mint] = in_top5.get(mint, 0.0) + p
            exp_rank[mint] = exp_rank.get(mint, 0.0) + (k + 1) * p
    ranked = sorted(in_top5.items(), key=lambda kv: (-kv[1], exp_rank[kv[0]] / kv[1], kv[0]))
    top5 = [m for m, _ in ranked[:places]]
    # pad (should not happen) with last mark's top coins
    if len(top5) < places:
        for r in last_rows:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) >= places:
                break
    # sizes: blended histograms, keep sum <= 1 after 6-dp rounding
    sizes_out = []
    for k in range(places):
        rows = [(p, b) for b, p in blended_size[k].items() if p >= MIN_SIZE_P]
        # keep sum <= 1 after 6-dp rounding: round down like the baseline
        ssum = 0.0
        for p, b in sorted(rows, key=lambda x: -x[0]):
            rp = math.floor(p * 1_000_000) / 1_000_000
            if rp > 0 and ssum + rp <= 1.0:
                sizes_out.append({"place": k + 1, "bin": b, "p": rp})
                ssum += rp
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes_out}
    if os.environ.get("MAKEFUN_DEBUG"):
        print(f"[debug] persist_rate={persist_rate:.2f} weights={[round(w, 2) for w in weights]}", file=sys.stderr)
        for k in range(places):
            named_sum = sum(r["p"] for r in probabilities if r["place"] == k + 1)
            nc = sum(newcomer_counts[k])
            caps = [c for ri in range(n_reads) for c in place_cap_samples[k][ri]]
            mean_cap = (sum(caps) / len(caps)) if caps else float("nan")
            print(f"[debug] place {k+1}: named_sum={named_sum:.3f} "
                  f"newcomer_share={nc / total_samples:.3f} "
                  f"mean_cap={mean_cap:.2f} n_cap={len(caps)}", file=sys.stderr)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
