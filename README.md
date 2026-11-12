# pumpfun-top5-forecaster

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/TheFoolByte/pumpfun-top5-forecaster/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/TheFoolByte/pumpfun-top5-forecaster/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A Monte Carlo log-cap forecaster for pump.fun's top 5 by market cap, 24 hours out. It reads the house's 10-minute snapshots, simulates each candidate coin's log10 market cap forward as a jump-diffusion (empirical step resampling, shrunk recent-slope drift that decays with age, a graduation hazard for bonding-curve coins), adds a newborn-rocket cohort at every answer read — because half of the top 5 at any read are coins that were not in the previous snapshot — and averages honest place probabilities and place-cap distributions over the seven answer reads from `task.json`.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, deterministic (fixed seed, byte-identical output across runs), no network. It reads `task.json` and `history.ndjson` from the input directory and writes `forecast.json` to the output directory. Runtime is a few seconds on the 2-hour practice history and ~10 seconds on a full 11-hour, 66-mark final history (1 CPU).

## How it works

- **Per-coin paths**: log10 market cap follows a random walk whose steps are resampled from the history's own observed 10-minute steps (heavy-tailed: most coins sit still, a few jump +0.3 dex or more), plus a shrunk recent-slope drift that decays with a 2-hour half-life, so growth saturates instead of running away over the final's 7–13 hour horizon.
- **Graduation**: a bonding-curve coin near completion (quote reserves approaching ~85 SOL) can graduate to its pool, which historically reprices it down by ~0.3 dex on median; the model applies the observed graduation jumps.
- **Newcomers**: at each answer read the simulation adds a cohort of newborn rockets whose sizes are drawn from the observed distribution of newborn sizes relative to the current place-1 level. A place a newcomer takes stays unnamed in the forecast, which is exactly how the scorer credits it — the model never pretends to name a coin it has not seen.
- **Sizes**: each place's market-cap distribution is the simulated distribution of that place's cap, binned to the 82 bins from `task.json`, so the size forecast widens honestly with horizon instead of sitting on the last mark's level.

## Validation

- The local scorer is an exact port of the harness's canonical scoring code, verified against the worked example in the kit's `RULES.txt` (both reads, mean and tie-break, to 6 decimals).
- On rolling-cut backtests of the practice history (train on the first 2–4 marks, score on the later marks as answer reads), the model beats the house baseline on 4 of 5 cuts, by +5 to +31 log-score points, with the largest gains at the longest horizons (the final's regime). The one losing cut is a single 10-minute read where the incumbents happened to hold.
- Two runs on the same input produce byte-identical `forecast.json`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=e3r838hp&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
