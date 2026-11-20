# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shangjf98-max/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shangjf98-max/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, five quantiles (5/25/50/75/95%) per trophy. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## How it forecasts

Exophase refreshes a game's counts on its own schedule, so the recorded reads are flat for hours and then jump. A rate is a ratio, `earners / players_tracked`, and the two sides move very differently: at the first refresh of launch night the tracked players jumped 638 → 3,868 while a trophy's earners grew only ~25%. Fitting the rate itself across a refresh fits the jump, not the play.

`main.py` therefore splits the ratio and fits each side separately: it compresses the history to its distinct states (one per refresh), fits log(players_tracked) and per-trophy log(earners) against log-time since release by least squares over the last 24 hours of states (each trophy's slope shrunk toward the pooled slope), cross-checks with a direct logit(rate) trend fit, and divides the two extrapolations at the target. It enforces the logical constraints (the chapter chain is non-increasing, the platinum is at most the smallest other rate, the 100% trophy stays at 100%) and widens the band with a random-walk term plus the fit's own residual uncertainty. The flat-history practice case uses the first refresh named in the kit's rules (3,868 tracked, earners +25%).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` (one `{"trophies": [{"id", "q": [q05, q25, q50, q75, q95]}]}` object per case). Deterministic: the same input bytes give the same output bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
