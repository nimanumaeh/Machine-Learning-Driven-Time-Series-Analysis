# evotrader

A play-money BTC arena where a population of small trading agents lives or
dies by what it earns. Nothing here touches a real account.

This is the substrate: free BTC data, candle and feature books for several
timeframes, a market with fees and slippage, and an evolutionary ecosystem
(metabolism, death, reproduction with mutation, carrying capacity, a random
control group, a buy-and-hold benchmark) with SQLite logging and checkpoints.

## Data (free, no API key)

```bash
# deep history: data.binance.vision bulk zips, plus REST for today
python -m evotrader download --seconds 60 --days 365      # 1-minute candles
python -m evotrader download --seconds 1 --days 7         # 1-second candles
# offline: synthetic BTC-like bars for testing the machinery
python -m evotrader synth --days 60
```

Hosts used: `data.binance.vision` (archives), `data-api.binance.vision`
(REST and live; falls back to `api.binance.com`, then `api.binance.us`).
In a Claude Code cloud environment these have to be allowed under the
environment's network access settings.

## Run

```bash
python -m evotrader evolve --data data/BTCUSDT_1m.csv --run runs/nursery
python -m evotrader live --seed-from runs/nursery --run runs/live   # forever; rerun to resume
python -m evotrader report --run runs/live
```

`evolve` replays history in order, so every decision is made on bars the
population has not seen yet. `live` polls 1-second candles and catches up
without gaps after an outage. Ecosystem knobs live in `config.py`.

## Tests

```bash
python -m pytest tests
```
