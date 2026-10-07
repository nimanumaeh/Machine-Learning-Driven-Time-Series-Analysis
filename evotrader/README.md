# evotrader

A play-money BTC arena. A population of small agents trades the BTCUSDT
perpetual (long or short, 1x to 125x). Agents that lose money die, and agents
that make money split into mutated children. Nothing here touches a real account.

## The environment: exactly one BTCUSDT perpetual account per agent

Modelled on Binance USDⓈ-M futures, one-way mode, isolated margin, market
orders (`exchange.py`). Everything an agent gains or loses comes from these
rules and nothing else:

| What | How it's charged | Default |
|---|---|---|
| Price | real 1-minute BTCUSDT perpetual candles | |
| Fill | decided on a candle's close, filled at the next minute's open | |
| Bid/ask spread | buys fill at price + half spread, sells at price - half spread | 0.05 USDT (tick is 0.10) |
| Exchange fee | on the notional of every fill, from the wallet | 0.05% (Binance VIP 0 taker; KuCoin 0.06%) |
| Funding | position x mark price x rate at every settlement, longs pay shorts when positive | real historical rates |
| Leverage | margin = notional / leverage; bigger positions get lower max leverage (Binance brackets) | 1x to 125x |
| Liquidation | when margin + unrealised PnL at the mark price reaches maintenance margin; the position's margin is lost | Binance maintenance tiers |
| Order size | multiples of 0.001 BTC; at least 100 USDT to open or add | |

Change the fee and spread to your own account's numbers with `--taker-fee`
and `--half-spread`. Bracket and lot-size defaults are in `config.py`.

The ecology (`world.py`) only reads equity and never adds or removes money:
- An agent dies below 50% of its reference equity.
- It splits its account exactly in half with a mutated child above 125%, at most once every 7 days.
- When the population is full, a newborn replaces the weakest agent.
- Random newcomers keep at least 200 agents alive and every timeframe explored.

A genome is:
- a small neural net, mapping market features plus the agent's own position, unrealised PnL and distance to liquidation, to a target position between full short and full long;
- a timeframe (1m to 1h);
- a leverage setting;
- the share of equity it puts up as margin;
- a rebalance dead-band.

Two yardsticks sit outside the ecology. Random agents under the same rules, never selected, show what luck earns. A 1x buy-and-hold shows what just owning BTC earns.

## Data (free, no API key)

```bash
python -m evotrader download                  # Aug 2017 -> yesterday, ~4.8M minutes
```

Sources:
- **Perpetual era:** the BTCUSDT perpetual from `data.binance.vision`: trade candles, mark-price candles and every funding settlement (September 2019 on).
- **Before September 2019:** the perpetual didn't exist yet, so the builder uses spot prices as both trade and mark price and charges 0.01% funding per 8 hours, the rate the perpetual settles at when it trades in line with spot. These rows are labelled `spot` in the CSV's last column; real ones say `perp`.
- **The current month:** its funding comes from `fapi.binance.com`. If that is blocked where you are, the data set ends at the last fully archived month.

Raw archives are cached in `data/raw`, and re-running only adds new days.
In a Claude Code cloud environment, allow `data.binance.vision` and
`fapi.binance.com` in the environment's network settings.

## Run

```bash
python -m evotrader evolve --data data/BTCUSDT_perp_1m.csv --until 2025-10-01   # keep the last year unseen
python -m evotrader evolve --data data/BTCUSDT_perp_1m.csv --resume              # then let them meet it
python -m evotrader live                       # carry on in real time, forever; rerun to resume
python -m evotrader report --run runs/history
```

History is replayed in order, so every decision is made on bars the
population has never seen. `live` continues the same run from where history
stopped, catching up gap-free. It holds back a bar that ends on a funding
settlement until Binance publishes that rate.

`python -m evotrader synth` makes synthetic BTC-like data for trying things out offline.

## Tests

```bash
python -m pytest tests
```
