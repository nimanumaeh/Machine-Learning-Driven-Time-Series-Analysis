# evotrader

A play-money BTCUSDT perpetual arena. A population of agents trades it long
or short, from 1x to 125x. Agents that lose their money die; agents that make
money split into children. Nothing here touches a real account.

- `docs/data.md`: what the arena is made of: the exact data streams, which years, and how rows become a world.
- `docs/relevance-realization-design.md`: the theory and the math of the relevance-realizing agents.

## Layers

| Layer | Module | What it is |
|---|---|---|
| Data | `data.py` | one row per minute, 24 columns: perp trades and order flow, mark price, funding, premium, spot, open interest, crowd positioning (free Binance archives + REST) |
| Body | `exchange.py` | one exact Binance-style BTCUSDT perpetual account per agent: isolated margin, liquidation by mark price, fees, spread, funding, lot size |
| Perception | `aspects.py` | 186 causal aspects (stream x operator x scale) an agent can attend to |
| Mind | `mind.py`, `body.py` | relevance-realizing agents: scarce attention, an online arena model, the affordance value of *their own* account, salience derived from it |
| Baseline mind | `brains.py` | small fixed neural nets shaped only by evolution |
| Ecology | `world.py` | equity is life: death, reproduction by account splitting, carrying capacity, newcomers, a never-selected control group |

## Use

```bash
python -m evotrader download                         # Aug 2017 -> yesterday, resumable
python -m evotrader coverage                         # which streams exist, month by month
python -m evotrader evolve --until 2025-10-01        # live through history, last year sealed
python -m evotrader evolve --resume                  # then meet the sealed year
python -m evotrader live --run runs/live             # history first, then real time, forever
python -m evotrader report --run runs/history
```

- `--brain net` runs the evolved-net baseline instead of the relevance-realizing agents.
- `--attention-mode random` or `fixed` runs the ablations.
- `--taker-fee` and `--half-spread` set your own account's costs.

Offline, there are two ways to try the machinery on a synthetic market with a planted effect (or with `--null`, without one):

```bash
python -m evotrader synth --days 90 && python -m evotrader evolve --store data/synthetic --symbol SYNTH
python experiments/relevance_check.py --world planted --attention salience
```

In a Claude Code cloud environment, allow `data.binance.vision`,
`fapi.binance.com` and `api.binance.com` in the environment's network
settings.

## Tests

```bash
python -m pytest tests
```
