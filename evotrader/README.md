# evotrader

A play-money BTCUSDT perpetual arena. A population of agents trades it long
or short, from 1x to 125x. Agents that lose their money die; agents that make
money split into children. Nothing here touches a real account.

- `docs/data.md`: what the arena is made of: the exact data streams, which years, and how rows become a world.
- `docs/relevance-realization-design.md`: the theory and the math of the relevance-realizing agents.
- `docs/planet.md`: the planet made of 1-second BTC data, and the smallest Vervaekean organisms that live on it.

## Layers

| Layer | Module | What it is |
|---|---|---|
| Data | `data.py` | one row per minute, 24 columns: perp trades and order flow, mark price, funding, premium, spot, open interest, crowd positioning (free Binance archives + REST) |
| Body | `exchange.py` | one exact Binance-style BTCUSDT perpetual account per agent: isolated margin, liquidation by mark price, fees, spread, funding, lot size |
| Receptors | `selfmade.py` | the raw columns, transduced generically; self-made agents build their own features from them |
| Mind (default) | `selfmade.py` | agents that build their own perception (feature programs over raw receptors, generate-and-test) and learn what each move is worth from their own counterfactual participation; salience over their own features |
| Engineered mind | `mind.py`, `aspects.py`, `body.py` | baseline: relevance realization over our 186 engineered aspects with our Kelly theory of value |
| Baseline mind | `brains.py` | small fixed neural nets shaped only by evolution |
| Ecology | `world.py` | equity is life: death, reproduction by account splitting, carrying capacity, newcomers, a never-selected control group |
| 1-second data | `data_seconds.py` | one row per second rebuilt from every perpetual trade (aggTrades), slow streams joined causally from the minute store |
| Planet | `planet.py` | a world made of the market: latitude is timescale (1 second at the equator to 1 day at the pole), longitude decides which senses exist; observatories and markers are built by its inhabitants |
| Life | `life.py` | very many minimal organisms as array columns: an exact account as body, self-made organs kept alive by salience, two opponent dials, a learned gate into a higher-order mode that rebuilds organs, migrates and builds |
| Watching | `planet_run.py`, `viewer.py` | censuses of life over time, checkpoints, and one self-contained HTML page per run |

## Use

```bash
python -m evotrader download                         # Aug 2017 -> yesterday, resumable
python -m evotrader coverage                         # which streams exist, month by month
python -m evotrader evolve --until 2025-10-01        # live through history, last year sealed
python -m evotrader evolve --resume                  # then meet the sealed year
python -m evotrader live --run runs/live             # history first, then real time, forever
python -m evotrader report --run runs/history
```

- `--brain self` (default) runs the self-made agents.
- `--brain rr` runs the engineered relevance-realizing baseline, and `--brain net` the evolved-net baseline.
- `--attention-mode random` or `fixed` runs the ablations.
- `--taker-fee` and `--half-spread` set your own account's costs.

The planet runs on the 1-second store, or on a synthetic market:

```bash
python -m evotrader seconds --from 2024-01-01              # every trade -> 1-second rows (after `download`)
python -m evotrader planet --from 2024-01-01 --run runs/planet
python -m evotrader planet --synthetic 10 --run runs/planet-synth          # or --null for nothing planted
python -m evotrader planet --run runs/planet --resume      # Ctrl-C saves; this carries on
python -m evotrader view --run runs/planet                 # runs/planet/planet.html
```

- `--width` sets the number of longitudes; the 8 latitudes are the timescales.
- `--population` is the floor below which newcomers arrive, and `--place-capacity` how many organisms a place holds at normal liquidity.

Offline, there are two ways to try the machinery on a synthetic market with a planted effect (or with `--null`, without one):

```bash
python -m evotrader synth --days 90 && python -m evotrader evolve --store data/synthetic --symbol SYNTH
python experiments/relevance_check.py --brain self --world planted
```

In a Claude Code cloud environment, allow `data.binance.vision`,
`fapi.binance.com` and `api.binance.com` in the environment's network
settings.

## Tests

```bash
python -m pytest tests
```
