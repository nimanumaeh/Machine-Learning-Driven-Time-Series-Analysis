"""Tunable knobs. Everything is play money.

The market section mirrors one Binance USDⓈ-M BTCUSDT perpetual account per
agent (one-way mode, isolated margin, market orders). Check the defaults
against your own fee tier and the exchange's current contract rules.
"""

from dataclasses import dataclass, field, asdict

# Binance BTCUSDT leverage brackets: (position notional cap in USDT, max
# leverage, maintenance margin rate). Bigger positions get less leverage.
BTCUSDT_BRACKETS = (
    (50_000, 125, 0.004),
    (250_000, 100, 0.005),
    (3_000_000, 50, 0.01),
    (15_000_000, 20, 0.025),
    (30_000_000, 10, 0.05),
    (80_000_000, 5, 0.10),
    (100_000_000, 4, 0.125),
    (200_000_000, 3, 0.15),
    (300_000_000, 2, 0.25),
    (500_000_000, 1, 0.50),
)

# Candle sizes (seconds) an agent's genome can choose to think in.
ALL_TIMEFRAMES = (60, 180, 300, 600, 900, 1800, 3600)


@dataclass
class Config:
    # --- market: exactly what a BTCUSDT perpetual account pays -------------
    taker_fee: float = 0.0005      # 0.05% of position value per fill (KuCoin 0.06%)
    half_spread: float = 0.05      # USDT; buys fill at mid + this, sells at mid - this
    qty_step: float = 0.001        # BTC lot size
    min_notional: float = 100.0    # smallest order that opens or adds, USDT
    max_leverage: int = 125
    brackets: tuple = BTCUSDT_BRACKETS
    initial_capital: float = 1000.0

    # --- ecology: rules of life and death; never touches anyone's money ----
    capacity: int = 400            # max living agents
    min_population: int = 200      # random newcomers top the population up to this
    min_per_niche: int = 8         # ...and keep every timeframe explored
    death_ratio: float = 0.5       # dies below this fraction of its reference equity
    repro_ratio: float = 1.25      # splits in two above this multiple of it
    min_repro_age_days: float = 7.0
    crossover_prob: float = 0.2
    timeframe_mutation_prob: float = 0.05

    # --- policy network ----------------------------------------------------
    hidden: int = 8
    init_sigma: float = 0.1        # starting mutation step size (self-adapts)

    # --- yardsticks (outside the ecology) -----------------------------------
    n_control: int = 40            # random agents that never reproduce

    snapshot_every_s: int = 3600
    timeframes: tuple = field(default_factory=lambda: ALL_TIMEFRAMES)
    seed: int = 0

    def to_dict(self):
        d = asdict(self)
        d["timeframes"] = list(self.timeframes)
        d["brackets"] = [list(b) for b in self.brackets]
        return d
