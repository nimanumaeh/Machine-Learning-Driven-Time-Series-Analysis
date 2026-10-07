"""Tunable knobs for the ecosystem. Everything is play money."""

from dataclasses import dataclass, field, asdict


# Candidate timeframes (seconds) an agent's genome can choose from. Only those
# that are a whole multiple of the feed's base bar size are usable in a run.
ALL_TIMEFRAMES = (30, 60, 180, 300, 600, 900, 1800, 3600)


@dataclass
class Config:
    # --- market frictions -------------------------------------------------
    fee_bps: float = 10.0          # taker fee per side (Binance spot ~0.1%)
    slippage_bps: float = 2.0      # adverse price move on every fill
    allow_short: bool = False      # long-only by default (spot)

    # --- ecology ----------------------------------------------------------
    capacity: int = 400            # max living agents (carrying capacity)
    min_per_niche: int = 8         # immigrants keep every timeframe explored
    initial_capital: float = 1000.0
    metabolism_bps_per_day: float = 5.0   # cost of being alive, % of equity
    death_ratio: float = 0.6       # die when equity < ratio * reference
    repro_ratio: float = 1.25      # split when equity >= ratio * reference
    min_repro_age_days: float = 1.0
    crossover_prob: float = 0.2
    timeframe_mutation_prob: float = 0.05

    # --- policy network ---------------------------------------------------
    hidden: int = 8
    init_sigma: float = 0.1        # starting mutation step size (self-adapts)

    # --- benchmarks -------------------------------------------------------
    n_control: int = 40            # random immortal agents, never selected

    # --- bookkeeping ------------------------------------------------------
    snapshot_every_s: int = 3600   # log ecosystem state every N data-seconds
    timeframes: tuple = field(default_factory=lambda: ALL_TIMEFRAMES)
    seed: int = 0

    def to_dict(self):
        d = asdict(self)
        d["timeframes"] = list(self.timeframes)
        return d
