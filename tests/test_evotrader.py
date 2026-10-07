import datetime as dt
import io
import pickle
import zipfile

import numpy as np
import pytest

from evotrader import body, data
from evotrader.aspects import AspectEngine, NAMES
from evotrader.brains import NetBrain
from evotrader.config import Config
from evotrader.data import COLUMNS, C, MINUTE
from evotrader.exchange import Accounts, Brackets
from evotrader.features import HISTORY, N_FEATURES, TimeframeBook
from evotrader.mind import RRBrain
from evotrader.synthetic import synthetic_rows
from evotrader.world import World


def cfg(**kw):
    c = Config(capacity=20, n_control=4, min_population=10, min_per_niche=1,
               timeframes=(60, 300), seed=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def account(capital=1000.0, **kw):
    a = Accounts(1, cfg(**kw))
    a.open(0, capital)
    return a


def trade(a, target_q, price, lev):
    return a.execute(np.array([0]), np.array([float(target_q)]), price, np.array([float(lev)]))


def row(t, price, funding=0.0, mark=None):
    r = np.full(len(COLUMNS), np.nan)
    m = price if mark is None else mark
    r[:9] = (t, price, price, price, price, 1.0, price, 10, 0.5)
    r[C["mark_high"]] = r[C["mark_low"]] = r[C["mark_close"]] = m
    r[C["funding_rate"]] = funding
    r[C["perp"]] = 1.0
    return r


def flat_rows(n, price=60000.0):
    return [row(k * MINUTE, price) for k in range(n)]


# --------------------------------------------------------------- accounting
def test_open_long_pays_fee_and_half_spread_only():
    a = account()
    trade(a, 0.1, 60000.0, 10)
    assert a.q[0] == pytest.approx(0.1)
    assert a.entry[0] == pytest.approx(60000.05)
    assert a.margin[0] == pytest.approx(0.1 * 60000.05 / 10)
    fee = 0.1 * 60000.05 * 0.0005
    assert a.fees[0] == pytest.approx(fee)
    assert a.equity(60000.0)[0] == pytest.approx(1000 - fee - 0.1 * 0.05)


def test_round_trip_profit_is_move_minus_costs():
    a = account()
    trade(a, 0.1, 60000.0, 10)
    trade(a, 0.0, 61000.0, 10)
    fees = 0.1 * 60000.05 * 0.0005 + 0.1 * 60999.95 * 0.0005
    assert a.q[0] == 0 and a.margin[0] == 0
    assert a.wallet[0] == pytest.approx(1000 + 0.1 * 1000 - fees - 0.1 * 0.1)


def test_short_profits_when_price_falls():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, -0.1, 60000.0, 20)
    assert a.equity(57000.0)[0] == pytest.approx(1000 + 300)
    trade(a, 0.0, 57000.0, 20)
    assert a.wallet[0] == pytest.approx(1300)


def test_flip_closes_then_opens_the_other_way():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 10)
    trade(a, -0.05, 62000.0, 10)
    assert a.q[0] == pytest.approx(-0.05)
    assert a.entry[0] == pytest.approx(62000.0)
    assert a.margin[0] == pytest.approx(0.05 * 62000 / 10)
    assert a.equity(62000.0)[0] == pytest.approx(1200.0)


def test_partial_close_releases_margin_proportionally():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 10)
    trade(a, 0.04, 60000.0, 10)
    assert a.q[0] == pytest.approx(0.04)
    assert a.margin[0] == pytest.approx(240.0)
    assert a.wallet[0] == pytest.approx(760.0)


def test_orders_respect_lot_size_min_notional_and_wallet():
    a = account()
    trade(a, 0.0014, 60000.0, 10)                  # 0.001 BTC = 60 USDT < 100 minimum
    assert a.q[0] == 0
    trade(a, 1.0, 60000.0, 1)                      # can only margin what the wallet holds
    assert a.q[0] == pytest.approx(0.016)
    assert a.wallet[0] >= 0


def test_liquidation_prices_match_binance_isolated_formula():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 50)                     # margin 120, maintenance rate 0.4%
    long_liq = (120 - 0.1 * 60000) / (0.1 * 0.004 - 0.1)
    assert a.liquidation_price(np.array([0]))[0] == pytest.approx(long_liq)
    assert len(a.liquidate(60000.0, long_liq + 0.01)) == 0
    assert list(a.liquidate(60000.0, long_liq - 0.01)) == [0]
    assert a.q[0] == 0 and a.liquidations[0] == 1
    assert a.wallet[0] == pytest.approx(880.0)     # isolated: only the margin is lost

    b = account(taker_fee=0.0, half_spread=0.0)
    trade(b, -0.1, 60000.0, 50)
    short_liq = (120 + 0.1 * 60000) / (0.1 * 0.004 + 0.1)
    assert b.liquidation_price(np.array([0]))[0] == pytest.approx(short_liq)
    assert list(b.liquidate(short_liq + 0.01, 60000.0)) == [0]


def test_funding_longs_pay_shorts_receive():
    a = Accounts(2, cfg(taker_fee=0.0, half_spread=0.0))
    a.open(0, 1000.0)
    a.open(1, 1000.0)
    a.execute(np.array([0, 1]), np.array([0.1, -0.1]), 60000.0, np.array([10.0, 10.0]))
    a.settle_funding(0.0001, 60000.0)
    assert a.equity(60000.0) == pytest.approx([999.4, 1000.6])


def test_brackets_cap_size_by_leverage_and_set_maintenance():
    br = Brackets(Config().brackets)
    assert br.max_notional(np.array([125, 100, 50, 20])) == pytest.approx([5e4, 2.5e5, 3e6, 1.5e7])
    mmr, cum = br.tier(np.array([40_000.0, 60_000.0]))
    assert mmr == pytest.approx([0.004, 0.005]) and cum == pytest.approx([0.0, 50.0])


def test_split_halves_a_leveraged_account_exactly():
    a = Accounts(2, cfg())
    a.open(0, 1000.0)
    trade(a, 0.1, 60000.0, 25)
    before = a.equity(61000.0)[0]
    liq = a.liquidation_price(np.array([0]))[0]
    a.split(0, 1)
    assert a.equity(61000.0) == pytest.approx([before / 2, before / 2])
    assert a.liquidation_price(np.array([0, 1])) == pytest.approx([liq, liq])


# -------------------------------------------------------------------- world
def test_flat_market_loses_exactly_the_fees():
    c = cfg(half_spread=0.0)
    w = World(c, NetBrain(c))
    for r in flat_rows(3 * 1440):
        w.step(r)
    s = w.snapshot()
    assert s["totals"]["fees"] > 0
    assert s["net_pnl"] == pytest.approx(-s["totals"]["fees"], abs=1e-6)


def test_targets_fill_at_next_minute_open():
    c = cfg(taker_fee=0.0, half_spread=0.0)
    w = World(c, NetBrain(c))
    for r in flat_rows(10):
        w.step(r)
    i = int(np.nonzero(w.alive & ~w.control)[0][0])
    w.lev[i], w.frac[i] = 2.0, 0.5
    w.pending[i] = 1.0                              # 1x long, decided at this close
    w.step(row(10 * MINUTE, 61000.0))
    assert w.acct.entry[i] == pytest.approx(61000.0)
    assert w.exposure(np.array([i]))[0] == pytest.approx(1.0, abs=0.07)


@pytest.mark.parametrize("brain", [NetBrain, RRBrain])
def test_checkpoint_resumes_deterministically(brain):
    rows = synthetic_rows(3000, seed=5)
    c = cfg()
    a = World(c, brain(c))
    for r in rows[:1800]:
        a.step(r)
    b = pickle.loads(pickle.dumps(a))
    for r in rows[1800:]:
        a.step(r)
        b.step(r)
    assert np.array_equal(a.equity(), b.equity())
    assert a.next_id == b.next_id


def test_relevance_realizing_world_runs_clean():
    c = cfg()
    w = World(c, RRBrain(c))
    for r in synthetic_rows(2 * 1440, seed=7):
        w.step(r)
    assert np.all(np.isfinite(w.equity()[w.alive]))
    assert w.brain.nupd[w.alive].max() > 100        # the arena models are learning
    s = w.snapshot()["mind"]
    assert s["salience_by_stream"] and abs(sum(s["salience_by_stream"].values()) - 1) < 1e-6


# ------------------------------------------------------------------- body
def test_affordance_sensitivities_match_finite_differences():
    args = dict(fund=0.0001, lam=0.5, L=10.0, xmax=5.0, cost=0.0006, kappa=2.0, grid=4001)
    mu, sig = 0.003, 0.02
    v, x = body.affordance(mu, sig, **args)
    e = 1e-6
    dmu = (body.affordance(mu + e, sig, **args)[0] - body.affordance(mu - e, sig, **args)[0]) / (2 * e)
    ls = np.log(sig)
    dls = (body.affordance(mu, np.exp(ls + 1e-5), **args)[0]
           - body.affordance(mu, np.exp(ls - 1e-5), **args)[0]) / 2e-5
    a, b = body.sensitivities(x, sig, 10.0, 2.0)
    assert dmu == pytest.approx(a, rel=1e-3)
    assert dls == pytest.approx(b, rel=1e-3)


def test_expected_growth_matches_exact_expectation():
    """Second-order body model vs exact E[log(1 + x (e^r - 1))], r ~ N(mu, sig^2).

    The neglected terms are of order (x^2 + |x|) mu^2: about 1e-5 here, where the
    per-period Sharpe (0.2) is far larger than markets offer.
    """
    z, wts = np.polynomial.hermite_e.hermegauss(80)
    wts = wts / wts.sum()
    mu, sig = 0.002, 0.01
    r = mu + sig * z
    for x in (1.0, 1.5, -1.5):                     # 2x leverage: liquidation out of reach
        exact = np.sum(wts * np.log1p(x * np.expm1(r))) - 0.0006 * abs(x)
        model = body.expected_growth(np.array(x), mu, sig, 0.0, 0.0, 2.0, 0.0006, 1.0)
        assert model == pytest.approx(exact, abs=(x * x + abs(x)) * mu * mu + 3e-6)


def test_liquidation_probability_matches_brownian_paths():
    rng = np.random.default_rng(1)
    L, sig = 25.0, 0.02
    paths = np.cumsum(rng.normal(0, sig / np.sqrt(400), (20_000, 400)), axis=1)
    hit = (paths.min(axis=1) <= -body.liq_distance(L, True)).mean()
    p_long, _ = body.liq_probability(L, sig)
    assert p_long == pytest.approx(hit, abs=0.02)


# ------------------------------------------------------------------- mind
def test_arena_model_learns_a_planted_relation():
    c = cfg()
    w = World(c, RRBrain(c))
    b = w.brain
    rng = np.random.default_rng(3)
    b.memory[0] = 5000.0
    b.P[0] = np.eye(b.D) * b.P0
    truth = np.zeros(b.D)
    truth[2], truth[-1] = 0.004, -0.001
    for _ in range(4000):
        o = rng.normal(size=b.B)
        phi = b._phi(o[None])[0]
        y = np.array([phi @ truth + 0.01 * rng.normal(), -3.0 + 0.05 * rng.normal()])
        b._rls(np.array([0]), phi[None], y[None])
    assert b.w[0, 0, 2] == pytest.approx(0.004, abs=5e-4)
    assert b.w[0, 1, -1] == pytest.approx(-3.0, abs=0.05)
    shrunk = b._shrunk(np.array([0]))[0, 0]
    assert abs(shrunk[2]) > 0.003 and np.abs(np.delete(shrunk[:b.B], 2)).max() < 1e-3


def test_perspective_follows_participation():
    """Same market, same model: a flat agent and a leveraged long see different worlds."""
    c = cfg()
    w = World(c, RRBrain(c))
    for r in synthetic_rows(1500, seed=2):
        w.step(r)
    b = w.brain
    i, j = np.nonzero(w.alive & ~w.control)[0][:2]
    for arr in (b.att, b.kappa, b.hmult, w.tf):
        arr[j] = arr[i]
    for k in (i, j):
        b.w[k] = 0.0
        b.w[k, 0, 0] = 0.0005                     # aspect 0 moves the drift
        b.w[k, 1, 1] = 0.3                        # aspect 1 moves the volatility
        b.w[k, 1, -1] = np.log(0.01)
        b.vcal[k] = 1.0
        b.P[k] = np.eye(b.D) * 1e-9               # certain: no shrinkage
        b.recent[k] = np.random.default_rng(0).normal(size=(b.R, b.B))
        b.recent_n[k] = b.R
        w.acct.open(k, 1000.0)
    w.lev[i], w.frac[i] = 2.0, 0.5
    w.lev[j], w.frac[j] = 50.0, 0.2
    w.acct.execute(np.array([j]), np.array([8 * 1000.0 / w.mark]), w.mark, np.array([50.0]))
    b.perspective(np.array([i, j]), w.t)
    flat, levered = b.salience[i], b.salience[j]
    assert flat[0] > 0.9                          # opportunity is figure for the flat agent
    assert levered[1] > 0.3                       # risk becomes figure when 8x in at 50x
    assert levered[1] / levered[0] > 50 * flat[1] / flat[0]


def test_attention_drops_the_least_salient_aspect():
    c = cfg()
    w = World(c, RRBrain(c))
    for r in synthetic_rows(1500, seed=4):
        w.step(r)
    b = w.brain
    i = int(np.nonzero(w.alive & ~w.control)[0][0])
    b.explore[i] = 1.0
    b.age[i] = 10
    b.salience[i] = np.linspace(1, 2, b.B)
    b.salience[i, 3] = 0.0
    before = b.att[i].copy()
    b._attend(np.array([i]))
    changed = np.nonzero(b.att[i] != before)[0]
    assert list(changed) == [3]
    assert b.w[i, :, 3].tolist() == [0.0, 0.0] and b.age[i, 3] == 0


# ----------------------------------------------------------------- aspects
def test_aspects_never_look_ahead():
    rows = synthetic_rows(2500, seed=1)
    a, b = AspectEngine(), AspectEngine()
    altered = rows.copy()
    altered[2000:, 1:9] *= 1.5                    # change the future only
    for r in rows[:2000]:
        va = a.update(r)[0].copy()
    for r in altered[:2000]:
        vb = b.update(r)[0].copy()
    assert np.array_equal(np.isnan(va), np.isnan(vb))
    assert np.allclose(va[~np.isnan(va)], vb[~np.isnan(vb)])


def test_missing_streams_are_unavailable_not_invented():
    rows = synthetic_rows(1500, seed=1)
    rows[:, C["open_interest_value"]] = np.nan
    eng = AspectEngine()
    for r in rows:
        values, avail = eng.update(r)
    oi = [k for k, n in enumerate(NAMES) if n.startswith("open_interest.")]
    assert not avail[oi].any() and np.isnan(values[oi]).all()
    assert avail[NAMES.index("premium.dev.60")]


def test_deviation_aspect_matches_its_definition():
    eng = AspectEngine()
    rows = synthetic_rows(400, seed=9)
    m = v = 0.0
    for k, r in enumerate(rows):
        x = r[C["premium"]]
        if k == 0:
            m, v = x * 0.2, 0.0
            d = x - 0.0
            m, v = 0.0 + 0.2 * d, 0.8 * (0.0 + 0.2 * d * d)
        else:
            d = x - m
            m, v = m + 0.2 * d, 0.8 * (v + 0.2 * d * d)
        values, _ = eng.update(r)
    assert values[NAMES.index("premium.dev.5")] == pytest.approx(
        np.clip((x - m) / np.sqrt(v + 1e-18), -5, 5), rel=1e-9)


# --------------------------------------------------------------------- data
def _zip(lines, header=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        body_ = ([header] if header else []) + [",".join(map(str, r)) for r in lines]
        zf.writestr("x.csv", "\n".join(body_))
    return buf.getvalue()


def _ms(*args):
    return int(dt.datetime(*args, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _klines(start, n, price, step=1):
    return [[start + k * MINUTE, price, price + 5, price - 5, price + k * step, 2, start + k * MINUTE + MINUTE - 1,
             2 * price, 30, 1.5, 1.5 * price, 0] for k in range(n)]


def test_parse_klines_reads_flow_columns_and_microseconds():
    rows = [[1735689600000000, 1, 2, 0.5, 1.5, 9, 1735689659999999, 13.5, 42, 6, 9.0, 0]]
    assert data.parse_klines(_zip(rows)) == {1735689600000: (1.0, 2.0, 0.5, 1.5, 9.0, 13.5, 42.0, 6.0)}


def test_parse_metrics_with_header_and_datetime_stamps():
    header = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
              "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
              "count_long_short_ratio,sum_taker_long_short_vol_ratio")
    blob = _zip([["2024-01-01 00:05:00", "BTCUSDT", 80000, 3.4e9, 1.2, 1.5, 2.0, 0.9]], header)
    assert data.parse_metrics(blob) == [(_ms(2024, 1, 1, 0, 5), 80000.0, 3.4e9, 1.2, 1.5, 2.0, 0.9)]


def test_assemble_joins_streams_with_lag_and_funding():
    t0 = _ms(2024, 1, 1, 7, 50)
    minutes = [t0 + k * MINUTE for k in range(12)]
    kl = lambda p: (p, p + 1, p - 1, p, 2.0, 2 * p, 30.0, 1.5)
    perp = {t: kl(100.0) for t in minutes}
    mark = {minutes[9]: kl(99.0)}
    premium = {t: (0, 0, 0, 0.0002, 0, 0, 0, 0) for t in minutes}
    spot = {t: kl(99.5) for t in minutes}
    funding = {_ms(2024, 1, 1, 8): 0.00025}
    metrics = [(t0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    rows, carry = data.assemble(minutes, perp, mark, premium, spot, funding, metrics)
    oi = rows[:, C["open_interest"]]
    assert np.isnan(oi[:6]).all() and (oi[6:] == 1.0).all()      # visible 6 minutes later
    assert rows[9, C["funding_rate"]] == 0.00025                  # bar ending 08:00
    assert rows[[0, 8, 10], C["funding_rate"]].tolist() == [0.0, 0.0, 0.0]
    assert rows[9, C["mark_close"]] == 99.0 and rows[0, C["mark_close"]] == 100.0
    assert rows[0, C["spot_close"]] == 99.5 and rows[0, C["premium"]] == 0.0002
    assert (rows[:, C["perp"]] == 1).all() and carry == metrics[0]


def test_build_market_store_from_archives(tmp_path, monkeypatch):
    aug = _ms(2019, 8, 31, 7, 58)                  # spot era, crosses 08:00 settlement
    sep_spot = _ms(2019, 9, 1, 0, 0)
    listing = _ms(2019, 9, 8, 7, 58)               # perpetual starts mid-month
    kl_header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    files = {
        "spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-08.zip": _zip(_klines(aug, 3, 9000)),
        "spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip(_klines(sep_spot, 2, 10000) + _klines(listing, 3, 10000)),
        "futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip(_klines(listing, 3, 10100), header=kl_header),
        "futures/um/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip([[listing + MINUTE, 0, 10200, 10000, 10150, 0, 0, 0, 0, 0, 0, 0]]),
        "futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2019-09.zip":
            _zip([[_ms(2019, 9, 8, 8) + 3, 8, 0.00025]],
                 header="calc_time,funding_interval_hours,last_funding_rate"),
    }
    monkeypatch.setattr(data, "_get", lambda url, cache_dir=None: files.get(url.split("/data/", 1)[1]))
    store = str(tmp_path / "market")
    n = data.build_market(store, start="2019-08", log=lambda *a, **k: None)
    rows = np.array(list(data.iter_rows(store)))
    assert n == len(rows) == 3 + 2 + 3
    by_t = {int(r[0]): r for r in rows}
    assert by_t[aug + MINUTE][C["funding_rate"]] == pytest.approx(0.0001)   # assumed, spot era
    assert by_t[sep_spot][C["close"]] == 10000 and by_t[sep_spot][C["perp"]] == 0
    assert np.isnan(by_t[sep_spot][C["spot_close"]])          # no fake basis in stand-in rows
    assert by_t[listing][C["close"]] == 10100 and by_t[listing][C["perp"]] == 1
    assert by_t[listing][C["spot_close"]] == 10000            # spot next door, joined
    assert tuple(by_t[listing + MINUTE][C["mark_high"]:C["mark_close"] + 1]) == (10200, 10000, 10150)
    assert by_t[listing + MINUTE][C["funding_rate"]] == pytest.approx(0.00025)
    assert by_t[listing][C["volume"]] == 2 and by_t[listing][C["taker_buy_volume"]] == 1.5
    assert data.build_market(store, start="2019-08", log=lambda *a, **k: None) == 0
    months = [m for m, *_ in data.coverage(store)]
    assert months == ["2019-08", "2019-09"]


class FakeClient:
    def __init__(self, minutes, funding_after_calls):
        self.minutes, self.calls, self.after = minutes, 0, funding_after_calls

    def klines(self, start_ms, symbol="BTCUSDT", source="perp"):
        p = {"perp": 100.0, "mark": 99.0, "premium": 0.0, "spot": 99.5}[source]
        rows = {t: (p, p + 1, p - 1, p, 2.0, 200.0, 10.0, 1.0) for t in self.minutes if t >= start_ms}
        return rows, {t: t + MINUTE - 1 for t in rows}

    def funding(self, start_ms, end_ms, symbol="BTCUSDT"):
        self.calls += 1
        return {_ms(2024, 1, 1, 8): 0.0003} if self.calls > self.after else {}

    def metrics(self, start_ms, symbol="BTCUSDT"):
        return [(_ms(2024, 1, 1, 7, 50), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]


def test_live_rows_wait_for_the_funding_rate(monkeypatch):
    start = _ms(2024, 1, 1, 7, 58)
    client = FakeClient([start, start + MINUTE, start + 2 * MINUTE], funding_after_calls=2)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    feed = data.live_rows(start, client=client, log=lambda m: None)
    got = [next(feed) for _ in range(3)]
    assert [int(g[0]) for g in got] == [start, start + MINUTE, start + 2 * MINUTE]
    assert [g[C["funding_rate"]] for g in got] == [0.0, 0.0003, 0.0]
    assert got[0][C["mark_close"]] == 99.0 and got[0][C["spot_close"]] == 99.5
    assert got[0][C["open_interest"]] == 1.0
    assert client.calls >= 3


def test_store_roundtrip_by_month(tmp_path):
    rows = synthetic_rows(3 * 1440, seed=0)
    data.save_store(str(tmp_path), "SYNTH", rows)
    back = np.array(list(data.iter_rows(str(tmp_path), "SYNTH")))
    assert np.array_equal(back, rows, equal_nan=True)


# -------------------------------------------------------------------- books
def test_book_aggregates_ohlcv():
    book = TimeframeBook(300)
    closed = [book.add(k * MINUTE, MINUTE, o, o + 2, o - 1, o + 1, 1.0)
              for k, o in enumerate([10, 11, 12, 13, 14])]
    assert closed == [False, False, False, False, True]
    assert book.closes[-1] == 15 and book.highs[-1] == 16
    assert book.lows[-1] == 9 and book.vols[-1] == 5.0


def test_features_finite_and_bounded():
    book = TimeframeBook(60)
    rng = np.random.default_rng(0)
    p = 100.0
    for k in range(HISTORY + 1):
        p *= np.exp(rng.normal(0, 0.01))
        book.add(k * MINUTE, MINUTE, p, p * 1.01, p * 0.99, p, rng.random() + 0.1)
    f = book.features()
    assert f.shape == (N_FEATURES,) and np.all(np.isfinite(f)) and np.all(np.abs(f) <= 2)
