import datetime as dt
import io
import pickle
import zipfile

import numpy as np
import pytest

from evotrader import data
from evotrader.config import Config
from evotrader.features import HISTORY, N_FEATURES, TimeframeBook
from evotrader.world import World

MIN = 60_000


def flat_bars(n, price=100.0, start=0):
    return [(start + k * MIN, price, price, price, price, 1.0) for k in range(n)]


def small_cfg(**kw):
    cfg = Config(capacity=20, n_control=4, min_per_niche=1, timeframes=(60, 300), seed=1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_book_aggregates_ohlcv():
    book = TimeframeBook(300)
    closed = [book.add(k * MIN, MIN, o, o + 2, o - 1, o + 1, 1.0)
              for k, o in enumerate([10, 11, 12, 13, 14])]
    assert closed == [False, False, False, False, True]
    assert book.closes[-1] == 15 and book.highs[-1] == 16
    assert book.lows[-1] == 9 and book.vols[-1] == 5.0


def test_book_closes_candle_cut_short_by_gap():
    book = TimeframeBook(300)
    book.add(0, MIN, 1, 1, 1, 1, 1)
    assert book.add(10 * MIN, MIN, 2, 2, 2, 2, 1)        # jumped two candles ahead
    assert list(book.closes) == [1]


def test_features_finite_and_bounded():
    book = TimeframeBook(60)
    rng = np.random.default_rng(0)
    p = 100.0
    for k in range(HISTORY + 1):
        p *= np.exp(rng.normal(0, 0.01))
        book.add(k * MIN, MIN, p, p * 1.01, p * 0.99, p, rng.random() + 0.1)
    assert book.ready
    f = book.features()
    assert f.shape == (N_FEATURES,) and np.all(np.isfinite(f)) and np.all(np.abs(f) <= 2)


def test_buying_costs_exactly_fees_and_slippage():
    w = World(small_cfg(metabolism_bps_per_day=0.0), 60)
    w.step(*flat_bars(1)[0])
    i = int(np.nonzero(w.alive & ~w.control)[0][0])
    before = w.equity(i)
    w.params[i] = 0.0
    w.params[i, -1] = 50.0                              # always wants to be fully long
    w.deadband[i] = 0.01
    w._act(int(w.tf[i]), np.zeros(N_FEATURES), 100.0)
    fee, slip = 10e-4, 2e-4
    notional = before / (1 + fee)
    expected = before - notional * fee - notional * (1 - 1 / (1 + slip))
    assert w.equity(i) == pytest.approx(expected)
    assert w.btc[i] * 100 / w.equity(i) > 0.99


def test_split_conserves_equity_and_resets_reference():
    w = World(small_cfg(), 60)
    w.step(*flat_bars(1)[0])
    i = int(np.nonzero(w.alive & ~w.control)[0][0])
    w.cash[i] *= 2                                      # pretend it doubled its money
    total = w.equity(i)
    n = int((w.alive & ~w.control).sum())
    w._reproduce(i)
    child = int(np.nonzero(w.agent_id == w.next_id - 1)[0][0])
    assert w.equity(i) + w.equity(child) == pytest.approx(total)
    assert w.ref[i] == w.ref[child] == pytest.approx(total / 2)
    assert w.growth[i] == pytest.approx(2.0)
    assert w.parent_id[child] == w.agent_id[i] and int((w.alive & ~w.control).sum()) == n + 1


def test_no_frictions_no_money_created_or_destroyed():
    cfg = small_cfg(fee_bps=0.0, slippage_bps=0.0, metabolism_bps_per_day=0.0)
    w = World(cfg, 60)
    for b in flat_bars(3 * 24 * 60):
        w.step(*b)
    assert w.snapshot()["net_pnl"] == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_resumes_deterministically():
    bars = list(data.synthetic_bars(4000, seed=5))
    a = World(small_cfg(), 60)
    for b in bars[:2000]:
        a.step(*b)
    b_world = pickle.loads(pickle.dumps(a))
    for b in bars[2000:]:
        a.step(*b)
        b_world.step(*b)
    assert np.array_equal(a.equity(), b_world.equity())
    assert a.next_id == b_world.next_id


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", "\n".join(",".join(map(str, r)) for r in rows))
    return buf.getvalue()


def test_archive_parse_handles_microsecond_stamps():
    rows = [[1735689600000000, 1, 2, 0.5, 1.5, 9, 1735689659999999, 0, 0, 0, 0, 0]]
    assert data.parse_archive(_zip(rows)) == [(1735689600000, 1.0, 2.0, 0.5, 1.5, 9.0)]


def test_download_from_archive_and_resume(tmp_path, monkeypatch):
    day0 = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=3)
    files = {}
    for d in range(3):
        day = day0 + dt.timedelta(days=d)
        t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()) * 1000
        rows = [[t0 + k * MIN, 1, 1, 1, 1, 1, t0 + k * MIN + MIN - 1] for k in range(1440)]
        files[f"BTCUSDT-1m-{day:%Y-%m-%d}.zip"] = _zip(rows)
    monkeypatch.setattr(data, "_get", lambda url, timeout=60: files.get(url.rsplit("/", 1)[1]))
    out = str(tmp_path / "btc.csv")
    n = data.download(out, days=3, source="archive", log=lambda *a, **k: None)
    assert 2 * 1440 <= n <= 3 * 1440
    bars = list(data.iter_csv(out))
    assert all(b2[0] - b1[0] == MIN for b1, b2 in zip(bars, bars[1:]))
    assert data.download(out, days=3, source="archive", log=lambda *a, **k: None) == 0
