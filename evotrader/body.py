"""The body model: what the market offers *this* account over the next horizon.

For a target exposure x (signed multiple of equity: 5 = 5x long) the expected
log-growth of equity over a horizon with log-return drift mu, volatility sig
and expected funding per unit of exposure fund is approximately

    E[Pi](x) = x (mu + sig^2/2 - fund) - kappa/2 x^2 sig^2 - cost |x - lam| - p_liq(x) loss(x)

(mu + sig^2/2 is the arithmetic drift of a log-normal price, so a plain 1x long
with kappa = 1 earns exactly mu.)

where lam is the current exposure (so only the change pays fees and spread),
kappa >= 1 is the agent's risk weight (1 = full Kelly), p_liq is the chance of
touching the isolated-margin liquidation price within the horizon (reflection
principle: 2 Phi(-d / sig)) and loss = -log(1 - |x| / L) is the log of losing
that margin. The first two terms are Kelly's; the rest is the exchange.

The affordance value is V = max_x E[Pi](x). By the envelope theorem its
sensitivities are dV/dmu = x* and dV/dlog(sig) = x* sig^2 (1 - kappa x*) -
loss(x*) dp_liq/dlog(sig): this is how participation shapes what is salient.
"""

import numpy as np

MMR = 0.004        # maintenance margin rate of Binance's first BTCUSDT bracket


def _erf(z):
    # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    s = np.sign(z)
    z = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-z * z)
    return s * y


def norm_cdf(z):
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def liq_distance(L, long):
    """Log price distance from entry to the liquidation price of a fresh isolated position."""
    m = 1.0 / L - MMR
    return np.where(long, -np.log1p(-m), np.log1p(m))


def liq_probability(L, sig):
    """(long, short) chance of touching the liquidation price within the horizon."""
    return (np.minimum(1.0, 2.0 * norm_cdf(-liq_distance(L, True) / sig)),
            np.minimum(1.0, 2.0 * norm_cdf(-liq_distance(L, False) / sig)))


def expected_growth(x, mu, sig, fund, lam, L, cost, kappa, p_long=None, p_short=None):
    """E[Pi] for exposures x; every argument broadcasts against x."""
    if p_long is None:
        p_long, p_short = liq_probability(L, sig)
    p = np.where(x > 0, p_long, np.where(x < 0, p_short, 0.0))
    loss = -np.log(np.maximum(1.0 - np.abs(x) / L, 1e-3))
    return (x * (mu + 0.5 * sig * sig - fund) - 0.5 * kappa * x * x * sig * sig
            - cost * np.abs(x - lam) - p * loss)


def affordance(mu, sig, fund, lam, L, xmax, cost, kappa, grid=41):
    """(V, x*) over the leading dimensions of the arguments (all same shape)."""
    mu, sig, fund, lam, L, xmax, cost, kappa = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (mu, sig, fund, lam, L, xmax, cost, kappa)))
    u = np.linspace(-1.0, 1.0, grid)
    u = np.sign(u) * u * u                     # denser near flat, where most optima are
    xs = np.concatenate([xmax[..., None] * u, np.clip(lam, -xmax, xmax)[..., None],
                         np.zeros(xmax.shape + (1,))], axis=-1)
    p_long, p_short = liq_probability(L, sig)
    e = expected_growth(xs, mu[..., None], sig[..., None], fund[..., None], lam[..., None],
                        L[..., None], cost[..., None], kappa[..., None],
                        p_long[..., None], p_short[..., None])
    k = np.argmax(e, axis=-1)[..., None]
    return np.take_along_axis(e, k, -1)[..., 0], np.take_along_axis(xs, k, -1)[..., 0]


def sensitivities(x, sig, L, kappa):
    """(dV/dmu, dV/dlog sig) at the optimum x (envelope theorem)."""
    d = liq_distance(L, x > 0)
    z = d / sig
    p = 2.0 * norm_cdf(-z)
    dp = np.where((x != 0) & (p < 1.0), 2.0 * norm_pdf(z) * z, 0.0)
    loss = -np.log(np.maximum(1.0 - np.abs(x) / L, 1e-3))
    return x, x * sig * sig * (1.0 - kappa * x) - loss * dp
