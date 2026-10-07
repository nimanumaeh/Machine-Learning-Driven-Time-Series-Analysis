# Experiments: does the relevance-realization loop find what is relevant?

Reproduce with `experiments/relevance_check.py`.

## Setup

- **Market.** A synthetic BTCUSDT perpetual (`evotrader/synthetic.py`) with all 24 data columns. A hidden "crowding" variable leaks noisily into five of the fifteen streams: premium, funding, basis, top-trader account and top-trader position ratios.
  - **Planted world:** crowding pushes the price against the crowd. Over 4 hours the correlation is about −0.18 with the best aspects, measured on independent windows. Real but faint.
  - **Null world:** the same streams move together but predict nothing.
- **Population.** 40–60 relevance-realizing agents plus 8 never-selected controls, which use random attention and never reproduce. Cadences 1m, 5m, 15m and 1h. Real exchange costs.
- **Duration.** 120 days, 172,800 minutes per run.
- **Base rate.** The crowding streams make up 32% of all 186 aspects, so a population attending at random puts about 0.32 of its salience there.

## Results: salience-driven versus random attention

Seeds 11–13 for the planted world; seed 11 for the null world.

| Condition | Agents' return (mean) | Controls (mean) | Edge over controls | Crowd salience, last 30 days | Crowd attention |
|---|---|---|---|---|---|
| planted, salience | −8.0% (+21.6, −28.1, −17.5) | −26.5% | +18.5 pts | 0.47 | 0.38 |
| planted, random | −10.4% (+17.1, −19.8, −28.7) | −30.3% | +19.9 pts | 0.51 | 0.35 |
| null, salience | −27.8% | −34.6% | +6.8 pts | 0.35 | 0.33 |
| null, random | −31.2% | −41.1% | +9.9 pts | 0.41 | 0.32 |

**What this shows:**

1. **Relevance is detected where it exists, and only there.** In the planted world, the population's salience moves onto the crowding streams (0.47–0.51, from about 0.3 at the start). In the null world it stays near the base rate. The top aspects in the planted runs are exactly the crowding proxies: top-trader account and position trends, open interest, basis, premium. In the null runs nothing stands out.
2. **The signal is worth about 10 points, and it is not reliably enough to profit.**
   - Selection beats never-selected controls in every world, largely by weeding out reckless traders.
   - The *extra* edge in the planted world over the null world (about +10 to +12 points) is what the signal is worth.
   - In absolute terms, only one of three planted seeds made money in 120 days. An effect of this size sits at the edge of what survives fees, which is a useful calibration for real data.
3. **Dropping the least salient aspect is no better than dropping one at random.** Across three seeds the two attention modes are indistinguishable. The useful work is done by the evidence-weighted arena model, which attaches salience to the right aspects among those it happens to attend, and by selection. The weak link is choosing *which new aspect to try*, which was uniform. That motivated anticipation (below).
4. **Grip, as first defined, misled.** It was higher in the null worlds, because predicting volatility (easy: it clusters) was mixed with predicting direction (the edge). It is now split into grip on direction and grip on volatility.

## Results: anticipation (query-key attention over the grammar)

Pending: the runs are in progress. Same seeds, salience mode with anticipation
switched on, plus a null-world run to check that it does not invent relevance.
