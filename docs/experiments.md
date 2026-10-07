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

Same markets (seeds 11–13), salience mode with anticipation switched on: new
aspects are chosen by learned query–key attention instead of uniformly.

| Planted world | Returns by seed | Mean | Profitable seeds | Crowd salience |
|---|---|---|---|---|
| salience, no anticipation | +21.6, −28.1, −17.5 | −8.0% | 1/3 | 0.47 |
| random attention | +17.1, −19.8, −28.7 | −10.4% | 1/3 | 0.51 |
| salience + anticipation | +25.7, −24.3, +29.1 | **+10.2%** | **2/3** | 0.46 |

**In the null world anticipation did not invent relevance.** Crowd salience was
0.26, below the 0.32 base rate. Returns were −30.2%, in line with the other
modes.

**Nine paired seeds (11–19), salience mode with and without anticipation:**

| Seed | With anticipation | Without | Difference |
|---|---|---|---|
| 11 | +25.7% | +21.6% | +4.1 |
| 12 | −24.3% | −28.1% | +3.8 |
| 13 | +29.1% | −17.5% | +46.6 |
| 14 | −21.6% | −21.3% | −0.3 |
| 15 | +1.0% | −5.0% | +6.1 |
| 16 | −20.1% | −20.1% | 0.0 |
| 17 | −25.6% | −24.0% | −1.6 |
| 18 | −12.0% | −17.7% | +5.7 |
| 19 | −0.4% | −5.7% | +5.3 |
| **Mean** | **−5.4%** | **−13.1%** | **+7.7** (sd 14.9) |

- **Results:** anticipation did better in 7 of 9 seeds, and was profitable in 3 of 9 against 1 of 9 without it.
- **The mean is carried by one seed (13).** The median gain is about +4 points.
- **Significance:** a one-sided sign test gives p ≈ 0.09, and a t-test on the differences gives t ≈ 1.6.
- **Verdict:** it probably helps a little, but this is not established. Run-to-run noise is large: never-selected controls swing about ±15 points between runs of the same market, from random draws alone.

**Grip, split.** In both worlds the median agent's grip on *direction* is about
zero (−0.005). Its grip on *volatility* is positive (+0.07 planted, +0.17 null).
The directional edge in the planted world is too faint to show in the typical
agent's log-score. The profits come from a minority of agents and from sizing up
when the signal is strong. This matters for the next step: loss of grip on
direction is not yet a usable trigger for "my scripts stopped working". A
changepoint detector on the agent's own outcomes is the better-founded signal.

## Results: self-made agents (no engineered features, no theory of value)

These agents (`selfmade.py`) are given only:
- raw receptors;
- their account's physics;
- generic capacities.

To find the planted effect they must *construct* a crowding detector, for example a long-normalized level of the premium or of the positioning ratios, or the perp–spot basis, out of raw receptors. Base rate: 23% of random programs read a crowding receptor or the basis.

| World / seed | Return | Controls | Crowd salience, first → last 30 days | Most salient self-built programs |
|---|---|---|---|---|
| planted 11 | −11.3% | +13.7% | 0.23 → 0.31 | `z1024(smooth16(close - spot))` (the basis), `z1024(smooth64(top_position_ls))`, also `z4096(week_cos)` |
| planted 12 | −24.1% | −43.3% | 0.20 → 0.34 (0.42 on the last day) | `z16384(smooth4(premium))` with 30% of the population's salience |
| planted 13 | −16.9% | −37.7% | 0.20 → 0.23 | nothing relevant stood out |
| null 11 | −35.0% | −36.1% | 0.22 → 0.17 | nothing; salience on crowding programs fell below the base rate |

**What this shows:**

1. **Genuine discovery happens, sometimes.**
   - In two of three planted seeds, the population built the relevant perceptions itself and concentrated salience on them. In seed 12 it built an ~11-day premium-level detector from the raw premium.
   - In the null world it built nothing of the kind.
   - None of these perceptions was given. They were composed, kept and inherited because they mattered to the agents' own values.
2. **Being told is worth something.**
   - Over the same 120 days, the engineered agents (our features and our Kelly theory) lost less: −5% to −8% on average.
   - The self-made agents lost −17%, against −22% for their controls, and did not convert discovery into profit within 120 days.
   - Constructing perception from scratch costs time, and the planted effect is faint.
3. **Directional grip of the self-made agents stays slightly negative** (about −0.01 to −0.08). Their learned values do not yet predict outcomes better than "nothing happens". That is expected at this signal-to-noise ratio, and it is what more history (five years of full data) and evolution of the physiology should be tested against.

**Next:**
- Longer runs: real history once it is reachable.
- More seeds.
- Letting evolution shape the perceptual physiology (operators and time constants).
