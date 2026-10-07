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

## Results: the planet (very many organisms on a world made of 1-second data)

Reproduce with:

```bash
python -m evotrader planet --synthetic 10 --run runs/planet-planted
python -m evotrader planet --synthetic 10 --null --run runs/planet-null
python -m evotrader planet --synthetic 10 --no-higher-order --run runs/planet-habits
python experiments/planet_check.py runs/planet-planted runs/planet-null runs/planet-habits
```

- **World.** A synthetic 1-second market (the planted crowding effect, or the null world) feeds a planet of 32 longitudes and 8 latitudes (1 second to 1 day). A place holds 12 organisms at normal liquidity, about 3,000 in all.
- **Population.** 1,000 founders. Whenever the population falls below 1,000, newcomers arrive, each with 1,000 USDT counted as money put in.
- **Duration.** 10 simulated days per world, about 1.5 hours of one CPU core each.
- **Comparisons.**
  - Planted against null: similar weather, but in the null world nothing can be predicted.
  - Planted against habits-only (the higher-order mode switched off): identical weather and identical founders. The difference is only consciousness.

### First attempt: consciousness that traded at random

In the first version, a conscious episode also picked an exploratory move at random. The three worlds were stopped after one to two days:

| World (days) | Planet net | Before fees and funding | Fees | Organisms that had learned anything |
|---|---|---|---|---|
| planted (1.8) | −49,649 | −105 | 49,687 | 137 of 1,000 |
| null (1.9) | −52,632 | −4,981 | 47,730 | 145 of 1,000 |
| habits only (0.8) | +2,914 | +10,556 | 7,711 | 321 of 1,012 |

- **Fast organisms paid about 5% of their capital a day in fees.** Organisms at the 1-second to 2-minute latitudes made 100 to 250 trades a day.
- **The random moves were pure cost.** Every organism already learns what all nine moves would have done, on every tick, so a random move teaches it nothing. Removing them cut fees at the fast latitudes about 30-fold in a 4-hour test.
- **Frequent consciousness also disrupted learning.** The gate's lesson (did grip improve more than usual after an episode?) was mostly noise, so its threshold random-walked. Fast organisms opened it nearly as often as allowed and kept rebuilding organs before they could learn through them. The habits-only world had three times as many organisms whose values carried evidence.
- **The fix, kept since.**
  - Both modes act on the organism's values.
  - Every episode raises the gate's threshold a little, and episodes that improve grip lower it. This makes consciousness costly, as Vervaeke says it is, and makes the gate an opponent process between that cost and what episodes pay.

### Ten days on the corrected planet

| World | Alive at the end | Born | Died | Generations | Largest family | Planet net | Before fees and funding | Fees | Liquidations | Holding BTC instead |
|---|---|---|---|---|---|---|---|---|---|---|
| planted | 1,120 | 460 | 340 | 10 | 63 | −101,933 (−10.2%) | −35,029 | 67,607 | 151 | +2.8% |
| null | 1,143 | 353 | 236 | 8 | 35 | −90,040 (−8.8%) | −25,986 | 64,017 | 92 | +4.6% |
| habits only (planted market) | 1,193 | 544 | 346 | 12 | 72 | −96,876 (−9.7%) | −27,776 | 69,706 | 150 | +2.8% |

**It behaves like a living world.**
- Every population grew past its 1,000 founders by births alone. The planted and habits-only worlds never needed a newcomer, and the null world needed 26.
- Families formed. The largest reached 63 members over 10 generations in the planted world, and 72 over 12 in the habits-only world.
- Latitudes filled and emptied unevenly.
  - In both worlds with consciousness, the 10-minute latitude emptied: 66 organisms (planted) and 27 (null), from about 125.
  - It did not empty in the habits-only world (154), where nobody migrates.
  - Fast latitudes (5 to 30 seconds) filled up.
- The gate learned to quiet consciousness where it did not pay.
  - At 30 seconds, the conscious share fell from 12% to 3%, and at 2 minutes from 22% to 10%.
  - The median threshold rose from 1.5 to 2.55.
  - The slow latitudes became more conscious over time (6 hours: 3% to 46%), but only because they need 16 harvests before the gate can open at all, and those took days.
- Culture was built and mostly fell. Over 21,000 observatories were built in each world with consciousness. The longest-standing lasted about 200 hours.
  - In the planted world, two of the six oldest read crowding streams: a premium level, `z256(smooth256(premium))`, and a positioning change.
  - In the null world, the oldest read price, the clocks and volume.
  - Six per world is an anecdote, not evidence.

**It does not make money, and it did not find the planted effect in ten days.**
- **Every world lost about 9 to 10% while holding BTC would have gained 3 to 5%.**
  - Fees were two thirds of the loss.
  - Before costs, the trading itself lost about 3%.
  - Liquidations (92 to 151) show leverage mattered.
- **The families that grew rich did it with leverage, not skill.** The richest families held 1.5 to 1.95 times their founder's stake, mostly at 6x to 18x leverage. Children of successful parents did worse than founders: a median of −7% to −9% since birth, against −1% to −2%. That is regression to the mean.
- **Relevance did not concentrate on the planted streams.** Salience on the crowding streams, relative to how much perception organisms gave them, was 0.75 to 1.6 in the planted world and 0.6 to 2.1 in the null world, with no consistent difference. Everywhere, organisms found their price streams most salient (1.2 to 2.2 times their share). That is natural: what a leveraged position is worth depends first on the price path.
- **Consciousness did not pay.** The habits-only world shares the planted world's weather and founders. It did slightly better on every measure: money, births, generations, and how many organisms' values carried any evidence (701 against 592).

**Why, as far as we can tell.**
1. **Ten days is short where the planted effect lives.** The crowding effect works over hours, at the 10-minute to 6-hour latitudes. There, ten days are only 40 to 1,440 harvests. The minute-world agents needed 120 days to concentrate salience, and only in some seeds.
2. **A conscious rebuild discards learning.** A rebuilt organ starts cold, and the weights learned through the organ it replaces are let go. The gate's cost now makes episodes rarer, but each one still sets the organism back.
3. **Over ten days, leverage decides who lives.** Selection rewarded lucky leverage, as it will on any short horizon. Selecting for skill needs time for luck to average out.

**What would change the picture:** months of history instead of days, real data, and a higher-order mode that keeps what it has learned when it reorganizes. These are the next experiments. A year of 1-second data at this population is about two and a half days of one CPU core.
