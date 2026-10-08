# The soup: life that is not designed

The planet (`docs/planet.md`) holds organisms we designed: their organs, values, dials and modes are ours. The soup goes further. Its organisms are bytes that run: nothing describes what they should do. What is set is matter, space, time, energy and a few physical laws, and the one rule of the economy: **energy enters the world only through trading BTC**. Whatever lives, dies, divides, trades well or badly does so because the laws allow it and because it pays.

Code:
- `evotrader/physics.py`: the laws, written once and compiled for the CPU and for CUDA.
- `evotrader/soup.py`: the world.
- `evotrader/weather.py`: what matter senses.
- `evotrader/gpu.py`: the world on a GPU.
- `evotrader/transplant.py`: reading organisms back as strategies.

Run: `python -m evotrader soup --minutes --from 2021-01-01 --until 2025-01-01 --run runs/soup` (add `--device gpu` on a CUDA machine; on cloud GPUs see `docs/gpu.md`). Without `--minutes` the world lives on 1-second data, a tick a second.

---

## 1. The laws (physics 3: all that is set)

| Law | What it says | Why this law |
|---|---|---|
| **Matter** | An organism is 64 bytes. They are rewritten by the organism itself, by its neighbors in meetings, by copying errors and by noise. When it dies, its matter returns to nothing. | Nothing about an organism is described anywhere else: its matter is all of it. |
| **Space** | A torus of sites, one organism per site, neighbors within 2 sites. Price, volume, trades and the clocks can be sensed everywhere. Each regional group of streams (order flow, premium, spot, open interest, positioning, funding, mark) exists only on smooth random continents covering about 55% of the surface. | Locality lets spatial structure, niches and refuges exist. Continents make where an organism lives matter. |
| **Time and thought** | Every tick (one row of data, at the finest resolution there is) each organism's own matter runs `--think` instructions (32), carrying on from where it stopped: the tape is a loop, and the instruction pointer and two heads persist. | Matter that thinks on its own. Every organism starts at the finest tick. A longer timescale (holding for hours, remembering a trend) is something its matter has to build. |
| **Chemistry** | Ten instructions move two heads, change bytes, copy between the heads and loop (BFF, Agüera y Arcas et al. 2024). Every other byte is inert. | Computation with nothing else assumed. |
| **Senses** | `S` replaces the byte under head 0 with the reading of the stream that byte names: how far the stream's level or change is above or below its own recent median (a percentile over the last 256 ticks; 0 is ordinary). A stream that does not exist where the organism lives reads 0. `E` reads its own energy, position or leverage. | The market enters only through these. |
| **Acts** | `A` sets the position it wants: a share of its equity, long or short, from the byte under head 0 (0: flat). The order fills at the next tick's open. `L` opts into leverage, 1× to 125× (1× unless it chooses more). | Waiting and acting are both choices. Leverage is a thing an organism opts into, not a given. |
| **Energy** | Each organism holds one exact BTCUSDT perpetual account: fees, spread, funding and the exchange's leverage brackets. Margin is isolated: a liquidation takes the margin of that position, never more, and there is no debt. | Survivors' wealth is market wealth. A loss is not the end. |
| **Cost of living** | Every tick an organism pays an upkeep for its body (`--upkeep`, USDT a day) and Kleiber's three quarters of what it holds (`--metabolism`: the days in which an organism holding one stake and earning nothing loses half of it). Every byte it writes costs heat (Landauer). | You can wait to preserve energy, but not forever. The poorer an organism, the slower it burns, so it can recover. Thinking costs. |
| **Hunger** | An organism eats when its trading since its last meal has paid for its living since: what it realized, after fees, funding and liquidations, exceeds what living cost it (the cost of living and heat). It never has to make more than one hunger's worth of living: a famine is no debt. One that has not eaten for `--starve` days is hungry. A hungry organism cannot divide, and any neighbor that has fed can put its child on its site, whatever the hungry one holds. A child is as hungry as its parent. | Successful trading is the only food. An organism can wait, and can lose and recover, but not forever: in time, those that trade successfully take the place of those that do not. Hunger moves space and energy to the fed; it destroys nothing. |
| **Death** | An organism whose equity falls to the floor (1 USDT) dies: its matter returns to nothing and its site to space. So does one whose site is taken by a neighbor's child. | Death is running out of energy, or losing one's place. |
| **Division** | A body holding at least `--divide-at`, and not hungry, tries to divide about once a cell cycle (`--cycle` days at one stake; more often the more it holds), into a random neighboring site. The site must be empty, or held by an organism that is hungry or holds less than `--takeover` (by default half) of what the parent holds. The child gets half of everything: its position in whole lots of the exchange (0.001 BTC) with their margin, and cash for the rest. It gets a copy of the parent's matter, with copying errors (`--mutation`), and goes on thinking where the parent was. An organism displaced this way dies, and the child takes over what it held. | Success becomes offspring. Space is limited, so a lineage spreads only into room or over organisms that have fallen behind. Nothing is lost when space changes hands. |
| **Meetings** | An organism wakes to meet a random living neighbor, `--meetings` times a day at one stake (Kleiber). Their tapes are joined and run once, at most `--meet` steps. Matter can be copied from one to the other; `A` and `L` act for the organism in whose half they sit; `T` can move a bite of energy between them (`--bite`, off by default). | Horizontal transfer of code: mating and parasitism. With bites off, energy passes between organisms only by takeover, so trading is the only way to grow. |
| **Noise** | Bytes flip at random, `--noise` per byte per day. | Variation. |

Defaults (`python -m evotrader soup --help`): hungry after 30 days without a meal, a cost of living that halves an idle stake in 730 days, no upkeep, division from 300 USDT (0.3 of a starting stake) about once a day, takeover of neighbors holding less than half, 90 meetings a day, no bites. They come from the calibration in §5.

**Conventions chosen so that the physics leans nowhere.** Each was found by running random matter on real data and asking why it leaned one way:
- Every byte the market reads or writes is signed, and 0 means nothing. The zero that ends every loop is stillness, not a position.
- Senses are percentiles of each stream's own recent history (histogram equalization, as a fly's photoreceptors encode contrast: Laughlin 1981). The median reads 0, and every value is equally likely.
- Opcodes sit in mirror pairs at ±(64−k), so code read as data is as often positive as negative.

The laws are scalar code acting on one site, or one pair of neighbors, at a time. A tick has three phases, live, birth and meet, and within a phase no two sites write the same memory. Randomness is a pure function of (seed, tick, stream, site). A world therefore does not depend on the order in which sites are visited, on how its ticks are chunked, or on the device: the CPU and the GPU compute the same world.

## 2. What is left to emerge

- **Strategies:** what to sense, when to act, how much to put at stake, at what leverage.
- **Timescales:** every organism starts at the finest tick. Holding for hours or weeks, or remembering a trend, takes structure its matter must build: loops, counters, bytes kept as memory.
- **Risk management:** isolated margin means the share at stake and the leverage together decide what a liquidation costs.
- **Lineages:** heredity is division. Lines of descent compete for space.
- **Horizontal transfer:** meetings copy code between neighbors (mating, parasitism).
- **Predation and cooperation:** with bites on (`--bite`), a loop of `T` drains a neighbor, and giving bites to kin can pay.

None of these is written. What pays is what persists.

## 3. Reading organisms back as strategies

An organism is code, so the organism is the strategy (`transplant.py`):
1. **Transplant.** Lift an organism (its matter, where its thought had got to, the leverage it had chosen) into a sandbox: the same physics and the senses of its birthplace, with evolution and the cost of living switched off.
2. **Test on unseen data.** It lives through data it has never seen. Its positions over time are its strategy. Its P&L there is the test, split into market, fees, funding and heat, and compared with holding BTC and with random matter.
3. **Read what drives it.** The senses its positions move with describe it, for example "follows the retail long/short account ratio".

This works however strange the matter becomes, because nothing about it has to be understood to be run.

## 4. What we measure

The census (`Soup.census`, drawn by `evotrader/viewer.py`):
- **Population:** organisms alive, births, deaths, lines of descent and their depth (generations).
- **Energy:** the living's equity against what was put in and against holding BTC; where the rest went (fees, funding, the cost of living, heat, what the dead left behind).
- **Timescales:** how long organisms hold a position (their age over the trades they have made), counted and weighted by energy. Graduation from the finest tick to longer ones shows here.
- **Leverage** chosen, positions held, maps of energy, life and position size.
- **Matter:** the most common programs, distinct tapes, high-order entropy.
- **Skill:** transplanted organisms against random matter on unseen data.

## 5. Calibrating the economy

Energy enters only by trading, so the economy decides what can live at all. Each problem below was found by running random matter through real minutes from 2021 on, on a 64 × 64 torus. Each fix is a law of the world, not a rule about organisms.

| The economy | What happened | What changed |
|---|---|---|
| Bodies divide in half at two stakes; a child eats a weaker neighbor it displaces, at 80% | Extinction. With a half-life of 180 or 365 days every line had died out by April 2023 or July 2024; at 730 days 51 organisms were left of 2,061 founders. There were 65 to 174 births in four years. | A child needs its parent to double, and real trading returns take years to double: selection never started. Bodies may divide at a size they already have (`--divide-at 300`). |
| Division at 300 USDT, every tick, eating at 80% | Thousands of births a day, but the churn of births and displacements burned 20% of the world's energy in three weeks. | Takeover is lossless: the child takes over what the displaced held. |
| Lossless takeover, division every tick | One lucky lineage of leveraged longs spread over the whole world in days (February 2021). The next dip wiped out the monoculture: 2.6 million USDT to 65,000 in a week. | A cell cycle (`--cycle`, a day at one stake): spreading takes time, so no bubble converts the whole world before it bursts. |
| Bites in meetings (`T`, up to 1 USDT each, 80% digestion) | Predators (loops of `T`) took over and burned 1.6 million USDT in a month, and the world fell to 340 organisms. Draining neighbors paid far better than trading. | Bites are off by default (`--bite 0`): energy passes between organisms only by takeover, so trading is the only way to grow. |
| No bites, division at 300 USDT once a cycle, takeover of neighbors holding less than half | A living world, full and stable, with 10,000 births in four months. But 80–89% of organisms never traded, and they held 87% of the energy. The largest lines were near-empty programs that write nothing, so they pay no heat. The world drained slowly through fees and the cost of living. | Random trading loses to fees, so not trading does as well as trading or better. Relative competition alone cannot make waiting fatal. |
| The same, but a child takes over a neighbor holding less than 75% or 90% of its parent | Selection got much stronger (85,000–124,000 births in four months; 2,000 lines down to 120–340), and stillness won faster: 96% never traded. Idle neighbors took over the sites of losing traders. | Hunger (`--starve`): survival requires trading that pays. |
| Hunger as death: no meal (a profitable close) for 7 days, and an organism dies | At day 8, 3,588 unfed organisms starved at once and their 1.65 million USDT vanished with them; the 558 left could not repopulate, and the world died out. Starving while holding 500 USDT is not physical either. | Hunger takes away an organism's place, not its life: a hungry organism cannot divide, and a fed neighbor's child can take its site and what it held. |
| Hunger as vulnerability, a meal being any close at a profit | Organisms that never traded fell from 88% to 18% within six weeks, and all survivors had fed. But churners stayed fed with tiny wins while fees ate the world: 930,000 USDT in six months, with trading before costs still negative. | A meal is trading since the last meal that made money after fees, funding and liquidations. |
| Meals net of fees | 98% looked fed without ever trading. Halving positions at each division had left dust shorts of a millionth of a bitcoin, and in 2021's positive funding any funding received counted as a meal. | Positions divide in whole lots (0.001 BTC). A meal must also pay for the living since the last one (the cost of living and heat). |
| Meals that pay for living since the last one (30 days) | Through the 2021 bull run, fed lines of traders spread over hungry ones. Organisms that never traded fell from 82% to about 25%, lines of descent from 1,900 to 12, and the deepest line reached 85 generations. After the May crash hardly anyone's trading paid, and the unpaid cost of living kept growing for as long as an organism went hungry. Births stopped in mid-2021; by the end of 2022 the median organism had not eaten for 577 days. | A famine is no debt: a meal never has to pay for more than one hunger's worth of living. |
| **The laws of §1 (defaults), 2021–2022** | The fed spread in the 2021 bull run, and organisms that never traded fell from 89% to 16%: 39,782 births, 16,486 of them over hungry neighbors, 76 generations, 41 lines. After the summer of 2021 almost nobody's trading paid even a month's living; with no one fed, no one could take over the hungry, and the world stood still (the median organism last ate 550 days before the end). 2,959 organisms were left with 210,000 of 2.06 million USDT: fees took 1.05 million, the cost of living 582,000, and trading before costs was −171,000. Without fees or spread (a control), selection pushed leverage up and the world lost 1.32 million before costs to 104,886 liquidations. | |

**What this says.** Every economy above ends the same way when energy can only come from trading: the world is a battery that runs down, because nothing selection found in 2021–2022 trades well enough to pay the market's costs. Transplanted onto January 2025 to September 2026 (BTC −10.7%), the ten richest organisms of the last world returned a median −9.0% (random tapes −0.9%), mostly by holding the leveraged positions they had. Hunger does what it is for: while anyone's trading pays, those whose trading pays take the place of those whose does not. It cannot make paying trades exist.

**How much four years can teach.** Selection can only tell skill from luck as well as the data can. At 1× a position in BTC swings about 70% a year, so a strategy with a Sharpe ratio of 0.5, good by any standard, is 1 standard error from luck after four years. Only strategies with Sharpe ratios near 2 stand out from luck within four years. What the data does hold plenty of are regimes, each rewarding a different behavior: two bull runs and a crash in 2021, the 2022 bear market, and the 2023–2024 recovery. Expect selection to find what those regimes rewarded first, and test everything on data it never saw (§3).

### Physics 1 and 2: choosing the sun

Before organisms, sites traded whatever their matter said, and the question was how the world's energy should come from BTC:

| Sun | What happened |
|---|---|
| Trading at up to 125×, every second, with rain refilling dead sites | Random matter traded itself to death: 92% of the planet's energy went in fees in one day. The rain paid random matter to gamble: 7 million USDT injected in one day. |
| Trading at up to 125×, no rain | Most sites were dead within a day; only matter that never trades kept its energy. |
| Trading at up to 1× (signals), minute data | Gentle: random matter loses about its fees, roughly 0.7% a week. Four years of it selected stillness, matter that does not trade (`docs/experiments.md`). |
| Fractional bites (a `T` could take everything) | 1.8 billion USDT changed hands in a week on a planet of 4.1 million. With digestion losses, that sloshing burned the planet in two weeks. Quantized bites fixed it. |

Why trading P&L is the right energy at all: a Kelly bettor's growth rate equals the mutual information between its signal and the outcome (Kelly 1956). Energy from trading is information about BTC turned into energy, the same equivalence that grounds Landauer heat.

## 6. Abiogenesis: life from random bytes

Matter alone, with no market, no organisms and nothing rewarding anything (`soup.Primordial`, `experiments/soup_emergence.py`, the exact BFF chemistry): every epoch, neighboring tapes are joined and run, and that is all. The run: 16,384 random tapes on a 2,048 × 8 lattice, seed 2.

For 48,000 epochs high-order entropy hovered near 0.6 and the most common tape had at most a handful of copies. Then, within 2,000 epochs (on one CPU core, after 39 minutes), self-replicators took over:

| Epoch | High-order entropy | Most common tape | It copies itself |
|---|---|---|---|
| 48,000 | 0.57 | ×1 | 17% |
| 49,000 | 0.86 | ×77 | 100% |
| 50,000 | 1.08 | ×31 (a family of variants) | 69% |

The family that took over is a palindrome (`·` is inert):

`·[<·····,······}···············]]···············}······,·····<[·`

What it does:
- Sitting in the first of two joined tapes, it loops: head 0 walks backwards from the end of the neighbor's tape, head 1 forwards through its own, and each byte is copied across. The neighbor ends up holding its exact reverse, and the original stays intact.
- Its instructions read the same in both directions (only its inert bytes do not), so the reversed copy is the same program. When the copy copies in turn, it reverses back.
- Sitting in the second tape, it stays intact.

When the run was saved, just after the transition, about 1% of all tapes were near copies (at least 90% of bytes equal to the most common one). Nobody wrote it.

The same search with the market instructions included found nothing in about 10⁹ interactions. S and E overwrite bytes with readings, which seems to inhibit the origin of life. Life can still reach the market: replicators born in matter alone can seed a market soup (`--matter`, `--layout bff`).

This run used soup physics 1's way of pairing (every tape starts one interaction an epoch, one after another). Physics 2 pairs sites by claims, all at once, and can run the same search at the scale of the original paper (2^17 tapes) on a GPU.

### Hosts and parasites

The replicator family was translated into the market soup's chemistry, instruction for instruction (`soup.translate`). It still writes its exact reverse into a neighbor there. Then 64 replicators were planted among 448 random tapes: a small market soup of 512 sites under physics 2 (§8: matter ran only when two neighbors met), with Landauer heat, on synthetic minutes. Counting tapes that really copy themselves:

| Hours | 0 | 8 | 24 | 40 | 48 | 64 | 80 | 96 | ... | 7 days |
|---|---|---|---|---|---|---|---|---|---|---|
| Replicators, no noise | 63 | 403 | 417 | 230 | 101 | 41 | 13 | 116 | | 512 (all) |

The replicators take over fast. Then a variant missing the copy loop's closing brackets spreads among them:

`·[<·····,······}·······························}······,······<[<`

Alone it copies nothing. When it starts a meeting with a replicator, it leaves the heads where the replicator's own copy loop then writes the variant's code over the replicator. It is a parasite that uses its host's copier, as in Tierra.

Hosts boom, parasites boom on the hosts, hosts crash, parasites starve with them, and hosts recover. These are host–parasite cycles that nobody designed.

With noise at the default rate, the first crash is final in a world this small: there is no refuge left to recover from. The same happened in a seeded world of 4,096 sites over real minutes (2021 on). Spatial models of hosts and parasites persist when space is large enough to hold refuges, which is what GPU-sized worlds are for (`docs/gpu.md`).

## 7. Open questions

- **Famine.** When nobody eats, nobody can take the place of the hungry, and the world stands still. Should a long hunger end in death (burning what the organism held) instead? Famines would then kill, as they do, at the cost of energy the world never gets back.
- **Skill or luck.** Four years of data hold only a handful of regimes (2021's two bull runs and its crash, the 2022 bear market, the 2023–2024 recovery). Selection can only reward what those regimes rewarded. Many seeds, bigger worlds and transplants onto unseen data are how to tell skill from a lucky fit.
- **Graduation.** Does matter that starts at the finest tick build longer timescales (memory, slower loops) when they pay?
- **Seconds.** 1-second data has 60 times more ticks than minutes, and 60 times more thought per day. That is what the GPU is for (`docs/gpu.md`).
- **Life meets the market.** Do replicators born in matter alone, seeded into a world of organisms, spread by meetings faster than selection by trading removes them?

## 8. Versions of the laws

- **Physics 1** (until October 2026): each second a fixed number of initiators were drawn in proportion to energy^0.75 and ran one after another. A site could meet several neighbors in a tick. Bytes flipped after the interactions, rain could refill dead sites, and the weather was computed by the planet in numpy.
- **Physics 2** (October 2026): sites, not organisms. Matter ran only when two neighbors met: each tick a site woke with a chance set by its energy (Kleiber), claimed a neighbor, and the pairs whose claims won ran their joined tape. Sites lay on latitudes of fixed timescales (1 minute to 2 weeks) that set when their senses updated and their orders filled. Exposure was capped by the world (`--max-exposure`), nothing died and nothing divided: a site without energy simply could not write. The same on CPU and GPU.
- **Physics 3** (`soup.PHYSICS`): the laws of §1. Organisms think on their own at the finest tick, pay to live, die, divide, take over weaker neighbors, choose their own leverage on isolated margin, and grow only by trading. A world records its physics in `meta.json`, and a run made under one version does not resume under another.

## Sources

- Agüera y Arcas, Alakuijala, Evans, Laurie, Mordvintsev, Niklasson, Randazzo & Versari (2024), *Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction*
- Ray (1991), *An approach to the synthesis of life* (Tierra)
- Plantec, Hamon, Etcheverry, Oudeyer, Moulin-Frier & Chan (2023), *Flow-Lenia: towards open-ended evolution in cellular automata through mass conservation and parameter localization*
- Linson, Clark, Ramamoorthy & Friston (2018), [The active inference approach to ecological perception](https://www.frontiersin.org/articles/10.3389/frobt.2018.00021/full)
- Bennett (2003), *Notes on Landauer's principle, reversible computation, and Maxwell's demon*
- Kleiber (1932), *Body size and metabolism*
- Lindeman (1942), *The trophic-dynamic aspect of ecology*
- Laughlin (1981), *A simple coding procedure enhances a neuron's information capacity*
- Kelly (1956), *A new interpretation of information rate*
