# The planet: a world made of the BTC market, and the smallest minds that can live on it

We build a planet whose climate is the Bitcoin market, second by second. We
seed it everywhere with very many of the smallest organisms that Vervaeke's
account of mind allows, and let it run: populations spread, specialize,
build, collapse and recover. We do not decide what matters to them. We build
the infrastructure that lets them find out, and that lets their way of
finding out change.

The document is in three parts:
- **Sections 1–2:** a close reading of Vervaeke, separating what we must engineer, what must be left to be learned, and what must stay changeable.
- **Sections 3–5:** the smallest organism, the planet, and how organisms shape the planet.
- **Sections 6–9:** money, data and compute, what we will watch for, and honest limits.

---

## 1. Reading Vervaeke for an engineer

Vervaeke's account is not a list of modules. It is a claim about what kind
of *process* a mind is. The parts that matter here:

1. **Relevance realization is grounded in self-making.**
   - Only a system that takes care of itself can care about anything. Vervaeke: genuine intelligence needs "an autopoietic system, a system that cares about information because it's taking care of itself".
   - With Jaeger, Riedl, Djedovic and Walsh (2024) he locates this in *organizational closure*: the organism's components produce and constrain one another so that the whole keeps itself going.
   - *Agency* is then setting and pursuing intrinsic goals through this self-manufacture. By contrast, "algorithms and machines only possess extrinsic purpose imposed from outside".
2. **It is a bioeconomic process, not a representation.** Relevance is realized by opponent processes that keep the system balanced between efficiency and resiliency, along three axes:
   - scope: general-purpose versus special-purpose;
   - tempering: exploiting versus exploring;
   - prioritization: focusing versus diversifying.
   None of these settings is ever "correct". The balance is continuously renegotiated with the world.
3. **Adaptivity.** In Di Paolo's sense: the organism does not merely survive. It senses and regulates its own relation to its viability boundary, so it acts *before* it dies.
4. **The agent–arena relationship is co-constituted.** The organism's capacities and the world's affordances specify each other. Participatory knowing is that fittedness, and it is the deepest of the four kinds of knowing. Perspectival (salience), procedural (skill) and propositional (claims) knowing are grounded in it, in that order, though they all run at once.
5. **Consciousness is recursive relevance realization.**
   - It is relevance realization applied to relevance realization.
   - It is engaged when situations are novel, complex or ill-defined.
   - It reorganizes how the system is realizing relevance: reframing, restructuring, insight.
   - Most relevance realization is unconscious, in habits and skills. Consciousness is costly, and it is not always on.
6. **4E cognition.**
   - Embodied: the body matters.
   - Embedded: cognition is situated in an environment.
   - Enacted: perceiving is something done, through action.
   - Extended: the world is used as part of the mind, through tools, marks and culture.
7. **The organism shapes its world, and the world shapes it.** This is structural coupling. In evolutionary terms it is niche construction: organisms modify their own and each other's selective environments. The modified environment is *inherited* by those who come after (ecological inheritance).
8. **Nothing is fixed.** This is the point behind "weights are not enough". A trained network with frozen weights has no ongoing self-making: its structure was imposed and stays imposed. A Vervaekean system keeps producing, maintaining and dissolving its own structures through its own activity.

## 2. What we engineer, what we leave open, what stays changeable

| Vervaeke | We engineer (infrastructure and biases) | Left open (learned, never specified) | Stays changeable |
|---|---|---|---|
| Self-making, precariousness | A viability variable (the organism's wealth) and a boundary (death); perceptual organs that decay unless they keep earning their upkeep | How it keeps itself going; which organs it builds | Every organ is produced, maintained and dissolved by the organism's own activity |
| Adaptivity | Interoception (its own wealth, exposure, recent outcomes) available to its learning | What to do when it nears its boundary | — |
| Relevance realization | Scarcity (a few organs at a time); salience as "how much does what the world offers *me* change when this feature moves", measured on its own learned values | Which features matter; what they mean | Salience is recomputed continually; organs are replaced by what turns out to matter |
| Opponent processing | Two dials, explore–exploit and efficiency–resiliency, each a balance of two opposed rates $d^* = P_+/(P_+ + P_-)$ | Where the balance sits in each situation | The dials move with the organism's own signals; their gains are inherited and mutate |
| Participatory knowing | A body coupled to a place: an account, a cell on the planet, a timescale (its latitude) | Its niche, its identity, where it settles | It can migrate, and so become a different kind of organism |
| Perspectival knowing | The salience law (above) | Its salience landscape | Continually |
| Procedural knowing | A habitual mode: act on learned values with little effort | Its habits | Habits keep learning; they are overridden when consciousness engages |
| Propositional knowing | Nothing directly | Emerges, if at all, as *public structures*: observatories that make one organism's perception available to every organism in a place | Structures decay unless they keep being used |
| Consciousness | A second, costly mode that can reorganize the first: rebuild organs deliberately, break frames, migrate, build. Plus signals (surprise, conflict, novelty, stagnation) that *could* trigger it | When to engage, and what to reorganize. The gate is learned from whether engaging paid off | The gate keeps learning; its starting point evolves. An organism may rarely or never engage |
| Insight | The capacity for bursts of structural turnover (frame-breaking) | When to break a frame and what new frame to make | — |
| 4E | Local perception only; moving changes what can be perceived; the ability to build and to mark | What to build, where, and for whom | Structures persist, decay and are inherited ecologically |
| Niche construction | Means: observatories (shared perception) and markers (stigmergy); a carrying capacity per place | Settlements, migrations, collective patterns, any "civilization" | Everything built can fall |
| Evolution | Inheritance with variation (fission with mutation) | Lineages | Genes for horizon, learning, dials and the gate all mutate |

**Deliberately absent.**
- Any market concept: trend, risk, crowding, momentum.
- Any reward beyond continuing to exist (wealth is life).
- Any propositional module or language.

**Still imposed, and why.**
- **Physiology** (capacities, not knowledge):
  - the menu of generic operators (difference, smoothing, swing, normalization) and time constants;
  - the learner (recursive least squares);
  - the motor repertoire (nine crop intensities, and moving to a neighboring cell).
- **The planet's design** (geography, latitudes, seasons): this is world-building.

None of these encodes what matters. Evolution can later be given the physiology too.

## 3. The smallest Vervaekean organism

Each organism is a few hundred numbers, so very many can live at once.

1. **A body.** Wealth, held in one exact BTCUSDT perpetual account. A cell on the planet. A timescale, set by its latitude. Death below half its reference wealth; division above 1.25 times.
2. **Organs of perception: K = 6 self-made programs.** Each is $z_\tau(\mathrm{op}_k(a - b))$ over whatever its place lets it sense. Each organ has an *integrity* that decays every tick and is restored only by being salient. Organs that stop mattering dissolve and are replaced. The organism keeps remaking its own perception.
3. **A participatory learner.** What each of its 9 possible crop intensities would have done to *its own* wealth in each perceived situation. It learns from the actual path of the world run through its own account: full counterfactual experience, no theory.
4. **A salience landscape.** The sensitivity of its own values to each of its own organs, given its current exposure and costs.
5. **Two opponent dials.**
   - **Explore–exploit:** pushed toward exploring by surprise and stagnation, toward exploiting by grip and learning progress.
   - **Efficiency–resiliency:** pushed toward pruning by stability, toward keeping variety by turbulence.
6. **Two modes.**
   - **Habitual (unconscious):** every tick it acts on its learned values. Every few ticks it re-landscapes salience and maintains its organs.
   - **Higher-order (conscious):** opened by a learned gate. It rebuilds organs deliberately (mutations of what is salient, guided by query–key anticipation), breaks frames when stuck, migrates, builds and marks. It costs disruption: new organs start cold, and moving resets perception.
   - **Gate learning:** the gate compares how grip changed after episodes with how it drifts without them, and adjusts itself. This is relevance realization about when to realize relevance at a higher order.
7. **Inheritance.** Children start with the parent's organs, values, dials and gate, then mutate.

## 4. The planet

**Latitude is timescale.**
- The equator ticks every second. Bands toward the poles tick every 5s, 30s, 2m, 10m, 1h, 6h and 1d.
- An organism's latitude sets how fast its world moves: its tick, and its crops' growing period (its horizon).
- The equator is turbulent and expensive: fees dominate there. The poles are slow and seasonal.
- Climate zones are the market at different resolutions.

**Longitude is which senses exist there.**
- Each data stream has continents, a smooth random geography fixed at the planet's birth, where it can be perceived. The streams are order flow, premium, the spot market next door, open interest, positioning, and funding.
- Price, volume and the clocks can be perceived everywhere.
- Information is distributed over the surface, as resources are on Earth. Some places are rich in senses and some are poor. Migrating changes what an organism can know.

**Weather is the data.** Each band aggregates the 1-second rows into its own bars, and the receptors read those bars generically. Expressed in planetary terms:
- temperature: price change;
- storms: price range;
- water flow: volume;
- wind: the taker flow;
- pressure: premium and basis;
- tides: the 8-hour funding;
- seasons: the day and week;
- earthquakes: liquidation cascades.

The organisms are never told these names. They see receptor numbers and build their own organs.

**Water is liquidity.** Each place supports a number of organisms that rises and falls with the band's trading volume. Rich markets carry more life.

**Crops are the money.**
- Planting with intensity $x$ in a band means holding exposure $x$ for the band's growing period, in the organism's own account.
- The harvest is exactly what the account does: price change, fees, funding, and liquidation as crop failure.
- The planet's economy *is* trading. Nothing else feeds anyone.

## 5. Organisms shaping the planet

We cannot change BTC: it is the sun and the weather. Organisms shape the planet the way life does, locally:

- **Observatories (extended perception, shared).**
  - A conscious organism can externalize one of its salient organs into its place.
  - Everyone living there can then perceive that organ's output as a new sense.
  - An observatory may also be a variant that looks at a stream the place itself cannot sense, like a telescope.
  - Observatories decay, and are repaired by the organisms that find them salient. Useful ones persist across generations. This is ecological inheritance: the closest thing to propositional knowledge, being public, persistent and shareable.
- **Markers (stigmergy).** Organisms leave traces of how life has gone in a place: recent harvests and crowding. Others can perceive these traces and use them to decide where to go.
- **Space.** Places fill up. Newborns spread to neighboring cells, migrations relieve crowding, and empty places get re-seeded.

Settlements, migrations, specialized peoples by latitude and continent, and the slow accumulation of maintained observatories (a civilization's infrastructure) are what we hope to see. None of it is scripted.

## 6. Money

- **Every gain and loss on the planet is an exact BTCUSDT perpetual account outcome.** Survivors, at their latitude, are organisms that read the market at that timescale well enough to beat fees.
- **What we read back:**
  - Their organs and salience: readable perceptions with track records.
  - The observatories that civilizations keep repairing: market perceptions that a whole population found worth maintaining.
  - Population dynamics by latitude: which timescales are habitable at all. That is a direct, if sobering, map of where an edge exists after costs.

## 7. Data and compute

- **1-second rows (`data_seconds.py`).**
  - The fast columns are rebuilt from every perpetual trade (aggTrades), so nothing is lost.
  - Slow columns come from the minute store (mark, funding, premium, spot, open interest, positioning). Each is forward-filled from when its minute closed, so the rows stay causal.
  - Storage is one file per day.
  - At 1 second, liquidation uses the trade high and low, because no 1-second mark price exists. This is stricter than the exchange.
- **Compute.** Organisms are columns in arrays, and each band only works on its own tick.
  - 1-second organisms are the expensive ones. They are also the ones fees starve.
  - Thousands of organisms run on a laptop CPU. The same array code ports to a GPU (JAX) for hundreds of thousands.

## 8. What we will watch for, and how we will know

- Populations by latitude and continent over time: settlement, collapse, recolonization.
- Which organs persist, where, and in which lineages. Which observatories survive.
- How often organisms are conscious, and whether that rises in regime breaks and falls in mastery.
- **Ablations:**
  - higher-order mode off;
  - observatories and markers off;
  - shuffled-weather planets, where nothing should settle except by luck.

## 9. Honest limits

- **Not real autopoiesis.** By the account of Jaeger, Vervaeke and colleagues, a computer simulation does not instantiate genuine autopoiesis or relevance realization. It models their organization.
- **The world is formal.** It is large but formal: a "big small world".
- **Faint signals.** At most timescales the market's edges after costs are faint, so most of the planet may be desert. If that happens, it is a finding, not a failure.

## Sources

- Jaeger, Riedl, Djedovic, Vervaeke & Walsh (2024), [Naturalizing relevance realization](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1362658/full)
- Vervaeke & Coyne, *Mentoring the Machines*: [interview on autopoiesis and caring](https://theunspeakablepodcast.libsyn.com/artificial-intelligence-for-dummies-or-at-least-normies-john-vervaeke-and-shawn-coyne-on-mentoring-the-machines)
- Di Paolo (2005), [Autopoiesis, adaptivity, teleology, agency](https://openevo.eva.mpg.de/literaturebase/di-paolo-2005-autopoiesis-adaptivity-teleology-agency/2/)
- Odling-Smee, Laland & Feldman, [*Niche Construction*](https://research-portal.st-andrews.ac.uk/en/publications/niche-construction-the-neglected-process-in-evolution-2/)
- Mahmood & Sutton (2013), representation search through generate-and-test
- [Binance public data](https://github.com/binance/binance-public-data/) (aggTrades, 1s klines)
