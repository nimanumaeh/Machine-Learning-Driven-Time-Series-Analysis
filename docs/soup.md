# The soup: life that is not designed

The planet (`docs/planet.md`) holds organisms we designed: their organs, values, dials and modes are ours. The soup goes further. It holds no organisms at all, only matter, space, time, energy and a few physical laws. Replication, death, mating, predation, cooperation, size limits and trading are not written anywhere; whatever of them appears does so because the laws allow it. BTC is the sun.

Code:
- `evotrader/physics.py`: the laws, written once and compiled for the CPU and for CUDA.
- `evotrader/soup.py`: the world.
- `evotrader/weather.py`: what matter senses.
- `evotrader/gpu.py`: the world on a GPU.
- `evotrader/transplant.py`: reading organisms back as strategies.

Run: `python -m evotrader soup --minutes --from 2021-01-01 --until 2025-01-01 --run runs/soup` (add `--device gpu` on a CUDA machine; on cloud GPUs see `docs/gpu.md`).

---

## 1. The laws (all that is set)

| Law | What it says | Why this law |
|---|---|---|
| **Matter** | Every site holds 64 bytes. Bytes are rewritten, never created or destroyed. | Conservation of matter bounds each cell, not each organism: an organism may span many sites. Mass conservation is what bounds creatures in Flow-Lenia without any cap. |
| **Space** | Sites lie on the planet's lattice: latitude is timescale, longitude decides which market streams can be sensed. Interactions happen only between neighbors (within 2 sites). | Locality is what lets spatial structure, niches and colonies exist. |
| **Chemistry** | Two neighboring tapes are joined and run as one program for at most 8,192 steps. Ten instructions move two heads, change bytes, copy between heads, and loop (BFF, Agüera y Arcas et al. 2024). Every other byte is inert. | Copying exists; replication does not. In BFF, self-replicators arise from random matter with no fitness function. |
| **Senses and acts** | Four more instructions: `S` reads a market stream, `E` reads the site's own energy or position, `A` sets its exposure to BTC, `T` moves a bite of energy between the two sites. Each acts for the site whose half of the tape it sits in, and never takes itself as its own argument. | The market enters only through these. A trading strategy is whatever sequence of these the matter happens to run. |
| **Energy** | Each site holds one exact BTCUSDT perpetual account (fees, spread, funding, liquidation). Energy enters and leaves only through the market, plus the losses below. Orders fill at the open after the site's latitude ticks. | The money of the world is real trading money, so survivors' wealth is market wealth. |
| **Time (Kleiber)** | Each tick a site wakes with a chance proportional to (its energy / its starting stake)^0.75. About `--interactions` sites wake a tick while every site holds its stake; more in a richer world, fewer in a poorer one. | Energy buys activity, with diminishing returns for the very large (Kleiber's law of metabolic scaling). A world that loses energy slows down. |
| **Meeting** | A woken site claims a random neighbor. A site takes part in at most one interaction a tick: where claims collide, the highest random claim on both sites wins. | Crowding, not a quota, decides who meets whom. No site can act in two places at once, so even the richest site has a speed limit. This is also what lets every pair run at the same time on a GPU. |
| **Heat (Landauer)** | Every irreversible byte write costs the writing site a little energy. A site that cannot pay cannot write. | Information and energy trade one for one (Landauer; Bennett 2003; the Maxwell's demon argument of Linson et al. 2018). Thinking, copying and growing all cost; matter without energy is inert, which is what death is here. What is too big for the information it harvests runs too hot and fades: a whale limit with no cap written anywhere. |
| **Bites and digestion** | One `T` moves at most one small quantum, and only part of it arrives (80% by default). | Predation must be organized (draining a neighbor takes a loop, and time) and is never lossless, as in every food web (Lindeman's trophic efficiency). |
| **Noise** | Each tick a few bytes flip at random, on average `--noise` per byte per `--interactions` sites' worth of meetings. | Variation. |

**Conventions chosen so that the physics leans nowhere.** Each was found by running random matter on real data and asking why it leaned one way:
- Every byte the market reads or writes is signed, and 0 means nothing. The zero that ends every loop is stillness, not a position.
- Senses are percentiles of each stream's own recent history (histogram equalization, as a fly's photoreceptors encode contrast: Laughlin 1981). The median reads 0, and every value is equally likely.
- Opcodes sit in mirror pairs at ±(64−k), so code read as data is as often positive as negative.

## 2. What is left to emerge

- **Replication:** a program that copies its bytes into a neighbor.
- **Death:** a site's energy runs out (its matter can no longer write), or its matter is overwritten.
- **Mating:** two joined tapes copy parts of each other.
- **Predation:** overwriting a neighbor takes its body and energy; loops of bites drain it.
- **Parasitism:** running on another program's copy loop.
- **Cooperation:** giving bites to neighbors that carry the same code.
- **Multicellularity:** structures spanning sites.
- **Trading:** sensing, transforming and acting.
- **Size:** whatever heat, Kleiber and the exchange's brackets allow.

None of these is written. With the laws above, what pays is what persists.

## 3. Reading organisms back as strategies

An organism is code, so the organism is the strategy (`transplant.py`):
1. **Transplant.** Lift any matter (one site's tape, a lineage, a region) and plant a colony of it in a sandbox with the same physics, latitude and senses as its birthplace, with evolution switched off.
2. **Test on unseen data.** It lives through data it has never seen. Its positions over time are its strategy. Its P&L there is the test, split into market, fees, funding and heat, and compared with holding BTC.
3. **Read what drives it.** The senses its positions move with describe it, for example "follows the retail long/short account ratio".

This works however strange the matter becomes, because nothing about it has to be understood to be run. A whole evolved soup can be run forward the same way: a fund of everything that survived.

## 4. What we measure

- **Complexity:** high-order entropy (Shannon entropy minus compressed size, in bits per byte). Near zero for random matter, it rises sharply when copies of a few programs fill the soup.
- **Life:** distinct tapes, the most common ones and whether they copy themselves, and copies per interaction.
- **Energy:** where it goes (fees, funding, heat, digestion), predation volume, and net money against holding BTC.
- **Skill:** transplanted organisms against random matter on unseen data.

Planned measures:
- colonies (connected regions of identical matter);
- individuals as Markov blankets (Linson et al. 2018: an agent is the set of states shielded from the rest by its sensory and active boundary);
- Bedau–Packard evolutionary activity (which components persist far longer than chance).

## 5. Choosing the sun

The world's energy must come from BTC in a way that survives being reverse engineered. Options tried so far:

| Sun | What happened |
|---|---|
| Trading at up to 125×, every second, with rain refilling dead sites | Random matter traded itself to death: 92% of the planet's energy went in fees in one day. The rain paid random matter to gamble: 7 million USDT injected in one day. |
| Trading at up to 125×, no rain | Most sites were dead within a day; only matter that never trades kept its energy. |
| Trading at up to 1× (signals), minute data | Gentle: random matter loses about its fees, roughly 0.7% a week, and selection has time to act. Leverage becomes a deployment choice, by Kelly sizing on the edge a strategy shows. |
| Fractional bites (a `T` could take everything) | 1.8 billion USDT changed hands in a week on a planet of 4.1 million. With digestion losses, that sloshing burned the planet in two weeks. Quantized bites fixed it. |

Why trading P&L is the right sun at all: a Kelly bettor's growth rate equals the mutual information between its signal and the outcome (Kelly 1956). Energy from trading is information about BTC converted to energy, the same equivalence that grounds Landauer heat.

## 6. Abiogenesis: life from random bytes

Matter alone, with no market and nothing rewarding anything (`experiments/soup_emergence.py`, the exact BFF chemistry). The run: 16,384 random tapes on a 2,048 × 8 lattice, seed 2.

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

The same search with the market instructions included found nothing in about 10⁹ interactions. S and E overwrite bytes with readings, which seems to inhibit the origin of life. Life can still reach the market: replicators born in matter alone can seed a market soup (`--matter`, `--layout bff`), and selection can take them from there.

This run used soup physics 1's way of pairing (every tape starts one interaction an epoch, one after another). Physics 2 pairs sites by claims, all at once, and can run the same search at the scale of the original paper (2^17 tapes) on a GPU.

## 7. Open questions

- **Life meets the market.** Seed a market soup with the replicators above and see whether selection turns copying into trading.
- **Calibrating heat.** Heat sets how much complexity can pay for itself. The gentle (no heat) and Landauer worlds are run side by side to see what it changes.
- **Compute.** With physics 2 the CPU needs about 0.6 ms a tick at 4,096 sites (21 minutes for 4 years of minutes); 1-second data has 60 times more ticks. A GPU runs every site at once (`docs/gpu.md`).

## 8. Versions of the laws

- **Physics 1** (until October 2026): each second a fixed number of initiators were drawn in proportion to energy^0.75 and ran one after another. A site could meet several neighbors in a tick. Bytes flipped after the interactions, rain could refill dead sites, and the weather was computed by the planet in numpy.
- **Physics 2** (`soup.PHYSICS`): the laws above. A world records its physics in `meta.json`, and a run made under one version does not resume under another.

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
