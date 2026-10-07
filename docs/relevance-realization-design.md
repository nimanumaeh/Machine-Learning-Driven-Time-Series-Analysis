# Relevance-realizing traders: a formal design

How to build agents whose ways of knowing are derived (*emanate*) from how
they participate in a market, and whose "consciousness" is the recursive
function Vervaeke describes: relevance realization applied to relevance
realization, which takes over when the habitual scripts stop working.

The document is in three parts:
- **Sections 1–7: the theory, made precise.** What each idea becomes as a mathematical object.
- **Sections 8–10: what it is for and how we would know it works.** The money angle, the tests, and the build plan.
- **Sections 11–12: reference.** A lookup table and the sources.

---

## 0. Where the project stands

**What exists already: the participatory substrate.**
- `exchange.py`: an exact BTCUSDT perpetual account. This is the agent's *body*: long or short, leverage, isolated margin, liquidation, fees, spread and funding.
- `world.py`: an ecology where equity is life. Death, reproduction and selection only read equity.
- `data.py`: real market stimuli from 2017 to now.

**What does not exist: everything that could realize relevance.**
- **The agents can't learn.** Today's agents are fixed functions from 14 numbers to a position, chosen by evolution. Nothing in them learns during a lifetime, attends, frames a situation, notices that a habit has failed, or learns how to learn. Selection over fixed functions is emergence by selection, and nothing more.
- **They face no relevance problem.** Every agent sees the same small, hand-picked feature vector, for free. So there is no frame problem, and without a frame problem there is nothing for relevance realization to solve.

**What this design claims, and doesn't.** It makes no claim that consciousness will appear. In a paper Vervaeke co-authored, Jaeger et al. (2024) argue that relevance realization can be simulated by algorithms but never fully instantiated by them, because algorithms live in "small worlds" that have already been formalized. What we can do is build the *organization* that relevance realization has, as faithfully as a formal system allows. Then we test whether agents with that organization keep their grip on the market better than agents without it.

---

## 1. Commitments

| | Commitment | Why |
|---|---|---|
| C1 | **Precarious being.** The agent's being is its equity under exact market physics. It can be liquidated and it can die. | Relevance is grounded in self-maintenance: something matters to the agent because it bears on the agent continuing to exist. |
| C2 | **Scarcity.** The agent faces an open-ended space of possible aspects of the world but has a fixed attention budget, much smaller than that space. | Without scarcity there is no relevance problem, and so no frame problem. |
| C3 | **Emanation.** Each way of knowing is *defined as following from* a more basic one (procession). Its source stays present in it (remaining). It is judged by how well it serves that source (reversion). | This is the user's "emanance". It is also Proclus' law that every effect remains in, proceeds from and reverts to its cause (*Elements of Theology*, prop. 35). |
| C4 | **Concurrency.** All four ways of knowing run at once, on different timescales. Their dependence on one another is real but partial, and adjustable. | Vervaeke gives a grounding order, but not a sequence in time. |
| C5 | **Recursion.** The operator that realizes relevance in the world is also applied to the agent's own processes. | This is consciousness as higher-order relevance realization. |
| C6 | **Data is stimulus, not the whole arena.** The arena contains: the market streams, the agent's own body, its own processes, its memory, and optionally its society. | The data is one class of stimulus among several. |

How "emanation" is used here, precisely:
- **Emergence:** higher-level structure is put together, bottom-up, from parts that already exist.
- **Emanation:** the derived level follows *necessarily* from the nature of its source. Plotinus' image is light from the sun: not assembled by chance, and not arbitrary design.

So we design **only the participatory level** (body, coupling, precariousness, scarcity), plus the fixed *laws* by which each level follows from its source. **The content of the levels is never hand-set.** Which aspects become salient, which frame forms, which skills and which claims: all of that is fixed by the agent's participation and its history. Learning still flows in both directions. The constitution is emanative; the content is learned.

---

## 2. Arena, body and participation

**Streams (the stimuli the agent can sample):**
- the market streams $m_t$ (the data);
- the agent's own processes $\iota_t$ (section 5);
- optionally, the population's aggregate behaviour $s_t$ (section 6).

**Body.** The body is $b_t = (W, q, \bar E, M, L)$:
- $W$: wallet;
- $q$: signed position in BTC;
- $\bar E$: entry price;
- $M$: isolated margin;
- $L$: leverage.

Its transition $T$ is exact: it is `exchange.py`. Equity is $E(b_t, p_t)$.

**Participation.** Let $x$ be a target exposure: a signed multiple of equity, so $x = 5$ means 5x long. Let $\xi$ be the arena's path over the next $h$ minutes. Define

$$\Pi_h(b, x; \xi) \;=\; \log E\big(T_h(b, x, \xi)\big) \;-\; \log E(b).$$

$\Pi$ is how the arena's motion becomes the agent's change of being. It is **transjective**:
- it is not a property of the price path alone;
- it is not a property of the agent alone;
- it is a property of the coupling between them.

In trading this is literal. Holding a position *is* taking part in the market's motion. Effective leverage $\lambda(b) = qp/E$ is the degree of participation. Flat means not taking part.

**Small-move form** (used for fast inner computation; realized outcomes always use the exact $T$):

$$\mathbb{E}[\Pi_h] \;\approx\; x(\mu_h + \tfrac12\sigma_h^2 - \phi_h) \;-\; \tfrac{\kappa}{2} x^2\sigma_h^2 \;-\; c\,|x-\lambda| \;-\; \pi_h(L)\,\ell(x)$$

where:
- $\mu_h$ and $\sigma_h$ are the drift and volatility of log price over $h$, so $\mu_h + \tfrac12\sigma_h^2$ is the arithmetic drift: a plain 1x long earns exactly $\mu_h$;
- $\kappa \ge 1$ is the agent's risk weight ($1$ is full Kelly);
- $\phi_h$ is expected funding;
- $c$ is the fee plus half-spread per unit of notional;
- $\pi_h(L) \approx 2\Phi(-d_L/\sigma_h)$ is the chance of touching the liquidation price within $h$. This follows from the reflection principle. The log-distance $d_L$ depends only on leverage, because the margin is isolated;
- $\ell(x) = -\log(1-|x|/L)$ is the log-loss of the isolated margin.

The first two terms are Kelly's: maximizing them is maximizing compounded growth.

**Grip** (the norm of participatory knowing: being *fitted* to the arena). The agent keeps a predictive distribution $Q$ over its own participation, for its actual exposure or, when flat, for the exposure it intends. Grip is how much better than a naive baseline the agent anticipates its own being:

$$\Gamma_t \;=\; \mathrm{EMA}_t\Big[\log Q_{t-h}\big(\Pi^{\text{real}}\big) - \log Q^{\text{naive}}_{t-h}\big(\Pi^{\text{real}}\big)\Big].$$

Grip is not profit. An agent can anticipate its own losses perfectly; reversion (section 3.5) and selection deal with that. Loss of grip is *participatory surprise*: the felt "this isn't working".

**Identity** (slow participatory variables): horizon $h$, leverage $L$, margin share $f$ (so $|x| \le fL$), and niche. Identity is what the agent *is* in relation to the arena. It changes slowly over a lifetime, and across generations by evolution.

---

## 3. The four ways of knowing as an emanative hierarchy

Grounding order (Vervaeke): participatory → perspectival → procedural → propositional. Each level is defined from the one before it (procession), carries it inside it (remaining), and is judged by what it does for it (reversion).

### 3.1 Participatory: knowing by being coupled

This level is the body, $\Pi$, grip $\Gamma$ and identity, all defined in section 2. Nothing here is learned bottom-up. It is the given coupling. We design this level; everything else follows from it.

### 3.2 Perspectival: salience follows from participation

**Affordance value.** What the world offers this body, now:

$$V(b, o) \;=\; \max_{x\in\mathcal X(b)} \mathbb{E}_{\xi\sim f_\theta(\cdot\mid o)}\big[\Pi_h(b, x;\xi)\big]$$

where:
- $o$ is the vector of attended aspects;
- $f_\theta$ is the agent's learned model of the arena;
- $\mathcal X(b)$ is the set of exposures its identity and the leverage brackets allow.

**Perspective operator.** This is the expected gradient outer product, or *active subspace* (Constantine, 2015), of the affordance value:

$$C_t \;=\; \mathbb{E}_{o\sim\rho_t}\Big[\nabla_o V(b_t, o)\,\nabla_o V(b_t, o)^\top\Big].$$

Here $\rho_t$ is the distribution of plausible present states, the *here and now*: recent attended aspects, or the model's samples of them.

$C_t$ gives the salience landscape directly:
- **Salience** of aspect $i$ is $s_{i,t} = (C_t)_{ii}$: how much the world's offer to *me* changes when that aspect moves.
- **Figure and ground.** The frame $F_t$ is the top-$k$ eigenvectors of $C_t$ (the figure). The rest is ground.
- **Togetherness.** The off-diagonal terms of $C_t$ bind aspects into one situation.
- **Hereness and nowness.** $C_t$ is indexed by this body $b_t$ and this present $\rho_t$.

These match Vervaeke's adverbial qualia, which he says relevance realization explains.

**Why this is emanation and not emergence.** $C_t$ is a function of the body's participation. By the envelope theorem, away from the constraints,

$$\nabla_o V \;=\; x^*\,\nabla_o\mu_h \;+\; x^*\sigma_h^2(1-\kappa x^*)\,\nabla_o\log\sigma_h \;-\; x^*\,\nabla_o\phi_h \;+\;(\text{liquidation terms in } L).$$

So the same market shows up differently depending on how one is in it:
- A lightly participating agent sees drift-bearing aspects as figure: *opportunity*, which counts linearly in $x^*$.
- A heavily participating agent sees volatility-bearing aspects as figure: *threat*, which counts quadratically.
- A 50x agent close to liquidation sees almost nothing but the short-horizon tail.

**As implemented: the world's own step size instead of an infinitesimal one.** $V$ has flat regions and corners: the no-trade band that costs create, and the corner where the best move is to step out. There the gradient is exactly zero, and a 50x agent that ought to leave would see nothing as salient. So the operator uses finite differences at the scale aspects actually move (they are in standard units, so the step is $\delta = 1$):

$$\Delta_i V = \frac{V(o + \delta e_i) - V(o - \delta e_i)}{2\delta}, \qquad C_t = \mathbb{E}_{o\sim\rho_t}\big[\Delta V\,\Delta V^\top\big].$$

Where $V$ is smooth this is the expected gradient outer product above. It is checked in `tests/` with two bodies in one market:
- a flat 2x agent puts 99.7% of its salience on the drift aspect;
- an agent holding 8x at 50x leverage puts more than half on the volatility aspect.

No one designs what is salient. The **law** $b \mapsto C$ is fixed. The **content** comes from participation plus the learned arena model.

**Attention under scarcity.**
- The attended set $\mathcal O_t$ is a subset of the aspect grammar $\mathcal A$, with $|\mathcal O_t| \le B$.
- The grammar is streams × timescales × operators: returns, volatility, z-score, range, slope, the basis between trade and mark price, funding, cross-correlations, time since an event, the agent's own drawdown, and so on. That is thousands of possible aspects.
- Salience can only be computed for aspects that are attended. New candidates come from three places: exploration of the grammar, propositions (section 3.4), and culture.
- This is the frame problem in miniature, and that is deliberate.

**Two rules learned from building it.**
- **Act only on evidence.** The arena model's weights are shrunk toward zero by their own uncertainty, $w_{\text{eff}} = w\,\max(0,\, 1 - \operatorname{Var}(w)/w^2)$. The shrunk weights are used for salience and for action. Without this, Kelly sizing turns estimation noise into leveraged positions: fees were 3.5 times higher in early tests.
- **Calibrate volatility by outcome.** The log-volatility head gives the *shape* of volatility across states. Its *level* is rescaled by the ratio of realized to predicted squared errors. Uncorrected, short-horizon volatility was off by 2 to 4 times. This calibration is itself part of grip.

### 3.3 Procedural: scripts follow from perspective

A script is $\omega_k = (\kappa_k, \pi_k, \varepsilon_k)$, defined over the coordinates of the frame it formed in, $z = F_{t_k}^\top o$:
- $\kappa_k(z) \in [0,1]$: where the script applies;
- $\pi_k(z, \lambda) \to x$: what it does;
- $\varepsilon_k$: its *expectation*, a predictive distribution of $\Pi$ while it runs. This is its little story of what should happen to me.

**How scripts behave:**
- **Running unconsciously.** The active script acts, cheaply, as long as its local surprise $u_{k,t} = -\log \varepsilon_k(\Pi^{\text{real}})$ stays within tolerance. This is "the script that goes along until it doesn't work".
- **Learning.** A script learns online (actor-critic) with reward $\Pi^{\text{real}}$, the log-growth of being.
- **Automatization.** When conscious deliberation (section 5) finds a better action, the active script is trained toward it. What was conscious becomes habit.
- **Arbitration** between habit and deliberation uses their relative reliability. This generalizes Daw, Niv & Dayan (2005).
- **Reversion.** A script is worth what it realizes of the affordance perspective reveals. Its regret is $\rho_k = \mathbb{E}[V(b,o) - \Pi^{\text{real}}]$ while it is active.
- **Re-framing.** When the frame changes, scripts are re-expressed in the new coordinates, or retired.

### 3.4 Propositional: claims follow from skill

A proposition is $\varphi_j = (c_j, \text{claim}_j, \text{credence}_j)$:
- $c_j$: a sparse condition, a conjunction of at most 3 thresholds on attended aspects or frame coordinates.
- **claim**: either about the arena (for example "$\mathbb{E}[r_h \mid c_j] < 0$", or "$\sigma_h$ is high") or about skill ("script $k$ works in $c_j$").
- **credence**: a Beta posterior under a proper scoring rule. It is updated only on data that arrives *after* the proposition was formed, so it is always out-of-sample.

How propositions work:
- **Genesis (procession).** They are abstracted from procedural experience: sparse rules that predict where scripts succeed or fail.
- **Use.**
  - In the workspace: they point attention at aspects, suggest which script to run, and support counterfactual "if $c$ then ..." reasoning.
  - They can be communicated to other agents, and to us.
- **Reversion.** A proposition's value is how much it improves script selection when consulted. True-but-useless propositions fade. False ones die.

### 3.5 Concurrency and partial dependency

All four levels update at once, on timescales

$$\tau_{\text{proc}} \;\ll\; \tau_{\text{persp}} \;\ll\; \tau_{\text{prop}} \;\ll\; \tau_{\text{part}}$$

that is: every decision; minutes to hours (or on events); days; a lifetime or generations.

**Dependency is soft.** Each level holds a cached copy of what it derives from its source and refreshes it at a rate $\gamma \in [0,1]$. For example, the frame scripts use is refreshed through its projector:

$$P^{\text{used}} \leftarrow (1-\gamma_t)\,P^{\text{used}} + \gamma_t\,F_tF_t^\top,$$

and then re-orthonormalized.
- $\gamma \to 0$: habits run detached from the current perspective. This is cheap and fast.
- $\gamma \to 1$: the level is re-grounded in its source.

Relevance realization sets the $\gamma$s. They rise with surprise and fall with stable grip. This is the precise sense of "a chain of dependency, but not 100% and not ordered in time".

**Reversion runs back up the chain.**
- Script regret pushes re-landscaping (perspectival).
- Persistent loss of grip pushes identity revision (participatory).
- False propositions push revision of the scripts they came from.

---

## 4. Relevance realization as opponent processing

Following Vervaeke, Lillicrap & Richards (2012), relevance is not optimized by a single objective. It is *realized* by self-organizing opponent processes.

Each dial $d \in [0,1]$ is the balance of two opposed pressures $P_\pm \ge 0$, each of which limits itself:

$$\dot d \;=\; \eta\,\big[P_+(1-d) - P_-\,d\big] \qquad\Longrightarrow\qquad d^* = \frac{P_+}{P_+ + P_-}.$$

The pressures are computed from the agent's own state, so the balance moves continuously with context. No one picks the setting.

| Dial | Pushes toward the first pole | Pushes toward the second pole | What it controls |
|---|---|---|---|
| **Tempering**: explore ↔ exploit | persistent surprise while learning progress has stalled | grip, together with positive learning progress | temperature of script selection; rate of exploring the aspect grammar; frame temperature |
| **Scope**: particularize ↔ compress | residuals that depend on context, $I(u; c)$ | redundant scripts; too little evidence per script | frame dimension $k$; splitting and merging scripts |
| **Prioritization**: diversify ↔ focus | entropy of the salience landscape; uncertainty | one dominant figure (concentrated salience) | weight on the risk terms in $V$ (variance, liquidation); margin kept in reserve; spread across horizons |
| **Resiliency ↔ efficiency** | instability of grip (the arena is changing) | stability and success | rate of generating new aspects, scripts and propositions versus pruning them |

**Learning progress** is the drop in a level's error between consecutive windows:

$$LP_{\ell,t} = \bar e_{\ell,[t-2w,\,t-w]} - \bar e_{\ell,[t-w,\,t]}.$$

This is the signal that separates *learnable* surprise from noise (Oudeyer & Kaplan, 2007). In a market, where most surprise is noise, it decides whether relevance realization chases ghosts.

**Edge of criticality.** The resiliency/efficiency balance is tuned so that internal reorganizations stay near critical: their avalanche sizes are heavy-tailed, and their branching ratio is about 1. The system should be neither frozen nor chaotic.

---

## 5. Consciousness: recursive, higher-order relevance realization

**The internal milieu.** These are the aspects of the agent itself:

$$\iota_t = \big[\Gamma,\ \Delta\Gamma,\ u_{\text{active}},\ \text{conflict},\ \text{novelty},\ LP_\ell,\ d_\bullet,\ \gamma_\bullet,\ H(s),\ \angle(F_t, F_{t-1})\big]$$

where:
- **conflict** is how much scripts and propositions disagree;
- **novelty** is the distance to the nearest episodic memory;
- $H(s)$ is the entropy of the salience landscape;
- the angle measures how much the frame has drifted.

**Ignition** (a global workspace, after Baars and Dehaene). Consciousness engages when

$$w^\top\iota_t > \theta_{\text{on}},$$

and stays engaged until $w^\top\iota_t < \theta_{\text{off}}$ (hysteresis). In practice that means situations that are novel, complex (conflict) or ill-defined (no script applies, salience is diffuse): the conditions Vervaeke gives for consciousness. While engaged:
- the states of every level become available to every other (broadcast);
- consciousness costs part of the same attention budget $B$, so it competes with perception. The agent must learn *when* being conscious is worth it.

**What the workspace does** (its interventions $m$):
1. **Re-landscape:** recompute $C_t$ with the current model, setting $\gamma_{\text{persp}} \to 1$.
2. **Re-frame:** attend to new aspects at the current frame temperature. This is frame-breaking.
3. **Scripts:** switch, inhibit, compose, spawn (copy and perturb) or merge them.
4. **Deliberate:** act on the model-based optimum $x^* = \arg\max_x \mathbb{E}[\Pi]$, and train scripts toward it.
5. **Propositions:** consult, form or retire them.
6. **Memory:** recall similar episodes and use them to simulate interventions before choosing one.
7. **Identity:** revise horizon or leverage. This is rare.

**Learning about learning.**
- A meta-policy $\mu(m \mid \iota)$ (a contextual bandit) is rewarded by
$$R^{\text{meta}} = \Delta\Gamma_{[t,\,t+w]} + \beta\,LP - \text{cost},$$
  that is, by whether the intervention *restored grip and restarted learning*.
- Its value function $V^{\text{meta}}(\iota)$ then gets **the same perspective operator** applied to the agent's own processes:
$$C^{\text{int}}_t = \mathbb{E}\big[\nabla_\iota V^{\text{meta}}\,\nabla_\iota V^{\text{meta}\top}\big].$$
- $C^{\text{int}}_t$ says which of its own processes matter right now. That is relevance realization applied to relevance realization (C5). Two levels of recursion are enough in practice.

**Insight** (restructuring, after Stephen & Dixon).
- **Impasse:** low $\Gamma$, $LP \approx 0$ for a while, and local interventions have failed.
- **Breaking frame:** the workspace raises the frame temperature, meaning the entropy of attention allocation and frame choice. It then searches re-framings.
- **Making frame:** when a configuration restores grip, the temperature anneals down.
- **Observable signature:** attention entropy rises and then falls before $\Gamma$ recovers, and the frame jumps (a large principal angle).

Insight is a phase transition in the perspective, not a parameter tweak.

---

## 6. Memory, extended mind, social arena

- **Episodic memory** (Budson, Richman & Kensinger, 2022: consciousness as a memory system for flexibly recombining episodes).
  - **What is stored:** an episode is written during each conscious engagement: frame coordinates, body, active script, intervention, and what then happened to $\Gamma$ and $\Pi$.
  - **What it is used for:**
    - novelty;
    - simulating candidate interventions before acting;
    - offline replay between eras ("sleep"), using only the agent's own past.
- **Extended mind.** The aspect grammar is a tool library. Propositions are external notes. Memory is a journal.
- **Social arena (optional).** Agents can see the population's aggregate positioning, leverage and recent liquidations, and a culture pool of propositions with track records. This is a world the agents make themselves, on top of the data.

## 7. Evolution and lifetime

The genome stops being a strategy. It carries priors and meta-parameters:
- the split of the attention budget and the initial aspects;
- the gains of the dial pressures;
- the ignition weights;
- the $\gamma$ rates;
- the identity priors ($h, L, f$);
- learning rates.

Lifetime learning supplies the content. The ecology that already exists selects on viability, so evolution selects *learners* (the Baldwin effect), not strategies that fit the past.

---

## 8. Money

- **Being is equity, and every value is log-growth of equity** (Kelly). So surviving and reproducing *is* compounding, and nothing else is rewarded.
- **What comes out for us:**
  1. **Agents.** Live paper trading, then candidates for small, strictly limited real capital.
  2. **Propositions with out-of-sample track records.** These are readable trading rules, with provenance.
  3. **Ecosystem instruments:**
     - aggregate grip, which can serve as a regime-change alarm when many agents lose grip at once;
     - consensus salience: what matters in the market right now.
  4. **The affordance sizer.** $x^* = \arg\max \mathbb{E}[\Pi]$ is a principled position sizer: Kelly with fees, funding and liquidation risk. It is useful on day one, before anything consciousness-like exists.

## 9. How we'd know

- **Out-of-sample by construction.** The most recent year stays sealed. Walk forward through eras.
- **Ablations**, each compared on survival and growth across regime changes:
  1. no consciousness (scripts only);
  2. bottom-up learned attention instead of emanative salience;
  3. no propositions;
  4. no scarcity (attend to everything);
  5. today's fixed-function agents.
- **Null worlds:**
  - shuffled returns, where no stable proposition should survive;
  - planted structure, which should be found;
  - a planted decoy aspect that correlates early and then stops, which should be learned and then *let go*. This is a frame-problem test.
- **Signatures:**
  - ignition rises at regime breaks and falls with mastery;
  - the insight signature (attention entropy up, then down, then grip recovers);
  - perspective differs systematically with participation (agents near liquidation attend to volatility);
  - proposition credences are calibrated.

## 10. Build plan

Each step is tested before the next one starts.

1. **Done: body model and affordance value** (`body.py`). Checked against exact expectations, Brownian first-passage and finite differences.
2. **Done: aspect grammar and the online arena model** (`aspects.py`, `mind.py`). 186 aspects, an attention budget of 10, recursive least squares with forgetting for drift and log-volatility.
3. **Done: perspective** (`mind.py`). $C_t$, frames and salience, and salience-driven attention. The emanation property is tested.
4. **Scripts.** Expectations, online learning, and habit/deliberation arbitration.
5. **Consciousness.** Internal milieu, ignition, interventions, the meta-bandit, insight annealing, and episodic memory.
6. **Propositions.** Abstraction, credence, and use in the workspace.
7. **Opponent-processing dials.**
8. **Evaluation.** Ablations, null worlds and dashboards.
9. **Evolution of meta-parameters; culture and the social arena** (optional).

---

## 10b. Borrowed math

Nothing here needs new mathematics. Each piece is an established result,
reinterpreted. The contribution is the arrangement: what is derived from
what, and what each part is judged by.

| Piece | Established math | Reinterpreted as |
|---|---|---|
| Body value $V$ | Kelly (1956), Itô's correction, reflection principle (first passage) | what the market offers *this* account |
| Arena model | recursive least squares with forgetting; shrinkage toward zero (ridge, James–Stein) | what the attended aspects say about drift and volatility, believed only on evidence |
| Perspective | active subspaces (Constantine 2015) / expected gradient outer product; envelope theorem | salience and frame, derived from participation |
| Anticipation | query–key dot-product attention (Vaswani et al. 2017), token embeddings | query = the body, keys = aspect structure (stream + operator + scale), output = where to look next |
| Scarce attention | top-k gating (mixture of experts), hard attention / glimpses (Mnih et al. 2014) | the attention budget |
| Learnable vs noise | learning progress (Oudeyer & Kaplan 2007) | reducible versus irreducible surprise |
| Opponent processing | two opposed first-order rates, $d^* = P_+/(P_+ + P_-)$ | the relevance-realization dials |
| Ecology | evolution strategies with self-adaptive mutation | lineages |

**Next steps, also borrowed:**

| Piece | Established math | Reinterpreted as |
|---|---|---|
| Loss of grip | Bayesian online changepoint detection (Adams & MacKay 2007): the posterior over time since the last regime change | ignition: the moment the scripts stop working |
| Learning about learning | IDBD (Sutton 1992): per-feature step sizes learned by meta-gradient, explicitly "learning feature relevance" | each aspect's learning rate and forgetting, tuned by experience |
| Scripts | options (Sutton, Precup & Singh 1999); Recurrent Independent Mechanisms (Goyal et al. 2021): top-k modules activated by attention, the rest untouched | habits that run until they don't |
| Workspace | shared global workspace (Goyal et al. 2022): specialists compete through attention for a limited-capacity workspace that is broadcast back | consciousness as limited-capacity broadcast |
| Insight | simulated annealing; entropy-driven search | breaking and making frames |

**Where transformer attention is deliberately *not* used:** the arena model.
Minute-level BTC returns have a very low signal-to-noise ratio, and a
sequence transformer would fit the noise. Recursive least squares with
shrinkage is the right tool there. Attention earns its place where the
problem is choosing among many items given a state: which aspect to look at,
and which module gets the workspace.

## 11. Lookup table

| Vervaeke | Formal object | Trading meaning |
|---|---|---|
| Participatory knowing (agent–arena, belonging) | coupling $\Pi$, grip $\Gamma$, identity $(h, L, f)$ | having a position; being moved by the market; the kind of trader one is |
| Transjectivity | $\Pi(b, x; \xi)$ depends on both $b$ and $\xi$ | a price move is profit, loss or nothing depending on one's position |
| Perspectival knowing (salience landscape, presence) | $C_t$: active subspace of $V$, frame $F_t$, salience $s$ | what matters *to me, now, in this position* |
| Adverbial qualia (here, now, together) | indexing by $b_t$ and $\rho_t$; off-diagonals of $C_t$ | situational awareness |
| Procedural knowing | scripts $(\kappa, \pi, \varepsilon)$ in frame coordinates | trading routines |
| Propositional knowing | $(c, \text{claim}, \text{credence})$, judged out-of-sample | explicit trading rules |
| Emanation (procession, remaining, reversion) | each level defined from its source, refreshed at rate $\gamma$, judged by its source | |
| Relevance realization (opponent processing) | dials with $d^* = P_+/(P_+ + P_-)$ | risk appetite, focus, exploration, adaptivity |
| Consciousness (higher-order relevance realization) | ignition on $\iota$; interventions; meta-bandit; $C^{\text{int}}$ | noticing that the playbook has stopped working, and changing *how* one learns |
| Insight | entropy-driven re-framing, a jump in $F$ | seeing the market differently after a regime break |
| Autopoiesis / precariousness | equity as being, liquidation, death | survival |

## 12. Sources

- Vervaeke, Lillicrap & Richards (2012). Relevance realization and the emerging framework in cognitive science. *Journal of Logic and Computation*. [ResearchGate](https://www.researchgate.net/publication/220387969_Relevance_Realization_and_the_Emerging_Framework_in_Cognitive_Science)
- Vervaeke & Ferraro (2013). Relevance realization and the neurodynamics and neuroconnectivity of general intelligence. [ResearchGate](https://www.researchgate.net/publication/299812171_Relevance_Realization_and_the_Neurodynamics_and_Neuroconnectivity_of_General_Intelligence)
- Andersen, Miller & Vervaeke (2022). Predictive processing and relevance realization. [PhilPapers](https://philpapers.org/rec/ANDPPA-11)
- Jaeger, Riedl, Djedovic, Vervaeke & Walsh (2024). Naturalizing relevance realization: why agency and cognition are fundamentally not computational. [Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1362658/full)
- Budson, Richman & Kensinger (2022). Consciousness as a memory system. [BU summary](https://www.bumc.bu.edu/camed/2022/10/03/new-explanation-for-consciousness/)
- Stephen & Dixon (2009). The self-organization of insight: entropy and power laws in problem solving. *Journal of Problem Solving*.
- Constantine (2015). *Active Subspaces*. SIAM.
- Daw, Niv & Dayan (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience*.
- Oudeyer & Kaplan (2007). What is intrinsic motivation? *Frontiers in Neurorobotics*.
- Sutton, Precup & Singh (1999). Between MDPs and semi-MDPs (options). *Artificial Intelligence*.
- Kelly (1956). A new interpretation of information rate. *Bell System Technical Journal*.
- Vaswani et al. (2017). Attention is all you need. *NeurIPS*.
- Mnih, Heess, Graves & Kavukcuoglu (2014). Recurrent models of visual attention. *NeurIPS*.
- Sutton (1992). Adapting bias by gradient descent: an incremental version of delta-bar-delta. *AAAI*. [mlanthology](https://mlanthology.org/aaai/1992/sutton1992aaai-adapting)
- Adams & MacKay (2007). Bayesian online changepoint detection. [arXiv:0710.3742](https://ar5iv.arxiv.org/html/0710.3742)
- Goyal et al. (2021). Recurrent independent mechanisms. *ICLR*. [mlanthology](https://mlanthology.org/iclr/2021/goyal2021iclr-recurrent)
- Goyal et al. (2022). Coordination among neural modules through a shared global workspace. *ICLR*. [arXiv:2103.01197](https://arxiv.org/abs/2103.01197)
- Proclus. *Elements of Theology*, prop. 35.
- Baars (1988), *A Cognitive Theory of Consciousness*; Dehaene, *Consciousness and the Brain* (2014).
