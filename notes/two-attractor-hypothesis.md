# Two-attractor models for the eupnea → gasping → eupnea transition

Working notes, 2026-11-01 (rev. 2).

Hypothesis under test: eupnea is governed primarily by the inspiration-off (I-off) attractor;
a concurrent inspiration-on (I-on) attractor dominates when inhibition falls away during the
transition to gasping, and reanchors I-off during recovery.

Sources:
- **B24** — Bush et al. 2024, *Latent neural population dynamics underlying breathing,
  opioid-induced respiratory depression and gasping*, Nat Neurosci 27:259–271.
- **B25** — Bush, Oliveira, Glovak & Ramirez 2025, *An inspiration-off attractor supports the
  robust and flexible control of breathing*, bioRxiv 2025.09.23.678177.

> **Revision note.** Rev. 1 of this document concluded that the transition is "autonomous
> oscillator → excitable system," based on B24 Fig 7. **That conclusion is withdrawn as the
> leading model** — see §1. It rests on a projection into a basis fit to eupnea, and the
> paper's own Extended Data contradicts the main-figure reading.

---

## 0. What the fitted dynamics are in eupnea (unaffected by §1)

B24 Fig 4j, rSLDS fits to "regenerative" recordings (n = 37/116 at K = 2):

| Partition | Dynamics | Counts |
|---|---|---|
| Inspiration | spiral, **mostly unstable** | 37/37 spirals; 28/37 **unstable** |
| Expiration | **stable node** | 32/37 nodes, 5/37 saddles, 3/37 rotational |

Eupnea: inspiration is a **repeller**, the I-off attractor is the expiratory **stable node**.
Their intersection generates the alternation. These are baseline fits and stand regardless of
§1.

---

## 1. The basis problem — why B24's gasping result cannot be read at face value

### The methods line

> "We fit PCA on data obtained during **100% O2 presentation** and apply that PCA decomposition
> to **the entire recording**." — B24 Methods, PCA decompositions

So every main-figure latent result for gasping (Fig 7e–g: ballistic trajectories, "stable region
between gasp efforts," loss of pre-/post-inspiratory activity) is **gasping data viewed through
a eupnea-defined lens**. None of it establishes what the gasping network is doing in its own
coordinates.

### The paper's own Extended Data contradicts the main-figure reading

| Figure | Finding | Implication |
|---|---|---|
| ED Fig 8 | PCA fit **separately per condition**, principal angles computed. **Gasps are the LEAST similar to O2** of every condition tested — below room air, sighs and morphine (n_gasp = 38) | The gasping subspace is substantially rotated away from the eupneic one. The eupneic basis is the wrong basis for gasping, *by the authors' own measurement*. |
| ED Fig 10e | **39% of expiratory units become inspiratory** during the gasp; 10% the reverse | Massive single-neuron phase reorganisation — matches the e097 breath-phase-shift observation |
| ED Fig 10g | A **"gasp only"** unit category (FR ratio ≥ 5) | Recruitment of units not engaged in eupnea |
| ED Fig 10f | Units become **more** respiratory-tuned during gasping | Not a degradation of respiratory organisation |

### Mechanistic account of the artifact

This is the part that matters. If ~39% of expiratory units flip to inspiratory, then the eupneic
PCs — whose weights come from a covariance structure in which those units load with *expiratory*
sign — will partially **cancel** flipped units against unflipped ones when applied to gasping
data. Rotational structure, which depends on the orderly phase lag between the two groups,
cancels first. What survives the projection is the common-mode inspiratory excursion.

That is precisely "ballistic, all-or-none, pre- and post-inspiratory activity lost." **The
artifact is predicted by the paper's own ED Fig 10e.** A rotation that has reorganised into a
new subspace and a rotation that has been destroyed look identical through a fixed eupneic
basis.

### Is the rSLDS refit a defence?

Only partly. They *did* refit during gasping (n = 11), so that result is not a projection
artifact. But:

- **d = 2 and K = 2 are imposed.** If the gasping process needs more dimensions (network
  expansion, §2) or a different partition structure, "eigenvalues no longer cleanly separate
  into distinct dynamics for inspiration and expiration" is exactly what **model
  misspecification** looks like. It is not independent evidence that I/E structure is lost.
- **Same units.** No refit recovers neurons the probe never sampled.

So both headline gasping results — "ballistic" and "eigenvalues don't separate" — have a
model/basis reading that is at least as good as the biological one.

---

## 2. Network expansion

Recent e097 analyses: a region **rostral to the VRC** is recruited specifically during gasping,
consistent with prior literature. Consequences:

1. **The state space itself grows.** Gasping is not the eupneic network in a different regime;
   it is a partly different network. You cannot test an attractor-handover hypothesis in a basis
   that does not span the new network.
2. **The dimensionality prediction inverts.** Rev. 1 predicted a dimensionality *collapse*
   (M5). Measured in the eupneic basis it will appear to collapse; measured natively it may well
   **rise**. Always report both.
3. **Possible sampling ceiling.** If the recruited region lies outside the probe span, no
   analysis recovers it. **Check e097's rostral extent before committing** — and see §5.
4. B25's whole framework models manipulations as input fields `V_k u_t` on a **fixed** `A_k`.
   Gasping changes `A_k` *and* the space `A_k` acts on. It is outside that framework's scope.

---

## 3. The revised hypothesis

> The inspiratory/gasping attractor lives in a subspace **partly orthogonal to the eupneic
> manifold**, supported by a partly different and more rostral population. During eupnea it is
> unobservable — that population is not engaged, and the eupneic basis does not span it. As
> inhibition withdraws, it is recruited and becomes dominant. Recovery is the re-establishment
> of the eupneic subspace and the recapture of the state into it.

This makes "reanchoring" literal and geometric rather than metaphorical, and it explains why
the published analysis saw a collapse: a rotation that moved out of the measured subspace reads
as a rotation that stopped.

It also rescues the percolation premise. Pre-inspiratory percolation is observed at the
single-neuron level in e097 and is consistent with ED Fig 10e/g; its absence from B24's Fig 7h
is a statement about the eupneic projection, not about the network.

### Remaining alternatives
- **Excitable regime** (rev. 1's reading) — still possible, now confounded. Must be tested in a
  gasping-native basis before it can be claimed.
- **Continuous degeneration, no switch** (B24's own discussion) — still the null for the
  forward limb, and also stated in the eupneic basis, so also weakened.
- **Neuromodulatory reconfiguration** — B24 discussion: gasping is "selectively dependent on
  serotonergic and noradrenergic mechanisms," so disinhibition may be necessary but not
  sufficient. See §5.

---

## 4. The asymmetry

Entry into gasping is fast and fairly stereotyped; return to stable eupnea is slow and more
variable across animals. **Neither paper analyses the recovery limb** — B24 has 38 gasping
recordings but stops at gasping. This is open ground.

Separable sources:

1. **Asymmetric slow-variable kinetics** — the control parameter falls fast and recovers slowly
   (energy state, ionic gradients incl. Cl⁻ reversal, vesicle pools, monoaminergic tone).
2. **Critical slowing down** near the reverse transition — predicts recovery is slow *in a
   specific window*, with rising variance and lag-1 autocorrelation.
3. **Destabilisation vs recapture** — going in, the state is pushed off a cycle
   (deterministic, stereotyped); coming out, it must be recaptured (noise-assisted,
   probabilistic). Predicts **low across-animal variance in onset, high in recovery**.
4. **Asymmetric recruitment/de-recruitment** — new: if gasping involves recruiting a population,
   the asymmetry may simply be that recruitment is faster than de-recruitment and
   re-coordination. **Measurable as a principal-angle time series** (§5.1).
5. **True hysteresis** — only if the fast subsystem is genuinely bistable.

**Caveat:** slow passage through a bifurcation produces a direction-dependent transition delay
and hence an apparent hysteresis loop **with no bistability at all**. Separate them by varying
the rate of the challenge: a genuine bistable loop keeps finite width as drift rate → 0; a
slow-passage loop collapses.

---

## 5. Revised analysis program — basis-aware throughout

### On existing data

1. **Principal angles as a continuous time series.** ED Fig 8 did this per *condition*; do it
   in sliding windows through **both limbs**. This turns "network reconfiguration" into a
   continuous order parameter, and comparing the forward and reverse subspace sequences is the
   hysteresis test in the right coordinates. **Highest-value single analysis.**
2. **Rotation strength in the gasping-native basis.** Refit PCA/rSLDS to gasping alone and ask
   whether rotational structure exists there — jPCA rotation strength, fraction of complex
   eigenvalues, enclosed trajectory area. **This is the direct test of the "ballistic" claim.**
   If rotation is present natively, Fig 7 is a basis artifact.
3. **Cross-projected variance.** How much gasping variance does the eupneic basis capture, and
   vice versa? Quantifies how misleading the fixed-basis projection is.
4. **Dimensionality per epoch, in native bases** (participation ratio / PCs for 80% var).
   Report native *and* eupneic-projected — the divergence between them is itself the result.
5. **Per-unit preferred-phase trajectories** through the transition (connects directly to the
   e097 phase-shift observation). Critically, distinguish:
   - existing units **continuously drifting** their preferred phase earlier, vs
   - a **distinct pre-I population being recruited**.
   These are different mechanisms and the rostral-recruitment claim predicts the second.
6. **Rostro-caudal band participation per epoch.** B24 already has the machinery (Fig 4p
   dropout analysis). Redo it per epoch: mid-VRC (preBötC/BötC) was required for regenerative
   dynamics in eupnea — **what is required during gasping?**
7. **Locate the "gasp only" units** anatomically. Do they carry the pre-I percolation?
8. **Eigenvalue tracking** (`Re(λ_insp)`, `Re(λ_exp)`, separation) — still worth doing, but
   **only in per-epoch native bases**, and reported alongside the fit diagnostics. In a fixed
   eupneic basis it is not interpretable.
9. **Forward/reverse superposition** of the latent path against a common physiological index.
10. **Early-warning signals** — variance + lag-1 autocorrelation through both limbs.
11. **Onset vs recovery timing variance across animals.**

**Practical caveats:** (a) only "regenerative" recordings support the rSLDS route — 37/116 in
B24, and it depended on the probe sitting mid-VRC; (b) if the gasping-recruited region is rostral
to the probe span, items 6–7 hit a sampling ceiling.

### New experiments

12. **Probe placement must span the gasping-recruited region.** This is the design implication
    of §2 and it conditions everything else: with a VRC-only span you cannot observe the
    population whose recruitment the hypothesis is about. Worth settling before the
    stimulation experiment is run.
13. **Time-resolved reset curves.** Brief randomly-timed pulses through the whole trajectory
    turn the PRC into a time-resolved measurement of the landscape; B25's reset-curve intercept
    with y = 1 becomes a scalar order parameter.
14. **Graded → all-or-none.** If the system does enter an excitable regime, stimulation should
    switch from graded phase shifts to all-or-none triggering with a refractory period (PRC
    type II → type I). Now a *test* of the excitable reading rather than a prediction from it.
15. **Concurrency control in normoxia.** If an inspiratory attractor coexists during eupnea
    rather than being built en route, a strong well-timed excitatory pulse in normoxia should
    produce a gasp-like episode that **outlasts the stimulus**. Under the §3 hypothesis the
    prediction is weaker — the supporting population may not be available in normoxia — which
    itself makes the result informative either way.
16. **Bidirectional kick during recovery** — excitatory should push back toward gasping,
    inhibitory should snap prematurely to eupnea, at parameter values where neither happens
    spontaneously. Cleanest positive evidence for coexistence.
17. **Rate variation** — true hysteresis vs slow passage (§4 caveat).
18. **Disinhibition as discriminator.** If gasping were disinhibition alone, pharmacological
    disinhibition should reproduce the gasping signature *including* the subspace rotation and
    the rostral recruitment. If it needs monoaminergic reconfiguration and recruitment, it will
    not. **Strengthened by §2:** it is much less likely that disinhibition alone recruits a new
    population, so this experiment now discriminates on a concrete, measurable axis (principal
    angle to the eupneic subspace) rather than on a qualitative impression of the dynamics.

---

## 7. Result: gasping-native rotation in e097 (2026-11-01)

Script `analysis/population/gasping_native_rotation.py` in the sorting_v2 repo; outputs in
`outputs/e097/gasping_native_rotation/`. Per epoch it builds the phase-averaged cycle, fits the
basis natively vs projecting through the baseline basis, and measures the cycle's **rank** and
**enclosed area in its own leading plane**, cross-validated across odd/even cycles.

### Anoxia #1 (`population_dynamics_v2`, 28 units, 20 gasps), softnorm / sqrt

| epoch | n | λ2/λ1 | RI cycle | RI cross-val | var in baseline basis | var native | cos∠ to baseline |
|---|---|---|---|---|---|---|---|
| baseline | 189 | 0.08 / 0.05 | 0.76 / 1.04 | 0.80 / 1.03 | 0.51 / 0.64 | 0.51 / 0.64 | 1.00 |
| anoxia | 79 | 0.09 / 0.07 | 0.75 / 0.98 | 0.69 / 0.98 | 0.48 / 0.57 | 0.50 / 0.62 | 0.93 / 0.74 |
| **gasping** | 20 | **0.23 / 0.13** | **1.13 / 1.10** | **1.06 / 1.08** | **0.41 / 0.07** | 0.62 / 0.58 | **0.60 / 0.21** |
| early recovery | 79 | 0.23 / 0.16 | 0.79 / 0.98 | 0.78 / 0.97 | 0.23 / 0.48 | 0.49 / 0.63 | 0.31 / 0.81 |
| mid recovery | 267 | 0.24 / 0.16 | 0.51 / 0.76 | 0.51 / 0.77 | 0.31 / 0.57 | 0.52 / 0.62 | 0.35 / 0.87 |
| later recovery | 220 | 0.18 / 0.10 | 0.62 / 0.82 | 0.61 / 0.81 | 0.33 / 0.60 | 0.50 / 0.63 | 0.36 / 0.91 |

### What this supports

1. **The gasping cycle is not rank-1.** A ballistic out-and-back means every unit rises and falls
   together — one component. λ2/λ1 for gasping is 0.23 / 0.13, the **highest of any epoch** and
   ~3× baseline. The second dimension is *more* prominent during gasping, not less.
2. **It encloses area in its own plane.** RI 1.13 / 1.10, the highest of any epoch and ≥ baseline,
   and it **survives cross-validation across gasps** (1.06 / 1.08). So with 20 gasps the rotation
   is consistent between halves of the data, not an artifact of a thin average.
3. **The eupneic basis is the worst possible basis for gasping.** It captures 0.41 / **0.07** of
   gasping variance against 0.62 / 0.58 natively — in the sqrt normalisation an **8× loss**. And
   gasping's cos∠ to baseline (0.60 / 0.21) is the lowest of every epoch, reproducing B24's
   ED Fig 8 within e097.

**Reading:** in e097 anoxia #1 the gasping population trajectory is rotational in its own
coordinates — by these measures at least as rotational as eupnea. That is direct support for the
basis argument of §1: B24's "ballistic, all-or-none" is, here, substantially an artifact of
viewing gasping through a eupnea-fit basis.

### What this does not establish

- **One animal, one trial, 20 gasps, 28 units.** Anoxia #2 has only 8 gasps and does **not**
  replicate cleanly: RI 0.98 → cross-validated **0.11** in softnorm (it holds in sqrt, 1.07 →
  1.04). Treat anoxia #2 as underpowered, not as a replication.
- RI slightly exceeds 1 for gasping; the circle reference is not a hard ceiling for non-elliptical
  loops, so read the values as relative, not absolute.
- Phase is defined from the diaphragm, so the measure presumes a cycle exists and asks whether the
  population traverses it in ≥2 dimensions. It does **not** test B24's onset-aligned picture
  directly — an onset-triggered average is still worth adding.
- Nothing here addresses the **rostral recruitment** claim: 28 units, and the probe span question
  (§5, caveat b) is untouched.

### Methods that failed, recorded so they are not retried

- **Per-unit circular shuffle of the mean cycle is an invalid null.** Rolling each unit
  independently spreads unit phases uniformly, which is exactly the configuration that produces a
  perfect circle, so the "null" scored *above* every real epoch (p95 ≈ 1.0 against real values of
  0.01–0.96). Replaced by odd/even cross-validation across cycles.
- **jPCA on a single mean cycle is underpowered here.** The skew-symmetric fit explained only
  0.06–0.29 of the derivative variance in *every* epoch including baseline, so the recovered jPC
  plane did not generalise across cycle halves. jPCA is defined across many condition-trajectories;
  one breath-average is one trajectory. The cycle-rank measure above replaced it.
- **Measuring enclosed area in PC1/PC2 of the binned data fails its positive control.** e097's PC1
  holds ~0.5 of the variance, much of it not respiratory, so even the eupneic cycle renders as a
  flat line (RI 0.42 with a visibly non-rotational trajectory). The plane must come from the
  cycle's own covariance.

### Next

The recovery limb is the open question and the two normalisations **disagree**: in softnorm the
subspace never returns to baseline (cos∠ 0.31–0.36 through later recovery, 220 cycles), in sqrt it
does (0.91). That bears directly on reanchoring (§3) and should be resolved before anything is
claimed about the recovery limb.
