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
coordinates — by these measures at least as rotational as eupnea.

> ⚠️ **Superseded by §10.** This was read as support for the basis argument of §1 — that B24's
> "ballistic, all-or-none" is substantially an artifact of a eupnea-fit basis. **Testing two gates
> of one of the 2024 paper's own animals did not replicate it.** e097 anoxia #1 is the outlier, not
> the rule. The basis-cost part of the result does replicate; the native-rotation part does not.

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

---

## 8. The analysis in plain terms

### Input
From each run's cached `ctx.pkl`: `Z`, an (84,000 × 28) matrix for anoxia #1 — spikes in 5 ms bins,
10 ms Gaussian smoothing, square-rooted, 420 s × 28 units; `ph`, respiratory phase per bin; `bcond`,
the epoch each bin belongs to; `dia_df`, one row per breath.

### Per-unit scaling: sqrt vs softnorm
A choice made **before** PCA, and it matters because PCA finds directions of maximum *variance*, so
whichever units swing hardest numerically dominate the components.

- **sqrt** — what Bush & Ramirez do: `sqrt(smoothed counts)`. The square root is a variance-stabilising
  transform for Poisson-ish counts (variance becomes roughly independent of rate), but there is no
  per-unit rescaling, so an 80 Hz unit contributes far more variance than a 3 Hz unit regardless of
  which is more respiratory-modulated.
- **softnorm** — Churchland et al. 2012: convert to Hz, then divide each unit by `(its range + 5 Hz)`.
  Every unit's modulation is scaled to roughly 0–1. The `+5` keeps a near-silent unit (range ~0.5 Hz)
  from being blown up to full scale by division by something tiny. Makes the analysis about *who
  modulates with the breath* rather than *who fires fastest*.

Why it matters here: record F18 found e097's PC2 under the Bush-style PCA is almost entirely **one
tonic unit (#89, loading 0.997)** — PC2's steps are single spikes of that unit. If one neuron owns a
component, any rotation measured in that plane measures that neuron. Both are reported: softnorm is
the more trustworthy measure of population geometry, sqrt keeps it comparable to the published
analysis. Agreement across the two means the result is robust to the choice; disagreement (the
recovery limb) is flagged, not resolved by picking one.

### baseline vs native basis — the contrast that *is* the experiment
- **Native** — fit PCA to that epoch's own bins; coordinates from how units covary during gasping itself.
- **Baseline** — fit PCA to eupneic bins only, then project everything into those fixed axes. B24's
  approach, and what `population_dynamics.py` already did.

Why the choice can change the answer: the eupneic axes encode *who fires with whom during normal
breathing*. ED Fig 10e reports **39% of expiratory units become inspiratory during the gasp**, so
those axes now mix units that have swapped roles — units once anti-correlated are now correlated and
partially cancel. Rotation depends on an orderly sequence of phase lags across units, so it cancels first.

### The measures
Each epoch is first reduced to a **phase-averaged mean cycle**: its bins sorted into 50 bins of
respiratory phase and averaged → a 50 × 28 matrix. This puts a 0.3 s eupneic breath and a 3 s gasp
cycle on the same footing.

| measure | what it asks | ballistic | rotational |
|---|---|---|---|
| **λ2/λ1** | PCA the mean cycle itself; ratio of the 2nd to 1st component variance. If every unit rises and falls together the cycle is **rank 1**. A rotation needs ≥2 dimensions with a phase offset. | ≈ 0 | appreciable |
| **RI cycle** | Enclosed area of the closed loop in its own top-2 plane (shoelace), normalised by the ellipse its covariance implies. Does the population leave one way and return another, or retrace? | 0 | 1 = circle |
| **RI cross-val** | Split cycles odd/even, build a mean cycle from each, find the plane from **odd only**, measure the **even** cycle in it. Real consistent rotation survives; noise does not. | collapses | tracks the full estimate |
| **var captured** | Fraction of the epoch's total variance the top 3 components of each basis account for. Same data, two coordinate systems. | — | — |
| **cos∠** | Similarity of two *subspaces* regardless of what happens inside them: principal angles between the epoch's top-3 and baseline's top-3, via the SVD of their inner product. Mean cosine, 1 = identical, 0 = orthogonal. | — | — |

The table in §7 reports **RI cycle for the native basis only**; the baseline-basis version appears in
the figures. The cleanest quantification of the basis cost is **var captured** (0.41/0.07 vs
0.62/0.58) and **cos∠**, not an RI contrast.

### Reading the figures
`*_basis_contrast.png` is the one built for explaining the point; `*_mean_cycle.png` is the
all-epochs diagnostic.

- **Top row — each epoch in its own coordinates.** Each panel has its own PCA, so position,
  orientation, scale, and the sign of each axis are arbitrary. **Only the shape within a panel is
  meaningful; panels cannot be compared to each other by position.**
- **Bottom row — all epochs in one fixed plane**, the plane in which the eupneic cycle rotates best,
  carried over unchanged with a common origin. Here positions, sizes and orientations **are**
  comparable. (The all-epochs diagnostic figure re-fits a plane per panel, so its bottom row is *not*
  comparable across panels — a limitation of that figure, fixed in the contrast figure.)
- A **loop** means the population visits different states going out and coming back — activity sweeps
  through units in sequence. A **flat or retraced path** means every unit rises and falls together.
- For the baseline column the two rows are near-identical by construction — a sanity check.

### The markers
Verified empirically against the data, not assumed: **φ = 0.000 at inspiration onset** and
**φ = 3.142 at inspiration offset**. Mean-cycle bin 0 therefore spans `[−π, −π + π/25)`, which is
**exactly** the window B25's Methods use to define the inspiration-off point ("the mean point in the
latent space where respiratory phase is immediately after the offset of inspiration
[−π < φ < −π + π/25]"). Bins 0–24 are expiration, bins 25–49 inspiration.

- **Red dot = the inspiration-off point**, by the paper's own definition.
- **Open green circle = inspiration onset** (bin 25).
- Arrow = direction of travel.

### What the shared axes revealed
In the eupneic basis the gasping cycle does **not** become a line — it **shrinks to near-nothing**
(RI stays ~0.95; it is the amplitude that collapses, consistent with capturing only 7–41% of the
variance). That is precisely B24's own description in ED Fig 10i: "the transition from eupnea into
gasping is smooth and characterized by a progressive 'shrinking' of the rotational trajectories."
**The shrinking is in large part a basis effect** — natively the loop is large. State the finding
that way; do not claim they saw a line where we see a loop.

---

## 9. What this replicates in B24, and what it does not

### Replicates (in e097 — a different recording, not their data)
- **The subspace-similarity ordering of ED Fig 8.** Gasping is the least similar epoch to baseline
  (cos∠ 0.60 / 0.21) — same direction, same relative ordering. Matches the 0.214 already in this
  project's `pa_M`.
- **The "shrinking" of ED Fig 10i** — clearly visible in the shared-axes bottom row.
- **The eupneic-basis appearance of collapse** (Fig 7e–g): projecting into their coordinates
  reproduces their observation.

### Does NOT replicate, and was not tested
- **This is not a reanalysis of their data.** e097 is a different animal and protocol; their 38
  gasping recordings are untouched. The claim is only that *in our recording the basis choice changes
  the answer* — which is a reason to reanalyse theirs, not a refutation of their result.
- Single-unit findings (39% expiratory→inspiratory, "gasp only" units, tuning strength) — no
  unit-level analysis was run.
- The rSLDS eigenvalue result — no rSLDS was fit here.
- **Onset-aligned** trajectories: Fig 7e is onset-aligned, this is phase-averaged. Not the same view.
- "Pre- and post-inspiratory activity is lost" — not directly tested.
- The correlation-structure change (Fig 7i–k) — not tested.
- Rostral recruitment — not tested; 28 units, probe span unchecked.

### New here (not in B24)
- The **rank test** (λ2/λ1) and the **cross-validated enclosed area in the native basis** — the actual
  new measurement, and the thing that distinguishes "rotation destroyed" from "rotation moved."
- **Variance captured per basis**, which puts a number on the cost of a fixed basis.

### Does B24 address the limitation?
**Partly, and honestly — but it is not carried back to the interpretation.**

In their favour:
- **ED Fig 8 is the native-basis control, and they ran it**: PCA fit separately per condition,
  principal angles between subspaces, with a shuffled lower bound.
- They **refit the rSLDS during gasping** (n = 11), so the dynamics claim does not rest on projection
  alone.
- ED Fig 10i shows the transition is gradual rather than asserting a switch.

Against:
- ED Fig 8 is framed as *evidence that structure is disrupted* ("Low-dimensional structure is
  disrupted during gasping"), not as a caveat on how Fig 7e–g should be read. The two analyses are
  never connected.
- Every trajectory panel carrying the "ballistic, all-or-none" claim (Fig 7e–g, ED Fig 10h, 10i) is in
  the eupneic basis.
- Having established that the gasping subspace is the least similar of any condition, the next step —
  re-measuring trajectory geometry *in that native subspace* — is not taken.
- The native-basis dynamics analysis they did run imposes d = 2 and K = 2 on the same units, so
  "eigenvalues no longer cleanly separate" is equally consistent with model misspecification.

**Fair verdict: they identified and measured the problem, then left it in the supplement framed as
corroboration rather than as a qualification.** ED Fig 8 is arguably the strongest internal reason to
re-read Fig 7.

---

## 10. Replication on the 2024 paper's own animals — it does not hold (2026-11-01)

Our own sorts of **m2020-19 gates g3 and g4**, which are recordings from the 2024 paper, run through
the same analysis. Outputs in `outputs/bush2024/gasping_native_rotation/`.

**These are two gates of the SAME mouse.** With e097 that is two animals, not three.

### Baseline vs gasping, softnorm / sqrt

| run | units | gasps | baseline RI | **gasping RI** | gasping RI cv | var base | var native | cos∠ |
|---|---|---|---|---|---|---|---|---|
| e097 anoxia #1 | 28 | 20 | 0.76 / 1.04 | **1.13 / 1.10** | 1.06 / 1.08 | 0.41 / 0.07 | 0.62 / 0.58 | 0.60 / 0.21 |
| e097 anoxia #2 | 27 | 8 | 0.41 / 0.94 | 0.98 / 1.07 | 0.11 / 1.04 | 0.31 / 0.18 | 0.70 / 0.57 | 0.35 / 0.32 |
| **m2020-19 g3 anchor** | 30 | **32** | 1.13 / 1.05 | **0.11 / 0.05** | 0.40 / 0.36 | 0.04 / 0.08 | 0.44 / 0.49 | 0.06 / 0.28 |
| **m2020-19 g4 anchor** | 16 | 20 | 0.87 / 0.97 | **0.11 / 0.69** | 0.59 / 0.76 | 0.56 / 0.05 | 0.67 / 0.57 | 0.65 / 0.09 |

### 1. The native-rotation result does NOT replicate

In three of four anchor measurements the gasping cycle has RI **0.05–0.11** — essentially a retraced
line — against e097's 1.10–1.13. Meanwhile **baseline rotates cleanly in both gates** (0.87–1.13),
so the positive control passes and the measure is working.

This is not a power problem: **g3 has 32 gasps, more than e097's 20.** And g3's protocol is the
close match to e097 (room air → N2, no hypercapnia before the first challenge), so a protocol
difference does not explain it either.

**So in the paper's own animal, the native-basis test supports B24's ballistic conclusion.** The
claim that "ballistic" is a basis artifact is not supported. e097 anoxia #1 is one recording out of
four, and it is the outlier.

### 2. What does replicate: the basis is lossy

In 7 of 8 measurements the baseline basis captures far less of gasping's variance than a native
basis (0.04–0.56 vs 0.44–0.70). The projection genuinely loses most of the gasping signal — ED
Fig 8's point is real. But **losing variance is not the same as destroying rotation**, and the two
must now be stated separately.

### 3. Gasping is not uniquely rotated away

Recovery epochs are often as far from baseline as gasping is: g3 sqrt has recovery 2 at 0.14 against
gasping 0.28; g4 softnorm has early recovery at 0.28 against gasping 0.65. In 3 of 7 anchor
measurements gasping is *not* the lowest-cos∠ epoch. The subspace moves and tends not to come back,
rather than gasping being a special excursion.

### 4. Sorting stability dominates these measures — the biggest caveat of all

m2020-19 has gasping-induced tissue movement (45–110 µm steps after gasping) that splits units. The
uncorrected sorts against the anchor sorts:

| | g3 uncorrected | g3 anchor | g4 uncorrected | g4 anchor |
|---|---|---|---|---|
| gasping RI (softnorm) | **1.02** | **0.11** | 0.26 | 0.11 |
| early recovery cos∠ | 0.00 / 0.01 | 0.34 / 0.40 | 0.01 / 0.01 | 0.28 / 0.25 |
| late/2nd recovery cos∠ | 0.00 / 0.00 | 0.42 / 0.14 | 0.00 / 0.01 | 0.32 / 0.49 |
| var captured, post-gasp | 0.00–0.01 | 0.13–0.37 | 0.00 | 0.18–0.32 |

Two things follow:

- With uncorrected sorting, **every post-gasp epoch is essentially orthogonal to baseline and
  captures ~0% of its variance** — the baseline units stop existing as the same clusters after the
  move. A drift artifact manufactures exactly the signature we were testing for.
- Drift-induced splitting also **inflates apparent gasping rotation ~10×** in g3 (RI 1.02
  uncorrected vs 0.11 anchor).

e097 is drift-free (median shift −0.8 µm across the gasping transition), so its result is **not**
this artifact. But any subspace or correlation-structure claim measured across gasping is highly
sensitive to sorting stability — **and that includes B24's ED Fig 8 and the Fig 7i–k correlation
result.** The paper's KS3 sort was drift-corrected (datashift2) and its g3 units largely track under
our anchor correction (54/63), so it is far from the uncorrected worst case; but the sensitivity is
real and is probably the most useful thing to come out of this exercise.

### Revised position

- **Dropped:** "B24's ballistic result is substantially a basis artifact." One of four recordings
  supports it; the paper's own animal contradicts it.
- **Kept:** the eupneic basis is genuinely lossy for gasping, and ED Fig 8 already said so.
- **New and solid:** subspace measures across gasping are dominated by sorting stability, which
  neither paper addresses.
- **Open:** why e097 anoxia #1 differs. Between-animal variability, unit count (16–30 units is thin
  for this), or something about that preparation. Worth understanding before any of it is claimed.

> ⚠️ **Partly revised by §11.** Chasing the difference showed that **including units with little
> respiratory modulation dilutes the cycle measure**. Once that is corrected, g4 does show a gasping
> rotation. The contradiction is graded, not binary: e097 > g4 > g3. "m2020-19 contradicts e097" is
> too strong as stated above.

---

## 11. Chasing the e097 difference (2026-11-01)

Five explanations tested against what is on disk. Code:
`analysis/population/gasping_rotation_controls.py` in the sorting_v2 repo, one `--test` per control;
every number below reproduces from it.

### Ruled out

**Unit count / statistical power.** Subsampling e097's 28 units down to g4's 16, 200 draws:
median gasping RI **1.09**, IQR [1.00, 1.13], 5–95% [0.17, 1.18], only 8% of draws below 0.3. At
n = 20 and 24 it is tighter still. Drop-one jackknife moves RI only within 1.04–1.19. **So
m2020-19's lower unit counts cannot explain its RI of 0.05–0.11.**

**Rostral / gasp-recruited units carrying the rotation.** Within e097:

| subset | n | gasping RI |
|---|---|---|
| all units | 28 | 1.13 |
| shallow half (<1700 µm, near probe tip) | 14 | 1.17 |
| deep half (≥1700 µm) | 14 | 1.01 |
| **drop gasp-recruited (FR ratio ≥ 1.5)** | 25 | **1.20** |

Dropping the recruited units *raises* RI, and both ends of the probe rotate. And by B24's ED Fig 10g
categories **g3 has more gasp-recruited units than e097** (5 "gasp only" + 4 "gasp increasing" = 9,
against e097's 3 + 0). Probe depth coverage is also comparable (e097 9–2722 µm, g3 203–3280, g4
68–3419). **The rostral-recruitment account is not supported as the carrier of the rotation** —
though it says nothing about what those units do otherwise.

### Refuted

**Phase-shift coherence.** The idea was that e097's population shifts phase coherently (preserving
an orderly ordering) while m2020-19 scrambles. The resultant length of per-unit baseline→gasping
phase shifts is the opposite: **e097 0.21, g3 anchor 0.41, g4 anchor 0.32.** e097 has the *least*
coherent shift and rotates most.

**Phase bunching.** If every unit became co-active at the gasp the cycle would be rank-1. Circular
concentration of preferred phases during gasping is similar everywhere — e097 0.30, g3 0.27, g4 0.23
— so no recording bunches more than another. Weighting each unit by its modulation depth (so tonic
units stop counting as evidence of tiling) does not separate them either: 0.51, 0.46, 0.52.

### Partly supported, and the most useful finding

**Respiratory modulation depth, and which units are included.** Mean `modulation_L` during gasping:
**e097 0.56, g4 0.33, g3 0.09** — g3's average unit is below the pipeline's 0.2 tonic threshold.
Restricting to modulated units:

| run | all units | best-modulated half | phasic (L ≥ 0.2) | strongly phasic (L ≥ 0.4) |
|---|---|---|---|---|
| e097 | 1.13 | 1.12 (n=13) | 1.12 (n=17) | 1.12 (n=15) |
| **g4** | **0.11** | 0.55 (n=8) | **1.04 (n=9)** | **1.63 (n=4)** |
| g3 | 0.11 | 0.02 (n=15) | — (only n=3) | — (n=2) |

**Cross-validated** — units selected on odd gasps, rotation measured on even gasps, so the selection
cannot manufacture the result:

| run | n selected | RI (even, selected) | RI (even, all units) |
|---|---|---|---|
| e097 anoxia #1 | 15 | **1.04** | 1.10 |
| g4 anchor | 7 | **0.70** | 0.25 |
| g3 anchor | 16 | 0.38 | 0.29 |

### What this means

- **Part of the failed replication was my own analysis choice.** Including units with little
  respiratory modulation dilutes the mean cycle. B24 and B25 both put *all* recorded units into the
  PCA, which is defensible for a variance decomposition but costly for a cycle measure.
- **The contradiction is graded, not binary:** e097 (1.04) > g4 (0.70) > g3 (0.38) cross-validated.
  g4 does show a gasping rotation once unmodulated units are excluded.
- **g3 stays low even after selection** and has very few modulated units during gasping, so it is
  closer to "cannot be assessed" than to "shown to be ballistic."
- **Caveat:** the pipeline's `modulation_L` (occupancy-corrected) and the resultant length
  recomputed here from the mean cycle do not agree on unit counts — 3 vs 16 phasic units in g3. The
  cross-validated table uses one consistent measure throughout; the per-pipeline counts should not
  be mixed with it.

### Still unexplained, and the leading remaining candidates
- **Inspired O2.** e097's baseline is room air; the B24 protocol supplies 100% O2. That changes the
  desaturation trajectory and so possibly the gasping state itself. It cannot affect the *native*
  gasping measure directly (which never touches the baseline basis), but it can change what gasping
  is. **Untested.**
- **Sorting quality.** Residual unit-identity error during gasping scrambles phase relationships and
  would lower measured rotation. e097's sorting is manually checked with minimal drift; confidence
  in m2020-19 is lower even anchored. **Untested as the cause, and not separable from the above with
  these data** — but demonstrably the right family of worry: anchoring raises phase-shift coherence
  from **0.13 → 0.41** in g3 and **0.19 → 0.32** in g4, so a sorting correction alone moves this
  class of measure threefold. Coherence itself is not the mediator (anchored g3 is *more* coherent
  than e097 and still does not rotate); the point is only that these quantities track the sort.

Both predict the observed gradient. Distinguishing them needs either a room-air B24-animal recording
or a sorting-quality manipulation (e.g. degrade e097's sort and watch RI fall).
