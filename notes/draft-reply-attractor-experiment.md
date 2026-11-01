# Draft reply — next experiment / two-attractor hypothesis

Attach: `outputs/bush2024/gasping_native_rotation/m2020-19_g3_population_dynamics_anchor_softnorm_basis_contrast.png`

---

Hi [Name],

I tried to test an idea about the gasping dynamics and it didn't survive, so let me report that
first.

The thought was this. The gasping panels in Figure 7 are projected into the PCA space fit on the
100% O2 period and applied to the whole recording, so "ballistic, all-or-none" might be describing
gasping seen through eupnea-shaped coordinates rather than the network itself. ED Fig 8 seemed to
support the worry — fit separately per condition, gasps come out the least similar to O2 of
anything tested — and ED Fig 10e has 39% of expiratory units turning inspiratory, which is more than
enough sign-flipping for the eupneic components to cancel rotational structure.

So I measured it directly: take each epoch's phase-averaged cycle, fit the basis to that epoch
rather than to baseline, and ask whether the cycle is rank-one (every unit rising and falling
together, which is what ballistic means geometrically) or whether it encloses area.

In our own anoxia recording the basis looked like it mattered a great deal — the gasping cycle came
out rotational in its own coordinates, more so than eupnea. But when I ran the same analysis on our
sorts of m2020-19 g3 and g4, the gasping cycle is a retraced line in its own basis: rotation index
0.05–0.11, against 0.87–1.13 for baseline in the same recordings. g3 has 32 gasps, more than our 20,
and its protocol is the close match to ours. **So the native-basis test doesn't rescue the rotation
in your animal — it reproduces your result.** I think our recording is the outlier and I wouldn't
lean on it.

What does hold up is narrower: in those gates the eupneic basis captures 4–8% of gasping's variance
against 44–49% natively. The projection genuinely loses most of the gasping signal, which is what ED
Fig 8 says. But losing variance is not the same as destroying rotation, and I had been conflating
the two.

The thing that did come out of this, and that I think matters more, is about sorting. m2020-19 has
tissue movement during gasping — we see 45 to 110 µm steps after the first gasp, and they persist.
With uncorrected sorting the units present at baseline split into new clusters across that move, and
then every post-gasp epoch comes out essentially orthogonal to baseline, capturing about zero per
cent of its variance. The same splitting inflates the apparent gasping rotation roughly tenfold in
g3. We ended up building a waveform-anchored correction to recover unit identity across the move.

The reason I raise it: any subspace-similarity or correlation-structure measure computed across
gasping inherits that sensitivity, ED Fig 8 and the correlation result in Fig 7i–k included. The KS3
sort is drift-corrected and the paper's g3 units do largely track under our anchoring, so it is
nowhere near the uncorrected worst case. But it is a real confound sitting in exactly the comparison
that matters, and I haven't seen it addressed. Happy to show you the before and after.

None of this touches the attractor prediction itself, which I still think is the right experiment.
It means the latent-space argument needs a different test than the one I tried — and that for the
VGlut2-through-hypoxia experiment we should plan the drift handling up front, because the tissue
moves at exactly the moment we care about.

The recovery limb is still untouched by anyone, and that is where I would look next.

Best,
Leila
