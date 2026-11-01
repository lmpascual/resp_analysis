# Draft reply — next experiment / two-attractor hypothesis

Hi [Name],

Yes — and I think the prediction is stronger than it first looks, because the main thing that
appears to argue against it in the 2024 paper may be a property of the analysis rather than of
the network.

The gasping panels in Figure 7 are projected into the PCA space fit on the 100% O2 period and
applied to the whole recording. So "ballistic, all-or-none," and the loss of pre- and
post-inspiratory activity, are statements about what gasping looks like through a eupnea-defined
lens rather than about what the gasping network is doing in its own coordinates.

What makes me fairly confident this matters is that the Extended Data already points the same
way. ED Fig 8 fits PCA separately per condition and computes principal angles, and gasps come
out the *least* similar to O2 of every condition tested — below room air, sighs and morphine.
And ED Fig 10e has 39% of expiratory units becoming inspiratory during the gasp. If that many
units flip sign, the eupneic PCs — whose weights assume the old phase relationships — will
partially cancel flipped against unflipped units when applied to gasping data. Rotational
structure depends on the orderly lag between those groups, so it cancels first, and what
survives the projection is the common-mode inspiratory excursion. Which is what Figure 7 shows.
A rotation that has moved into a new subspace and a rotation that has been destroyed are
indistinguishable through a fixed basis.

Two things we're seeing in e097 fit that reading. The preinspiratory percolation is clearly
there at the single-neuron level in the phase-shift analyses, even though it's absent from the
population projection — which is what you'd expect if it has reorganised out of the eupneic
subspace. And a region rostral to the VRC appears to be recruited specifically during gasping,
consistent with the older literature. If the network genuinely expands, the state space grows,
and we can't test an attractor-handover hypothesis in a basis that doesn't span the new network.

So I'd restate the prediction geometrically: the inspiratory attractor lives in a subspace
partly orthogonal to the eupneic manifold, carried by a partly different population. It isn't
visible during eupnea because that population isn't engaged and the eupneic basis doesn't span
it; as inhibition falls away it's recruited and becomes dominant; and recovery is the
re-establishment of the eupneic subspace and the recapture of the state into it. That makes the
reanchoring idea literal rather than metaphorical.

Before the stimulation experiment I'd want two analyses on recordings we already have. First,
refit the space to gasping alone and ask whether rotational structure is there natively — if it
is, the ballistic picture is a basis artifact and the whole framing shifts. Second, principal
angles in sliding windows through both the entry and the recovery, which turns the
reconfiguration into a continuous measure and lets us ask whether the two limbs follow the same
path. Nobody has looked at the recovery limb at all — the 2024 analysis stops at gasping — and
in e097, and it looks like in the newer animals too, the return to stable eupnea is considerably
slower and more variable across animals than the entry.

One design point follows: the probe has to span the recruited region, or we can't observe the
population the hypothesis is actually about. Worth settling before we commit.

And it sharpens your second experiment. If gasping were disinhibition alone, removing inhibition
should reproduce the whole signature including the subspace rotation and the rostral
recruitment. I'd be surprised if disinhibition alone recruits a new population — which makes
that a clean test on a measurable axis rather than on an impression of the dynamics.

Happy to go through any of it whenever.

Best,
Leila
