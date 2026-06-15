# Magma

Implements Magma, a drop-in wrapper that masks an adaptive optimizer's updates and modulates them by momentum-gradient alignment.

Magma builds on the observation that randomly masking parameter updates acts as a curvature-dependent regularizer that smooths the optimization trajectory. Rather than masking uniformly, Magma scales the surviving updates by how well the current stochastic gradient agrees with the accumulated momentum: a high cosine similarity keeps the update near full strength, while a poorly aligned gradient is suppressed. Parameters are partitioned into disjoint blocks $b$, and a Bernoulli mask plus an alignment score are applied per block on top of the update $\Delta_t^{(b)}$ produced by a base optimizer (Adam, RMSProp, LaProp, or Muon), whose moments are always updated densely.

$$
\begin{aligned}
\tilde{s}_t^{(b)} &= \mathrm{sigmoid}\!\left(\frac{\mathrm{cossim}\!\left(m_t^{(b)}, g_t^{(b)}\right)}{\tau}\right) \\
s_t^{(b)} &= 0.9\, s_{t-1}^{(b)} + 0.1\, \tilde{s}_t^{(b)} \\
z_t^{(b)} &\sim \mathrm{Bernoulli}(0.5) \\
\theta_{t+1}^{(b)} &= \theta_t^{(b)} - s_t^{(b)}\, z_t^{(b)}\, \Delta_t^{(b)}
\end{aligned}
$$

where $\theta^{(b)}$ are the parameters of block $b$, $g_t^{(b)}$ the stochastic gradient, $m_t^{(b)}$ the first-moment (momentum) estimate, $\Delta_t^{(b)}$ the update direction from the base optimizer, $\mathrm{cossim}$ the cosine similarity, $\tau>0$ a temperature ($\tau=2$ in experiments), $s_t^{(b)}$ the EMA-smoothed alignment score, and $z_t^{(b)}$ an independent Bernoulli$(0.5)$ mask drawn per block each step.

Reference: Taejong Joo, Wenhan Xia, Cheolmin Kim, Ming Zhang, Eugene Ie, "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers", arXiv 2026. https://arxiv.org/abs/2602.15322

---
[Back to the Canon](../README.md)
