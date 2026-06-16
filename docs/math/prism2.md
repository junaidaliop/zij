# PRISM

Implements PRISM, a gauge-invariant differentially private optimizer for LoRA fine-tuning.

A LoRA update $Z = AB^\top$ is non-identifiable: many factor pairs $(A,B)$ encode the same $Z$, so applying DP-SGD directly to the factors injects gauge-dependent noise that can be unboundedly amplified. PRISM instead clips and perturbs the *intrinsic* tangent update on the fixed-rank manifold $\mathcal{M}_r$, so the released step depends only on $Z$ and not on the chosen factorization. Per-example gradients are lifted into the tangent space, clipped by a global intrinsic norm, averaged, and perturbed with an isotropic tangent-space Gaussian; the result is passed through a gauge-invariant adaptive preconditioner whose ridge floors are scaled to the known DP noise level, then retracted back onto $\mathcal{M}_r$.

For LoRA module $\ell$ with frozen weight $W_0$ and factors $(A_\ell, B_\ell)$, one PRISM step over a minibatch of $b$ samples is

$$
\begin{aligned}
\Delta Z_{i,\ell} &= P_{A_\ell,B_\ell}(G_{i,\ell}), \qquad
s_i = \Big( \textstyle\sum_{\ell} \lVert \Delta Z_{i,\ell} \rVert_F^2 \Big)^{1/2}, \qquad
\alpha_i = \min\!\Big\{ 1, \frac{C}{s_i} \Big\} \\
\Delta Z_\ell^{\mathrm{dp}} &= \frac{1}{b} \sum_{i=1}^{b} \alpha_i \, \Delta Z_{i,\ell} \; + \; \frac{\sigma C}{b}\, P_{A_\ell,B_\ell}(\Xi_\ell), \qquad \Xi_\ell \sim \mathcal{N}(0, I) \\
m_\ell &\leftarrow \beta_1 m_\ell + (1-\beta_1)\,\Delta_\ell^{\mathrm{dp}}, \qquad
V_\ell \leftarrow \beta_2 V_\ell + (1-\beta_2)\, \frac{(\Delta_\ell^{\mathrm{dp}})^\top \Delta_\ell^{\mathrm{dp}}}{d_\ell} \\
U_\ell &= m_\ell\, (V_\ell + \lambda_\ell I)^{-1/2}, \qquad
\lambda_\ell \asymp \frac{\sigma C^2}{b}\,\frac{\mathrm{tr}(\cdot^{-1})}{r} \\
Z_\ell^{+} &= \mathrm{Retr}_r\!\big( Z_\ell - \eta\,(U_{A,\ell} B_\ell^\top + A_\ell U_{B,\ell}^\top) \big)
\end{aligned}
$$

where $P_{A_\ell,B_\ell}$ projects a full gradient $G_{i,\ell}$ onto the rank-$r$ tangent space $T_{Z_\ell}\mathcal{M}_r$ (gauge invariant), $C$ is the intrinsic clip threshold, $\sigma$ the noise multiplier, $\eta$ the learning rate, $\beta_1,\beta_2$ the moment decays, $m_\ell$ the first moment, $V_\ell \in \mathbb{R}^{r\times r}$ the right-invariant rank-space second moment, $d_\ell \in \{m_\ell, n_\ell\}$ the module dimension, $\lambda_\ell$ the DP-aware ridge floor (scaled by the known noise level and module geometry, e.g. $\lambda_{A,\ell} \asymp \tfrac{\sigma C^2}{b}\,\mathrm{tr}(N_\ell^{-1})/r$), and $\mathrm{Retr}_r$ the best rank-$r$ approximation (truncated SVD) onto $\mathcal{M}_r$. The moments and direction for $A_\ell$ and $B_\ell$ are computed analogously, giving $(U_{A,\ell}, U_{B,\ell})$.

Reference: Shihao Wang, Xueru Zhang, "PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA", ICML 2026. https://arxiv.org/abs/2606.00944

---
[Back to the Canon](../index.md)
