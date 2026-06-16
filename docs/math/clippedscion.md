# Clipped Scion

Implements Clipped Scion, a norm-constrained LMO method that fuses gradient-norm clipping into the Scion update.

Clipped Scion specializes the Generalized Gradient Norm Clipping (GGNC) template to the (product) max-norm geometry, recovering a clipped variant of the unconstrained Scion algorithm. A momentum buffer $d_t$ is fed to a linear minimization oracle over the unit norm ball to produce a normalized direction $v_t$; the step size is then adaptively clipped by the threshold $\rho$ against the dual-paired quantity $\langle d_t, v_t\rangle$. Choosing the per-layer norm recovers familiar oracles: the $\ell_\infty$ ball gives $\mathrm{lmo}(d)=\mathrm{sign}(d)$ (sign descent), while the spectral norm gives the Muon-style $\mathrm{lmo}(W)=\mathrm{msign}(W)$.

$$
\begin{aligned}
d_t &= \alpha\, g_t + (1-\alpha)\, d_{t-1} \\
v_t &= -\,\mathrm{lmo}(d_t) \\
\eta_t &= \min\{\rho,\ \langle d_t, v_t\rangle\} \\
\theta_{t+1} &= \theta_t - \gamma\, \eta_t\, v_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the stochastic gradient, $d_t$ is the momentum estimate ($d_0=0$), $\alpha \in (0,1]$ is the momentum coefficient, $\mathrm{lmo}(d) \in \arg\min_{x:\lVert x\rVert\le 1}\langle d, x\rangle$ is the linear minimization oracle over the (per-layer) norm ball, $\rho>0$ is the clipping threshold, and $\gamma \in (0,1)$ is the step size. Without clipping ($\eta_t = \langle d_t, v_t\rangle$ unbounded, or $\rho\to\infty$) the update reduces to plain unconstrained Scion.

Reference: Thomas Pethick, Wanyun Xie, Mete Erdogan, Kimon Antonakopoulos, Antonio Silveti-Falls, Volkan Cevher, "Generalized Gradient Norm Clipping & Non-Euclidean (L0, L1)-Smoothness", arXiv 2025. https://arxiv.org/abs/2506.01913

---
[Back to the Canon](../index.md)
