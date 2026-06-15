# Scion

Implements Scion, a norm-constrained linear-minimization-oracle optimizer.

Scion maintains a gradient average and applies the linear minimization
oracle (LMO) of a chosen norm ball to it, which adapts the update to the
geometry of the problem. With `momentum` denoting one minus the usual
decay factor, the update is


$$
\begin{aligned}
d_t &= (1 - \mu)\, d_{t-1} + \mu\, g_t \\
\theta_t &= (1 - \gamma)\, \theta_{t-1}
    - \gamma\, \rho\, \mathrm{lmo}_{\mathcal{C}}(d_t)
\end{aligned}
$$

where $\gamma$ is the learning rate, $\mu$ is `momentum`,
$\rho$ is `scale` (the radius of the norm ball $\mathcal{C}$),
and $\mathrm{lmo}_{\mathcal{C}}$ is the linear minimization oracle. The
weight-shrinkage term $(1 - \gamma)\theta_{t-1}$ keeps the iterate
inside the norm ball; setting `unconstrained=True` drops it, giving the
unconstrained variant


$$
\theta_t = \theta_{t-1} - \gamma\, \rho\, \mathrm{lmo}_{\mathcal{C}}(d_t).
$$

The norm ball is selected per parameter group via `norm` (one of
`'Auto'`, `'SpectralConv'`, `'ColNorm'`, `'RowNorm'`, `'BiasRMS'`,
`'Spectral'`, or `'Sign'`); for matrix parameters the spectral LMO
orthogonalizes the gradient average through a Newton-Schulz iteration.

Reference: Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu,
Antonio Silveti-Falls, Volkan Cevher, "Training Deep Learning Models with
Norm-Constrained LMOs", ICML 2025.
https://arxiv.org/abs/2502.07529

---
[Back to the Canon](../README.md)
