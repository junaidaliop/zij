# SPlus

Implements SPlus, a stable whitening optimizer.

SPlus preconditions each matrix parameter with a Kronecker-factored,
Shampoo-style whitening of the gradient, but replaces the cached
matrix-inverse update with a bounded one that pairs a slowly updated
eigenbasis with an instantaneous sign normalization. For a matrix
parameter $\theta \in \mathbb{R}^{m \times n}$ with gradient
$G_t$, the optimizer maintains a momentum $M_t$ and two side
covariances $L_t, R_t$:


$$
\begin{aligned}
M_t &= \beta_1 M_{t-1} + (1 - \beta_1)\, G_t \\
L_t &= \beta_2 L_{t-1} + (1 - \beta_2)\, G_t G_t^\top \\
R_t &= \beta_2 R_{t-1} + (1 - \beta_2)\, G_t^\top G_t
\end{aligned}
$$

Every `inverse_steps` steps the cached eigenbases are refreshed from the
symmetric eigendecompositions $L_t = Q_L \Lambda_L Q_L^\top$ and
$R_t = Q_R \Lambda_R Q_R^\top$. The momentum is rotated into that
basis, the sign is taken element-wise, and the result is rotated back:


$$
U_t = Q_L\, \mathrm{sign}\!\left(Q_L^\top M_t\, Q_R\right) Q_R^\top
$$

The update is scaled to transfer across network width by
$\gamma_t = \gamma \cdot 2 / (m + n)$ for two-dimensional parameters
and by a constant `nonstandard_constant` otherwise, giving
$\theta_t = \theta_{t-1} - \gamma_t U_t$. Non-matrix parameters fall
back to a sign update $U_t = \mathrm{sign}(M_t)$. An exponential
moving average of the iterates is tracked so that
`eval` can swap in the averaged weights, which removes the parameter
noise that large learning rates introduce.


**Note:** Call `eval` before validation or inference to use the averaged

parameters, and `train` to restore the raw iterates before resuming
optimization.

Reference: Kevin Frans, Sergey Levine, Pieter Abbeel, "A Stable Whitening
Optimizer for Efficient Neural Network Training", 2025.
https://arxiv.org/abs/2506.07254

---
[Back to the Canon](../index.md)
