# DP-LSSGD

Implements DP-LSSGD, a differentially private SGD variant that denoises the privatized gradient with a Laplacian smoothing operator.

Standard differentially private SGD privatizes each step by adding isotropic Gaussian noise to the minibatch gradient, which degrades utility. DP-LSSGD keeps the same noise injection (so the privacy guarantee is unchanged) but, as a post-processing step, multiplies the noisy gradient by the inverse of the Laplacian smoothing matrix $A_\sigma = I - \sigma L$. Because $A_\sigma^{-1}$ acts as a low-pass filter on the coordinates, it suppresses the high-frequency component of the injected noise on the fly, lifting the utility of privacy-preserving empirical risk minimization without weakening the differential privacy.

$$
\begin{aligned}
A_\sigma &= I - \sigma L, \\
g_t &= \frac{1}{b}\sum_{i \in B_t} \nabla f_i(\theta_t) + z_t, \qquad z_t \sim \mathcal{N}(0, \nu^2 I), \\
\theta_{t+1} &= \theta_t - \eta\, A_\sigma^{-1} g_t.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta$ the learning rate, $B_t$ a minibatch of size $b$, $z_t$ the per-step Gaussian privacy noise with variance $\nu^2$ per coordinate, $\sigma \ge 0$ the smoothing strength, $L$ the discrete one-dimensional Laplacian with periodic boundary conditions (so $A_\sigma$ is tridiagonal with $1+2\sigma$ on the diagonal, $-\sigma$ on the off-diagonals and corners), and $A_\sigma^{-1}$ the resulting low-pass smoothing operator. Setting $\sigma = 0$ recovers ordinary DP-SGD.

Reference: Bao Wang, Quanquan Gu, March Boedihardjo, Farzin Barekat, Stanley J. Osher, "DP-LSSGD: A Stochastic Optimization Method to Lift the Utility in Privacy-Preserving ERM", arXiv 2019. https://arxiv.org/abs/1906.12056

---
[Back to the Canon](../index.md)
