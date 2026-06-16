# QIASO

Implements QIASO (Quantum-Inspired Adaptive Superposition Optimization), a gradient-free optimizer that maintains a probability distribution over a fixed set of candidate weight vectors and reshapes it toward low-loss candidates.

Rather than tracking a single parameter vector, QIASO keeps $K$ candidates $\{\theta_1,\dots,\theta_K\}$ together with an amplitude (probability) vector $p_t$ on the simplex. Each iteration solves a KL-regularized variational problem that trades expected candidate loss against proximity to the previous distribution; its closed-form solution is the exponential-weights (Gibbs / softmax) reweighting of the amplitudes. To escape poor local minima, candidate weights are occasionally perturbed by a quantum-tunneling-inspired Gaussian operator whose firing probability decays exponentially over training.

$$
\begin{aligned}
p_{t+1} &= \arg\min_{q \in \Delta_K}\ \Big\{ \langle q, L\rangle + \tfrac{1}{\eta}\, D_{\mathrm{KL}}(q \,\|\, p_t) \Big\}, \\
p_{t+1}(k) &= \frac{p_t(k)\,\exp(-\eta\,\ell_k)}{\sum_{j=1}^{K} p_t(j)\,\exp(-\eta\,\ell_j)}, \qquad \ell_k = L(\theta_k), \\
\theta_k &\leftarrow \theta_k + \epsilon\,\zeta\,, \quad \zeta \sim \mathcal{N}(0,\sigma^2 I), \quad \text{with probability } \rho_t = \rho_0\, e^{-\lambda t}.
\end{aligned}
$$

where $\theta_k$ are the candidate weight vectors, $p_t$ the amplitude (probability) vector on the simplex $\Delta_K$, $\ell_k = L(\theta_k)$ the loss of candidate $k$, $\eta>0$ the inverse-temperature step parameter, $D_{\mathrm{KL}}$ the Kullback-Leibler divergence, $\epsilon$ the tunneling perturbation scale, $\sigma^2$ its Gaussian variance, and $\rho_0, \lambda$ the initial perturbation probability and its decay rate.

Reference: Irsa Sajjad, Mashail M. AL Sobhi, "The quantum-inspired adaptive superposition optimization for neural network training", AIMS Mathematics 2026. https://www.aimspress.com/article/doi/10.3934/math.2026010

---
[Back to the Canon](../index.md)
