# LORENZA

Implements LORENZA, a memory-efficient low-rank optimizer that combines GaLore-style subspace training with a zeroth-order sharpness-aware perturbation.

LORENZA trains in a low-rank gradient subspace, like GaLore, but replaces the costly SVD that selects the subspace with a randomized range finder (SSRF): one Gaussian sketch of the gradient followed by a QR factorization yields the projection basis $Q_t$ in $O(mnr)$ time. Gradients are projected into this subspace, an Adam update runs there, and the step is projected back into full space. The subspace is recomputed every $T$ steps.

On top of this, LORENZA adds a sharpness-aware (SAM) ascent step whose perturbation is estimated with cheap forward-only finite differences inside the same low-rank subspace, avoiding the extra full backward pass that standard SAM requires. The perturbation directions $P_j$ live in the column space of $Q_t$, the directional derivative is estimated by symmetric finite differences, and the resulting ascent direction $G_t^{\mathrm{Pert}}$ is normalized and used to move the weights before the real gradient $G_t^{\mathrm{SAM}}$ is taken.

$$
\begin{aligned}
Q_t &= \mathrm{QR}(g_t\,\Omega), \qquad \Omega \sim \mathcal{N}(0,\tfrac{1}{r})^{n\times r} \quad (\text{every } T \text{ steps}) \\
P_j &= Q_t\,\mathrm{diag}(u_j)\,Q_t^{\top}, \qquad u_j \sim \mathcal{N}(0, I_r) \\
G_t^{\mathrm{Pert}} &= -\frac{1}{q}\sum_{j=1}^{q}\frac{f(\theta_t + \mu P_j) - f(\theta_t - \mu P_j)}{2\mu}\,P_j \\
G_t^{\mathrm{SAM}} &= \nabla f\!\left(\theta_t + \rho\,\frac{G_t^{\mathrm{Pert}}}{\lVert G_t^{\mathrm{Pert}} \rVert_F}\right) \\
\hat g_t &= Q_t^{\top} G_t^{\mathrm{SAM}} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\hat g_t, \qquad \hat m_t = \frac{m_t}{1-\beta_1^{\,t}} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\hat g_t^{\,2}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \eta\, Q_t\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters (a weight matrix $W$), $g_t$ the gradient, $\eta$ the learning rate, $Q_t \in \mathbb{R}^{m\times r}$ the orthonormal low-rank projection basis from the randomized range finder, $\Omega$ a Gaussian sketch, $r$ the target rank, $P_j$ low-rank perturbation directions drawn inside the subspace, $\mu$ the zeroth-order smoothing radius, $q$ the number of finite-difference samples, $G_t^{\mathrm{Pert}}$ the estimated sharpness ascent direction, $\rho$ the SAM perturbation radius, $G_t^{\mathrm{SAM}}$ the gradient at the perturbed point, $\hat g_t$ the projected gradient, $m_t/v_t$ the first/second moments with bias corrections $\hat m_t/\hat v_t$, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ a small stability constant. The subspace $Q_t$ is refreshed every $T$ steps.

Reference: Yehonathan Refael, Iftach Arbel, Ofir Lindenbaum, Tom Tirer, "LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training and Fine-Tuning via Efficient Zeroth-Order Adaptive SAM Optimization", ICML 2025. https://arxiv.org/abs/2502.19571

---
[Back to the Canon](../index.md)
