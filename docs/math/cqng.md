# CQNG

Implements CQNG, a conjugate-gradient extension of quantum natural gradient with per-step line-searched hyperparameters.

Quantum natural gradient (QNG) preconditions the gradient of a variational cost $L(\theta)$ with the inverse Fubini-Study metric of the parametrized quantum state, giving updates that follow the geometry of state space rather than parameter space. CQNG (Modified Conjugate Quantum Natural Gradient) adds a conjugate-gradient memory term: the search direction blends the current preconditioned gradient with the previous direction, and instead of fixing the step size and conjugate coefficient, both are chosen at each iteration by minimizing the cost along the resulting direction. The label "Modified" reflects that the exact conjugacy condition $d_t^{\top} F\, d_{t-1} = 0$ is not strictly enforced; the dynamic choice of coefficients takes its place.

$$
\begin{aligned}
d_t &= \begin{cases} -F^{ij}\,\partial_j L(\theta_t) & t = 0 \\ -F^{ij}\,\partial_j L(\theta_t) + \beta_t\, d_{t-1} & t \ge 1 \end{cases} \\
(\alpha_t, \beta_t) &= \arg\min_{\alpha,\beta}\; L\!\left(\theta_t - \alpha\, F^{ij}\,\partial_j L(\theta_t) + \beta\, d_{t-1}\right) \\
\theta_{t+1} &= \theta_t + \alpha_t\, d_t
\end{aligned}
$$

where $\theta$ are the variational parameters, $L$ the cost, $\partial_j L$ the gradient, $F^{ij}$ the inverse of the Fubini-Study metric $F_{ij} = \mathrm{Re}\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle$, $d_t$ the search direction, $\alpha_t$ the step size, and $\beta_t$ the conjugate coefficient, the latter two found jointly by line search at each step.

Reference: Mourad Halla, "Modified Conjugate Quantum Natural Gradient", arXiv 2025. https://arxiv.org/abs/2501.05847

---
[Back to the Canon](../index.md)
