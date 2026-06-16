# QN-SPSA

Implements QN-SPSA, a quantum natural gradient method that approximates the Fubini-Study metric with simultaneous perturbation stochastic approximation.

Quantum natural gradient preconditions the gradient with the Fubini-Study metric tensor $g$ (one quarter of the quantum Fisher information), but evaluating $g$ exactly costs $O(d^2)$ circuit evaluations. QN-SPSA estimates both the loss gradient and the metric with simultaneous perturbation: a first-order SPSA stochastic gradient, and a second-order SPSA point estimate of $g$ built from fidelity differences along two random directions $\Delta_1,\Delta_2 \in \{-1,+1\}^d$. The raw metric estimate is averaged over iterations and regularized to stay positive definite before being inverted.

$$
\begin{aligned}
\hat g_t &= -\tfrac{1}{2}\,\frac{\delta F}{2\epsilon^2}\,\frac{\Delta_1\Delta_2^\top + \Delta_2\Delta_1^\top}{2}, \\
\delta F &= F\!\left(\theta_t,\theta_t+\epsilon\Delta_1+\epsilon\Delta_2\right) - F\!\left(\theta_t,\theta_t+\epsilon\Delta_1\right) - F\!\left(\theta_t,\theta_t-\epsilon\Delta_1+\epsilon\Delta_2\right) + F\!\left(\theta_t,\theta_t-\epsilon\Delta_1\right), \\
\bar g_t &= \frac{t}{t+1}\,\bar g_{t-1} + \frac{1}{t+1}\,\hat g_t, \\
\tilde g_t &= \sqrt{\bar g_t \bar g_t} + \beta I, \\
\hat \nabla f_t &= \frac{f(\theta_t+\epsilon\Delta) - f(\theta_t-\epsilon\Delta)}{2\epsilon}\,\Delta, \\
\theta_{t+1} &= \theta_t - \eta\, \tilde g_t^{-1}\, \hat \nabla f_t,
\end{aligned}
$$

where $f$ is the loss, $F(\theta,\theta') = |\langle\psi(\theta)|\psi(\theta')\rangle|^2$ is the state fidelity, $\Delta,\Delta_1,\Delta_2$ are random $\pm 1$ perturbation vectors, $\epsilon$ is the perturbation size, $\eta$ is the learning rate, $\bar g_t$ is the running-averaged metric, and $\beta>0$ regularizes the matrix square root to keep $\tilde g_t$ positive definite and invertible.

Reference: Julien Gacon, Christa Zoufal, Giuseppe Carleo, Stefan Woerner, "Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information", Quantum 2021. https://quantum-journal.org/papers/q-2021-10-20-567/

---
[Back to the Canon](../index.md)
