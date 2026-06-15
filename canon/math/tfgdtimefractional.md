# TFGD (Time-fractional)

Implements TFGD (Time-Fractional Gradient Descent), gradient descent driven by a Caputo time-fractional gradient flow.

TFGD replaces the integer-order gradient flow $w_t = -\nabla J(w)$ with the fractional flow $\partial_t^\alpha w(t) = -\nabla J(w)$, $0 < \alpha < 1$, where $\partial_t^\alpha$ is the Caputo derivative. Discretizing this with the first-order Grünwald–Letnikov formula produces an update in which every past parameter state contributes through a memory term weighted by fractional coefficients, so the step combines the current gradient with the full history of displacements from the initial weights. The memory dependence is most beneficial when $\alpha$ is close to 1 (the paper reports gains around $\alpha \in [0.95, 0.99]$); $\alpha \to 1$ recovers ordinary gradient descent.

$$
\begin{aligned}
w_{k+1} &= w_0 - \eta^{\alpha}\!\left( \nabla J(w_k) + \sum_{i=0}^{k} \phi^{(\alpha)}_{k+1-i}\,(w_i - w_0) \right), \\
\phi^{(\alpha)}_n &= \frac{n-1-\alpha}{n}\,\phi^{(\alpha)}_{n-1}, \qquad \phi^{(\alpha)}_0 = 1.
\end{aligned}
$$

where $w$ are the parameters, $w_0$ the initial weights, $\eta$ the learning rate (time step), $\alpha \in (0,1)$ the fractional order, $\nabla J(w_k)$ the gradient at step $k$, and $\phi^{(\alpha)}_n = (-1)^n \binom{\alpha}{n}$ the Grünwald–Letnikov memory weights computed by the stated recurrence.

Reference: Jingyi Xie, Sirui Li, "Training Neural Networks by Time-Fractional Gradient Descent", Axioms 2022, 11(10), 507. https://doi.org/10.3390/axioms11100507

---
[Back to the Canon](../README.md)
