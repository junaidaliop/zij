# QUIVER

Implements QUIVER, an adaptive forward-gradient optimizer for parameterized quantum circuits.

QUIVER avoids the parameter-shift rule by estimating the gradient from a tunable number $V$ of random directional derivatives, each measured with a finite-difference pair of $M$-shot circuit evaluations. Averaging $V$ such directional probes gives an unbiased forward-gradient estimator, and the descent step uses this estimator in place of the exact gradient.

The distinctive part is the measurement budget: rather than fixing the number of directions and shots, QUIVER derives a closed-form minimum-cost allocation for the shots $M_\ell$ assigned to each direction, balancing measurement variance against the contribution that direction makes to the descent gain. Rademacher directions ($\kappa = 1$) minimize the estimator's second moment.

$$
\begin{aligned}
\tilde\nabla_{v}^{\,\epsilon} f &= \frac{f^{M}(\theta + \epsilon v) - f^{M}(\theta - \epsilon v)}{2\epsilon}, \\
\tilde\nabla^{F} f(\theta_t) &= \frac{1}{V}\sum_{\ell=1}^{V} \big(\tilde\nabla_{v^{\ell}}^{\,\epsilon} f\big)\, v^{\ell}, \\
M_\ell^{*} &= \frac{2\,\mathrm{Var}_m\!\big[\tilde\nabla_{v^{\ell}} \mathcal{L}_m\big]}{C\,\lVert \nabla\mathcal{L}\rVert^{2} - (\nabla_{v^{\ell}}\mathcal{L})^{2}}, \qquad C = \frac{2V}{L\,\eta\,(N + V + \kappa - 2)}, \\
\theta_{t+1} &= \theta_t - \eta\, \tilde\nabla^{F} f(\theta_t).
\end{aligned}
$$

where $\theta$ are the circuit parameters, $\eta$ the learning rate, $f^{M}(\cdot)$ the $M$-shot sample-mean of the loss, $\epsilon$ the finite-difference step, $v^{\ell}$ random unit directions, $V$ the number of directions per step, $M_\ell^{*}$ the optimal shots for direction $\ell$, $N$ the number of parameters, $L$ the loss smoothness constant, $\kappa$ the fourth-moment constant of the direction distribution ($\kappa=1$ for Rademacher, $\kappa=3$ for Gaussian), $\mathcal{L}_m$ the single-shot loss, and $\mathrm{Var}_m$ the per-shot measurement variance.

Reference: Brian Coyle, Snehal Raj, Virag Umathe, El Amine Cherrat, Elham Kashefi, "Adaptive directional gradients for parameterised quantum circuits", arXiv 2026. https://arxiv.org/abs/2606.09734

---
[Back to the Canon](../index.md)
