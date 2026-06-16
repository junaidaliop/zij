# SGD with adaptive preconditioning

Implements SGD with Adaptive Preconditioning, a unified AdaGrad-type preconditioned stochastic gradient method.

The method takes a preconditioned mirror-descent step: instead of subtracting a scaled gradient, it minimizes the linearized objective plus a quadratic proximal term weighted by the inverse preconditioner $H_k^{-1}$. The preconditioner itself is defined as the solution of an auxiliary optimization problem over a structured subspace $\mathcal{H}$ of self-adjoint operators (multiples of identity, diagonal, or one-sided matrix operators), which recovers AdaGrad-Norm, AdaGrad, and ASGO/One-sided Shampoo as special cases depending on the choice of $\mathcal{H}$.

With the potential $\phi(h) = \delta h + \eta^2/h$, the preconditioner has a closed form: it is an inverse square root of the running sum of gradient outer products projected onto $\mathcal{H}$, generalizing the AdaGrad accumulator. An optional Nesterov coupling (line $\bar\theta_{k+1}$ below, with $\alpha_k = 2/(k+2)$) accelerates convergence and explains why diagonal preconditioning and momentum combine well in Adam.

$$
\begin{aligned}
S_k &= \sum_{i=0}^{k} g_i \langle g_i, \cdot \rangle \\
H_k &= \eta\,\big(\delta I + \mathrm{proj}_{\mathcal{H}}(S_k)\big)^{-1/2} \\
\theta_{k+1} &= \arg\min_{\theta}\; \langle g_k, \theta \rangle + \tfrac{1}{2}\,\lVert \theta - \theta_k \rVert^2_{H_k^{-1}} \;=\; \theta_k - H_k\, g_k \\
\bar\theta_{k+1} &= \alpha_k\, \theta_{k+1} + (1 - \alpha_k)\, \bar\theta_k
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_k$ the (stochastic) gradient, $H_k$ the positive-definite preconditioner, $S_k$ the accumulated outer-product operator, $\mathrm{proj}_{\mathcal{H}}$ the projection onto the chosen preconditioner subspace $\mathcal{H}$, $\delta>0$ a stability constant, $I$ the identity, and $\alpha_k$ the Nesterov coupling coefficient. The last line is used only in the accelerated variant; for plain Adaptive SGD it is omitted.

Reference: Dmitry Kovalev, "SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration", arXiv 2025. https://arxiv.org/abs/2506.23803

---
[Back to the Canon](../index.md)
