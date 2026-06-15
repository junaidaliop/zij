# PSGD

Implements PSGD, preconditioned stochastic gradient descent that learns a preconditioner by matching curvature from gradient and parameter perturbations.

PSGD scales the gradient by a positive-definite preconditioner $P$, like a quasi-Newton method, but estimates $P$ online from random perturbation pairs instead of from a Hessian. Writing $P = Q^\top Q$ with $Q$ upper triangular keeps $P$ positive definite, and $Q$ is updated by a multiplicative (relative) gradient step on the Lie group of triangular matrices. The preconditioner minimizes the criterion $c(P) = \mathbb{E}[\,\delta g^\top P\, \delta g + \delta\theta^\top P^{-1} \delta\theta\,]$, whose optimum balances the perturbations of the preconditioned gradient against those of the parameters, yielding $P R_g P = R_\theta$ with $R_g = \mathbb{E}[\delta g\, \delta g^\top]$ and $R_\theta = \mathbb{E}[\delta\theta\, \delta\theta^\top]$.

$$
\begin{aligned}
\nabla_{\mathcal{E}} &= 2\,\mathrm{triu}\!\left(Q\, \delta g\, \delta g^\top Q^\top - Q^{-\top} \delta\theta\, \delta\theta^\top Q^{-1}\right) \\
Q &\leftarrow Q - \frac{\mu_0}{\max|\nabla_{\mathcal{E}}|}\, \nabla_{\mathcal{E}}\, Q \\
\theta_{t+1} &= \theta_t - \mu\, Q^\top Q\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the stochastic gradient, $P = Q^\top Q$ the preconditioner with $Q$ upper triangular, $\delta\theta$ a small parameter perturbation and $\delta g$ the resulting gradient perturbation, $\mathrm{triu}(\cdot)$ the upper-triangular part, $\mu > 0$ the step size, and $0 < \mu_0 < 1$ the normalized step size for the preconditioner update.

Reference: Xi-Lin Li, "Preconditioned Stochastic Gradient Descent", arXiv 2015. https://arxiv.org/abs/1512.04202

---
[Back to the Canon](../README.md)
