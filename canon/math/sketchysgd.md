# SketchySGD

Implements SketchySGD, a stochastic quasi-Newton method that preconditions the gradient with a randomized low-rank curvature estimate.

SketchySGD periodically builds a rank-$r$ Nyström approximation $\hat{H}$ of a subsampled Hessian using a Gaussian test matrix and $r$ Hessian-vector products, then takes preconditioned minibatch steps against the regularized inverse $(\hat{H} + \rho I)^{-1}$. The preconditioner is refreshed infrequently (once per epoch by default) and held fixed in between, so its cost amortizes over many cheap stochastic updates. Because $\hat{H} = \hat{V}\hat{\Lambda}\hat{V}^\top$ is low rank, the inverse-vector product is applied in closed form via the Woodbury identity rather than by forming or inverting a $p \times p$ matrix.

$$
\begin{aligned}
\hat{H} &= \hat{V}\hat{\Lambda}\hat{V}^\top \quad (\text{Nyström sketch of the subsampled Hessian, rank } r) \\
v_k &= \hat{V}(\hat{\Lambda} + \rho I)^{-1}\hat{V}^\top g_t + \tfrac{1}{\rho}\left(g_t - \hat{V}\hat{V}^\top g_t\right) \\
\theta_{t+1} &= \theta_t - \eta\, v_k
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the minibatch stochastic gradient, $\rho$ the regularization (Levenberg-Marquardt) parameter, $\hat{V}$ and $\hat{\Lambda}$ the orthonormal factors and eigenvalues of the rank-$r$ Nyström Hessian approximation $\hat{H}$, and $v_k = (\hat{H} + \rho I)^{-1} g_t$ the preconditioned search direction computed by the Woodbury formula. The sketch is recomputed every update interval (default once per epoch) from a Hessian minibatch; defaults are $r = 10$ and $\rho = 10^{-3}$.

Reference: Zachary Frangella, Pratik Rathore, Shipu Zhao, Madeleine Udell, "SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates", arXiv 2023. https://arxiv.org/abs/2211.08597

---
[Back to the Canon](../README.md)
