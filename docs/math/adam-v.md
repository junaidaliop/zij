# Adam+

Implements Adam+, a stochastic method that adds adaptive variance reduction on top of Adam-style momentum.

Adam+ keeps a single moving average $z_t$ of the gradient (a first-moment estimate) but evaluates the gradient at an extrapolated point $\hat\theta_{t+1}$ rather than the current iterate. This extrapolation acts as a variance-reduction mechanism, while the step size is normalized by the square root of the norm of $z_t$, replacing Adam's coordinate-wise second moment with a single adaptive scale.

$$
\begin{aligned}
\gamma_t &= \frac{\alpha\,\beta^{a}}{\max\!\left(\lVert z_t\rVert^{1/2},\,\epsilon\right)} \\
\theta_{t+1} &= \theta_t - \gamma_t\, z_t \\
\hat\theta_{t+1} &= \left(1 - \tfrac{1}{\beta}\right)\theta_t + \tfrac{1}{\beta}\,\theta_{t+1} \\
z_{t+1} &= (1-\beta)\, z_t + \beta\, g_{t+1}(\hat\theta_{t+1})
\end{aligned}
$$

where $\theta$ are the parameters, $z_t$ is the first-moment estimate (initialized as $z_0 = g_0(\theta_0)$), $g_{t+1}(\hat\theta_{t+1})$ is the stochastic gradient evaluated at the extrapolated point, $\alpha \ge 1$ is the step-size parameter, $\beta \in (0,1)$ is the moment decay rate, $a \ge 1$ is the exponent applied to $\beta$, and $\epsilon$ is a small stability constant (defaults $\alpha=0.1$, $a=1$, $\beta=0.1$, $\epsilon=10^{-8}$).

Reference: Mingrui Liu, Wei Zhang, Francesco Orabona, Tianbao Yang, "Adam+: A Stochastic Method with Adaptive Variance Reduction", arXiv 2020. https://arxiv.org/abs/2011.11985

---
[Back to the Canon](../index.md)
