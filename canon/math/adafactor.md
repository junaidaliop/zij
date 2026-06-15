# Adafactor

Implements Adafactor, an adaptive method that stores only row and column
statistics of the squared gradients, giving sublinear memory cost.

For a matrix parameter with gradient $G_t$, Adafactor avoids the full
second-moment matrix of Adam by keeping two vectors: a running average of
the per-row sums $R_t$ and the per-column sums $C_t$ of $G_t^2$. The
second moment is reconstructed as the rank-one outer product
$R_t C_t / (\boldsymbol{1}^\top R_t)$, which is the best nonnegative rank-one
approximation under the generalized Kullback-Leibler divergence. The decay
rate increases over time as $\hat{\beta}_{2,t} = 1 - t^{-c}$, removing the
need for bias correction. The unscaled update is then clipped so its
root-mean-square does not exceed a threshold $d$, and is scaled by a
relative step size proportional to the magnitude of the parameters
themselves, so the step adapts to each tensor's scale.

$$
\begin{aligned}
\hat{\beta}_{2,t} &= 1 - t^{-c} \\
R_t &= \hat{\beta}_{2,t}\, R_{t-1} + (1 - \hat{\beta}_{2,t})\,(G_t^2 + \epsilon_1 \boldsymbol{1}\boldsymbol{1}^\top)\,\boldsymbol{1} \\
C_t &= \hat{\beta}_{2,t}\, C_{t-1} + (1 - \hat{\beta}_{2,t})\,\boldsymbol{1}^\top (G_t^2 + \epsilon_1 \boldsymbol{1}\boldsymbol{1}^\top) \\
\hat{V}_t &= \frac{R_t\, C_t}{\boldsymbol{1}^\top R_t} \\
U_t &= \frac{G_t}{\sqrt{\hat{V}_t}}, \qquad
\hat{U}_t = \frac{U_t}{\max\!\left(1,\ \mathrm{RMS}(U_t)/d\right)} \\
\alpha_t &= \max\!\left(\epsilon_2,\ \mathrm{RMS}(\theta_{t-1})\right)\, \rho_t \\
\theta_t &= \theta_{t-1} - \alpha_t\, \hat{U}_t
\end{aligned}
$$

where $\theta$ are the parameters, $G_t$ is the gradient, $R_t$ and $C_t$
are the row and column second-moment accumulators, $\hat{V}_t$ is the
factored second-moment estimate, $\hat{\beta}_{2,t}$ is the increasing
decay rate, $\boldsymbol{1}$ is the all-ones vector, $\mathrm{RMS}(\cdot)$ is
the root-mean-square over all entries, $d$ is the update-clipping
threshold, $\rho_t$ is the relative step size, $\alpha_t$ is the effective
step size, and $\epsilon_1, \epsilon_2$ are small constants for numerical
stability.

Reference: Noam Shazeer and Mitchell Stern, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost", ICML 2018.
https://arxiv.org/abs/1804.04235

---
[Back to the Canon](../README.md)
