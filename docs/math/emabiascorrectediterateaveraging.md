# EMA bias-corrected iterate averaging

Implements BEMA, a bias-corrected exponential moving average of the iterate trajectory.

Plain EMA of the parameters smooths the optimization trajectory but lags behind the current iterate, since recent updates are underweighted. BEMA adds a debiasing term proportional to the displacement $\theta_t - \theta_0$ from a periodically refreshed snapshot, which cancels the lag of the EMA estimate. Both the EMA decay $\beta_t$ and the correction weight $\alpha_t$ anneal with the step count, mirroring the maximum-likelihood estimator in the quadratic setting. The snapshot $\theta_0$ and both averages are reset throughout a burn-in of $\tau$ steps, after which BEMA is refreshed every $\phi$ steps.

$$
\begin{aligned}
\alpha_t &= (\rho + \gamma t)^{-\eta}, \qquad \beta_t = (\rho + \gamma t)^{-\kappa} \\
\hat{\mu}^{\mathrm{EMA}}_t &= (1 - \beta_t)\,\hat{\mu}^{\mathrm{EMA}}_{t-1} + \beta_t\,\theta_t \\
\hat{\mu}_t &= \alpha_t\,(\theta_t - \theta_0) + \hat{\mu}^{\mathrm{EMA}}_t
\end{aligned}
$$

where $\theta_t$ is the optimizer's iterate at step $t$, $\theta_0$ is the most recent snapshot (refreshed during burn-in), $\hat{\mu}^{\mathrm{EMA}}_t$ is the running EMA, $\hat{\mu}_t$ is the bias-corrected average returned, $\kappa$ is the EMA power, $\eta$ is the bias power, $\gamma$ is the multiplier, and $\rho$ is the lag offset; defaults are $\kappa = 0.5$, $\eta = 0.2$, $\gamma = 1.0$, $\rho = 10$.

Reference: Adam Block, Cyril Zhang, "EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes", arXiv 2025. https://arxiv.org/abs/2508.00180

---
[Back to the Canon](../index.md)
