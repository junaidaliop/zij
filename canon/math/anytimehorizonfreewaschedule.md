# Anytime (Horizon-Free WA schedule)

Implements the Anytime (Horizon-Free WA) schedule, a learning-rate schedule whose decay does not depend on the total training horizon, paired with weight averaging.

Most pretraining recipes use horizon-dependent schedules (e.g. cosine) that must be tuned to a fixed step budget. The anytime schedule instead decays the step size polynomially in $t$ alone and recovers the loss of a well-tuned cosine schedule at every point along training, with weight averaging (model merging) supplying the variance reduction that ordinarily comes from annealing to zero. The base learning rate is constrained by $\eta \lesssim 1/\mathrm{Tr}(H)$, where $H$ is the data covariance; the optimal decay exponent is $\gamma^\star = \max\{1 - a/b,\, 0\}$, set by the source exponent $a$ and capacity exponent $b$.

$$
\begin{aligned}
\eta_t &= \frac{\eta}{t^{\gamma}}, \qquad 0 < \gamma < 1, \\
\tau_t &= \left(\tfrac{1}{2}\right)^{f/t}, \\
\bar{w}_{t+1} &= (1 - \tau_t)\,\bar{w}_t + \tau_t\,\theta_t.
\end{aligned}
$$

where $\theta_t$ are the SGD iterates, $\eta_t$ is the step size at iteration $t$, $\eta$ is a horizon-independent base learning rate, $\gamma$ is the polynomial decay exponent, $\bar{w}_t$ is the averaged (merged) weight, and $\tau_t$ is the EMA coefficient set so the half-life equals a fraction $f$ of the current time $t$. A practical $1/\sqrt{t}$ variant uses $\eta_t = \eta\sqrt{\alpha/(t+\alpha)}$ with tunable $\alpha > 0$.

Reference: Alexandru Meterez, Pranav Ajit Nair, Depen Morwani, Cengiz Pehlevan, Sham Kakade, "Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging", arXiv 2026. https://arxiv.org/abs/2602.03702

---
[Back to the Canon](../README.md)
