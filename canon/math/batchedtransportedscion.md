# Batched / Transported Scion

Implements Batched / Transported Scion, scale-invariant stochastic conditional-gradient optimizers that take steps inside a norm ball via a linear minimization oracle.

Scion replaces the usual descent direction with the linear minimization oracle (LMO) of a momentum-averaged gradient over the unit ball of a chosen norm (typically the spectral norm on weight matrices). Because the LMO depends only on the direction of its argument, the resulting step is scale-invariant. The *batched* variant averages $B$ stochastic gradients per step before feeding them into the momentum buffer, which controls the heavy-tailed noise that makes plain stochastic LMO steps unstable.

The *transported* variant evaluates the gradient not at the current iterate but at an extrapolated point $Y_t$ obtained by carrying forward the previous step, in the spirit of a momentum/extragradient correction. This exploits Hessian smoothness for a better complexity rate while keeping the same LMO-based update on the primary sequence $\theta_t$.

$$
\begin{aligned}
Y_t &= \theta_t + \frac{\beta_t}{1-\beta_t}\,(\theta_t - \theta_{t-1}) \quad \text{(transported only; else } Y_t = \theta_t), \\
\bar{g}_t &= \frac{1}{B}\sum_{i=1}^{B} g(Y_t, \xi_t^i), \\
m_{t+1} &= \beta_t\, m_t + (1-\beta_t)\,\bar{g}_t, \\
\theta_{t+1} &= \theta_t + \eta_t\,\mathrm{lmo}(m_{t+1}), \\
\mathrm{lmo}(S) &\in \arg\min_{\|X\|\le 1}\ \langle S, X\rangle.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the stepsize, $\beta_t$ the momentum coefficient, $g(\cdot,\xi)$ a stochastic gradient, $\bar{g}_t$ its average over a batch of $B$ samples, $m_t$ the momentum buffer, $\|\cdot\|$ the chosen norm (spectral norm for matrices) with $\mathrm{lmo}$ its linear minimization oracle over the unit ball, and $Y_t$ the extrapolated (transported) evaluation point. Setting $Y_t=\theta_t$ recovers the batched Scion method.

Reference: Jiayu Zhang, Tianyi Lin, "Scale-Invariant Neural Network Optimization: Norm Geometry and Heavy-Tailed Noise", arXiv 2026. https://arxiv.org/abs/2605.18528

---
[Back to the Canon](../README.md)
