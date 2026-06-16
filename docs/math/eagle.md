# EAGLE

Implements EAGLE, a learning-rate-free optimizer that estimates a step from the secant ratio of consecutive parameter and gradient changes.

EAGLE (Early Approximated Gradient based Learning rate Estimator) treats the per-coordinate ratio of the change in parameters to the change in gradients as a finite-difference approximation of the inverse Hessian along the optimization trajectory, giving a secant-style approximation of Newton's method that needs no explicit learning rate. For a locally quadratic loss this places the next iterate at the minimizer in a single step.

Because the ratio diverges when consecutive gradients are nearly equal, EAGLE pairs the rule with an adaptive switching mechanism: when the gradient difference falls below a threshold $\tau_t$ (or the local curvature is upward-opening) it falls back to an Adam step, otherwise it uses the EAGLE step.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta \cdot \frac{\theta_t - \theta_{t-1}}{g_t - g_{t-1}} \cdot g_t \\
\tau_t &= \min\!\left(\max\!\left(\alpha \cdot \frac{\sigma_g}{\mu_g + \epsilon},\; \tau_{\min}\right),\; \tau_{\max}\right) \\
\theta_{t+1} &= \begin{cases} \mathrm{Adam}(\theta_t) & \text{if } |g_t - g_{t-1}| < \tau_t \\ \theta_t - \eta \cdot \dfrac{\theta_t - \theta_{t-1}}{g_t - g_{t-1}} \cdot g_t & \text{otherwise} \end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate (set to 1 in the idealized analysis), $g_t$ the gradient, all ratios taken element-wise; $\tau_t$ is the dynamic threshold, $\mu_g,\sigma_g$ the mean and standard deviation of the gradient norms over the last 10 steps, $\sigma_g/\mu_g$ their coefficient of variation, $\alpha$ a scaling factor ($5\times10^{-3}$), $\tau_{\min},\tau_{\max}$ its bounds ($10^{-5},10^{-2}$), and $\epsilon$ a stability constant.

Reference: Takumi Fujimoto, Hiroaki Nishi, "EAGLE: Early Approximated Gradient-based Learning-rate Estimator", arXiv 2025. https://arxiv.org/abs/2502.01036

---
[Back to the Canon](../index.md)
