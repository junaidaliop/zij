# M-SVAG

Implements M-SVAG, a momentum optimizer that scales each coordinate of the update by an estimate of its gradient signal-to-noise ratio.

M-SVAG decouples the two effects fused inside Adam: a sign-based direction and a per-coordinate variance adaptation. Instead of dividing by $\sqrt{v_t}$, it keeps the momentum direction $m_t$ and multiplies it by a factor $\gamma_t \in [0, 1]$ that shrinks coordinates whose stochastic gradient variance is large relative to the squared mean, leaving low-noise coordinates near full step size. The variance is estimated from the same exponential moving averages of $g_t$ and $g_t^2$, with a bias correction $\rho(\beta_1, t)$ that accounts for the correlation between the moment estimates.

$$
\begin{aligned}
m_t &= \frac{\beta_1 \tilde{m}_{t-1} + (1 - \beta_1) g_t}{1 - \beta_1^{t+1}}, \qquad
v_t = \frac{\beta_1 \tilde{v}_{t-1} + (1 - \beta_1) g_t^2}{1 - \beta_1^{t+1}} \\
\rho(\beta_1, t) &= \frac{(1 - \beta_1)\,(1 + \beta_1^{t+1})}{(1 + \beta_1)\,(1 - \beta_1^{t+1})}, \qquad
\hat{s}_t = \frac{v_t - m_t^2}{1 - \rho(\beta_1, t)} \\
\gamma_t &= \frac{m_t^2}{m_t^2 + \rho(\beta_1, t)\,\hat{s}_t} \\
\theta_{t+1} &= \theta_t - \eta\,(\gamma_t \odot m_t)
\end{aligned}
$$

where $m_t$ is the bias-corrected first moment, $v_t$ the bias-corrected second moment, $\hat{s}_t$ an unbiased estimate of the gradient variance, $\rho(\beta_1, t)$ the bias-correction term tying the two estimates together, $\gamma_t$ the per-coordinate variance-adaptation factor, $\eta$ the learning rate, $\beta_1$ the moment decay, and $\odot$ elementwise multiplication.

Reference: Lukas Balles, Philipp Hennig, "Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients", ICML 2017. https://arxiv.org/abs/1705.07774

---
[Back to the Canon](../index.md)
