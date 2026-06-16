# MaxVA

Implements MaxVA, an Adam-style optimizer that replaces the fixed second-moment decay rate with a per-coordinate coefficient chosen to maximize the observed gradient variance.

Standard adaptive methods use a constant decay $\beta_2$ to average squared gradients. MaxVA ("Maximum Variation Averaging") instead picks, at every step and for every coordinate, the averaging coefficient $\beta_t$ that maximizes the estimated variance of the gradient. A larger estimated variance produces a smaller effective step, so the optimizer slows down in noisy or sharply curved directions and moves faster where gradients are stable. The optimal $\beta_t$ has a closed form and is clipped into $[\underline{\beta}, \overline{\beta}]$ before use.

$$
\begin{aligned}
\Delta g_t &= g_t - u_{t-1}, \qquad \sigma_{t-1}^2 = v_{t-1} - u_{t-1}^2 \\
\tilde\beta_t &= \frac{\Delta g_t^2 + \sigma_{t-1}^2}{w_{t-1}\,(\Delta g_t^2 - \sigma_{t-1}^2) + \Delta g_t^2 + \sigma_{t-1}^2} \\
\beta_t &= \max\!\big(\underline{\beta},\, \min(\overline{\beta},\, \tilde\beta_t)\big) \\
\tilde m_t &= \beta_1 \tilde m_{t-1} + (1-\beta_1)\, g_t \\
\tilde u_t &= \beta_t \tilde u_{t-1} + (1-\beta_t)\, g_t \\
\tilde v_t &= \beta_t \tilde v_{t-1} + (1-\beta_t)\, g_t^2 \\
w_t &= \beta_t w_{t-1} + (1-\beta_t) \\
u_t &= \tilde u_t / w_t, \qquad v_t = \tilde v_t / w_t \\
\theta_t &= \theta_{t-1} - \eta_t \, \frac{\sqrt{w_t}}{1-\beta_1^{\,t}} \, \frac{\tilde m_t}{\sqrt{\tilde v_t} + \epsilon}
\end{aligned}
$$

where $g_t$ is the gradient, $\tilde m_t$ the first moment with fixed decay $\beta_1$ (denoted $\alpha$ in the paper), $\tilde u_t$ and $\tilde v_t$ the running mean and second moment of the gradient with adaptive decay $\beta_t$, $w_t$ the accumulated normalizer that supplies bias correction, $u_t,v_t$ their bias-corrected values, $\sigma_{t-1}^2$ the estimated gradient variance, $\eta_t$ the learning rate, and $\epsilon$ a stability constant; $\beta_t$ is clipped to $[\underline{\beta}, \overline{\beta}]$ with typical $\underline{\beta}=0.5$ and $0.98 \le \overline{\beta} \le 1$.

Reference: Chen Zhu, Yu Cheng, Zhe Gan, Furong Huang, Jingjing Liu, Tom Goldstein, "MaxVA: Fast Adaptation of Step Sizes by Maximizing Observed Variance of Gradients", arXiv 2020. https://arxiv.org/abs/2006.11918

---
[Back to the Canon](../index.md)
