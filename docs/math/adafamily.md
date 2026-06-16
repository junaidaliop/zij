# AdaFamily

Implements AdaFamily, a parameterized family of Adam-like adaptive methods that interpolates between Adam, AdaBelief, and AdaMomentum.

A single hyperparameter $\mu \in [0, 1]$ selects how the second moment is formed. The squared quantity it tracks is a convex blend of the raw gradient $g_t$ and the first moment $m_t$, scaled by a normalization factor $c$. At $\mu = 0$ it recovers Adam (variance of $g_t$), at $\mu = 0.5$ AdaBelief (variance of $g_t - m_t$), and at $\mu = 1$ AdaMomentum (second moment of $m_t$); intermediate values yield a continuum of optimizers.

$$
\begin{aligned}
c &= 2\,(1 - |\mu - 0.5|) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\big(c\,((1 - \mu) g_t - \mu\, m_t)\big)^2 + \epsilon \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected values, $\epsilon$ a stability constant, and $\mu \in [0,1]$ the family hyperparameter with $c$ its normalization factor.

Reference: Hannes Fassold, "AdaFamily: A family of Adam-like adaptive gradient methods", arXiv 2022. https://arxiv.org/abs/2203.01603

---
[Back to the Canon](../index.md)
