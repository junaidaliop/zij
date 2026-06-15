# CLion

Implements CLion, a cautious variant of Lion that falls back to the identity function when the sign-update direction has small entries.

Lion forms a sign-based update direction $c_t$ by interpolating momentum and the current gradient, then steps along $\mathrm{sign}(c_t)$. CLion keeps this structure but applies the sign only when every nonzero coordinate of $c_t$ is large enough; if the smallest nonzero magnitude falls below a threshold $\nu$, it uses $c_t$ unchanged. This guards against the gradient-explosion behavior that pure sign updates can induce and yields a tighter generalization bound.

$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
\theta_t &= \theta_{t-1} - \eta\big(h(c_t) + \lambda \theta_{t-1}\big), \qquad
h(c_t) =
\begin{cases}
\mathrm{sign}(c_t), & \min_{j \in S_t} |(c_t)_j| \ge \nu, \\
c_t, & \text{otherwise},
\end{cases} \\
m_t &= \beta_2 m_{t-1} + (1 - \beta_2) g_t,
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $\beta_1,\beta_2 \in (0,1)$ the decay rates, $\lambda$ the decoupled weight decay, $\nu > 0$ the magnitude threshold, and $S_t = \{\, j : (c_t)_j \ne 0 \,\}$ the set of nonzero coordinates of $c_t$.

Reference: Feihu Huang, Guanyi Zhang, Songcan Chen, "CLion: Efficient Cautious Lion Optimizer with Enhanced Generalization", arXiv 2026. https://arxiv.org/abs/2604.14587

---
[Back to the Canon](../README.md)
