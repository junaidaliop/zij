# AutoSGD

Implements AutoSGD, automatic learning-rate selection for stochastic gradient descent.

AutoSGD removes the learning rate from SGD by treating step-size tuning as part of the optimization. Within each episode it runs three parallel SGD streams that share the same gradient noise but use a smaller, equal, and larger learning rate ($c\gamma_t$, $\gamma_t$, $C\gamma_t$). At the end of the episode a statistical decision determines whether to increase, keep, decrease, or restart the learning rate, and the corresponding stream's final iterate is carried forward. The deterministic special case (AutoGD) reduces to a line-search over the same three candidate steps.

Over an episode of length $\tau_t$ with shared noise $u_{t,k}$, the streams evolve as

$$
\begin{aligned}
\bar{x}_{t,k+1} &= \bar{x}_{t,k} - c\,\gamma_t\, g(\bar{x}_{t,k}, u_{t,k}), \\
x_{t,k+1} &= x_{t,k} - \gamma_t\, g(x_{t,k}, u_{t,k}), \\
\bar{\bar{x}}_{t,k+1} &= \bar{\bar{x}}_{t,k} - C\,\gamma_t\, g(\bar{\bar{x}}_{t,k}, u_{t,k}), \\
(x_{t+1}, \gamma_{t+1}) &= \begin{cases}
(\bar{\bar{x}}_{t,\tau_t},\; C\gamma_t), & I_t = 1 \quad (\text{increase}), \\
(x_{t,\tau_t},\; \gamma_t), & S_t = 1 \quad (\text{stay}), \\
(x_{t,\tau_t},\; c\gamma_t), & D_t = 1 \quad (\text{decrease}), \\
(x_t,\; c\gamma_t), & R_t = 1 \quad (\text{restart}),
\end{cases}
\end{aligned}
$$

where $g(x,u)$ is the stochastic gradient with $\mathbb{E}[g(x,u)] = \nabla f(x)$, $\gamma_t$ is the current learning rate, $c<1<C$ are the shrink/grow factors (default $C = 1/c = 2$, i.e. halving/doubling), and the mutually exclusive indicators $I_t, S_t, D_t, R_t \in \{0,1\}$ are set by a statistical test comparing the accumulated objective progress of the three streams.

Reference: Nikola Surjanovic, Alexandre Bouchard-Côté, Trevor Campbell, "AutoSGD: Automatic Learning Rate Selection for Stochastic Gradient Descent", arXiv 2025. https://arxiv.org/abs/2505.21651

---
[Back to the Canon](../README.md)
