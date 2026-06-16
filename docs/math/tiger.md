# Tiger

Implements Tiger, a tight-fisted sign-momentum optimizer.

Tiger keeps a single momentum buffer and forms the update from its sign,
which makes it a SignSGD variant with momentum and weight decay, and a
special case of Lion with $\beta_1 = \beta_2 = \beta$:


$$
\begin{aligned}
m_t &= \beta m_{t-1} + (1 - \beta)\, g_t \\
\theta_t &= (1 - \gamma \lambda)\, \theta_{t-1}
            - \gamma \mathrm{sign}(m_t)
\end{aligned}
$$

where $m_t$ is the momentum buffer, $\gamma$ the learning rate,
$\beta$ the momentum decay, and $\lambda$ the decoupled weight
decay. With `weight_decouple=False` the decay is instead added to the
gradient as an L2 penalty, and with `fixed_decay=True` the decoupled decay
factor is $(1 - \lambda)$ rather than $(1 - \gamma \lambda)$.

Reference: Jianlin Su, "Tiger: A Tight-fisted Optimizer", GitHub 2023.
https://github.com/bojone/tiger

---
[Back to the Canon](../index.md)
