# MG-ADSGD

Implements MG-ADSGD (Multi-Gossip Accelerated DSGD), a decentralized stochastic optimizer for strongly convex problems.

MG-ADSGD couples Nesterov-type primal-dual extrapolation with a multi-round fast gossip averaging (FGA) primitive. It maintains a three-sequence accelerated structure: $X$ is the primal iterate, $Y$ is the extrapolated query point at which the stochastic gradient is evaluated, and $Z$ is the auxiliary descent (momentum) variable. The gradient is mini-batched over $R$ samples and the same $R$ controls the number of accelerated gossip rounds, so increasing $R$ simultaneously reduces gradient variance and improves consensus across the network.

The fast gossip operator $\mathrm{FGA}_R$ applies a Chebyshev-type accelerated mixing recurrence $A^{(r+1)} = (1+\eta)\,W A^{(r)} - \eta A^{(r-1)}$ for $R$ rounds against the mixing matrix $W$. Per outer step $k$, on every node $i$ the stacked iterates update as

$$
\begin{aligned}
g_t &= \frac{1}{R}\sum_{s=1}^{R} \nabla F\!\left(Y^{(k)};\, \xi_{k+1,s}\right), \\
Z^{(k+1)} &= \mathrm{FGA}_R\!\left( \frac{\tfrac{\theta}{\gamma} Z^{(k)} + \mu\, Y^{(k)} - g_t}{\tfrac{\theta}{\gamma} + \mu} \right), \\
X^{(k+1)} &= \mathrm{FGA}_R\!\left( (1-\theta) X^{(k)} + \theta\, Z^{(k+1)} \right), \\
Y^{(k+1)} &= (1-\theta) X^{(k+1)} + \theta\, Z^{(k+1)},
\end{aligned}
$$

where $\gamma$ is the stepsize, $\theta = \tfrac{1}{2}\sqrt{\mu\gamma}$ is the momentum parameter, $\mu$ is the strong-convexity constant, $R$ is the shared gossip-round and mini-batch size, $g_t$ is the variance-reduced stochastic gradient at the extrapolated point $Y^{(k)}$, and $\mathrm{FGA}_R(\cdot)$ is the $R$-round fast gossip averaging operator over mixing matrix $W$.

Reference: Ming Sun, Kun Yuan, "Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization", arXiv 2026. https://arxiv.org/abs/2606.07496

---
[Back to the Canon](../README.md)
