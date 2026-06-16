# DAT-SGD

Implements DAT-SGD (Decentralized Anytime SGD), a decentralized SGD that decouples gradient queries from iterates via anytime averaging.

Each of the $M$ machines keeps two sequences: an iterate $w_t^i$ updated by a weighted local SGD step, and a query point $x_t^i$ formed as a running weighted average of the iterates at which gradients are evaluated. After the local updates, both sequences are mixed with neighbors through a doubly stochastic gossip matrix $P$. Using weights $\alpha_t = t$, the anytime averaging removes the usual sequential dependence of the query point on the latest iterate, which lets the gradient computations across rounds be parallelized.

$$
\begin{aligned}
g_t^i &= \nabla f_i(x_t^i, z_t^i), \\
w_{t+1/2}^i &= w_t^i - \eta\,\alpha_t\,g_t^i, \\
x_{t+1/2}^i &= \frac{\alpha_{1:t-1}}{\alpha_{1:t}}\,x_t^i + \frac{\alpha_t}{\alpha_{1:t}}\,w_{t+1/2}^i, \\
w_{t+1}^i &= \sum_{j=1}^{M} P_{ij}\,w_{t+1/2}^j, \\
x_{t+1}^i &= \sum_{j=1}^{M} P_{ij}\,x_{t+1/2}^j,
\end{aligned}
$$

where $\eta$ is the learning rate, $\alpha_t$ is a nonnegative weight sequence (the paper uses $\alpha_t = t$), $\alpha_{1:t} = \sum_{\tau=1}^{t}\alpha_\tau$ is the cumulative weight, $z_t^i \sim \mathcal{D}_i$ is the sample drawn at machine $i$, $g_t^i$ is the stochastic gradient evaluated at the query point $x_t^i$, and $P$ is a symmetric doubly stochastic gossip matrix with entries $P_{ij}$.

Reference: Ofri Eisen, Ron Dorfman, Kfir Y. Levy, "Enhancing Parallelism in Decentralized Stochastic Convex Optimization", ICML 2025. https://arxiv.org/abs/2506.00961

---
[Back to the Canon](../index.md)
