# LT-ADMM

Implements LT-ADMM, a communication-efficient distributed ADMM that lets each agent run several local training steps between communication rounds.

Each agent $i$ minimizes its local loss $f_i$ subject to consensus with its neighbors $\mathcal{N}_i$. Instead of solving the ADMM proximal step exactly, every round $k$ runs $\tau$ inner gradient steps on a local variable $\phi_{i,k}^t$ that approximates the proximal update; the result becomes $x_{i,k+1}$. Agents then exchange the auxiliary variables $z_{ij}$ and update them, so communication happens only once per $\tau$ local steps. The gradient $g_i$ can be a plain mini-batch estimate or a SAGA-style variance-reduced estimate using a per-sample memory table $r_{i,h,k}^t$.

$$
\begin{aligned}
\phi_{i,k}^0 &= x_{i,k}, \\
\phi_{i,k}^{t+1} &= \phi_{i,k}^t - \gamma\left(g_i(\phi_{i,k}^t) + \rho\,|\mathcal{N}_i|\,\phi_{i,k}^t - \sum_{j\in\mathcal{N}_i} z_{ij,k}\right), \quad t=0,\dots,\tau-1, \\
x_{i,k+1} &= \phi_{i,k}^\tau, \\
g_i(\phi_{i,k}^t) &= \frac{1}{|\mathcal{B}_i|}\sum_{h\in\mathcal{B}_i}\left(\nabla f_{i,h}(\phi_{i,k}^t) - \nabla f_{i,h}(r_{i,h,k}^t)\right) + \frac{1}{m_i}\sum_{h=1}^{m_i}\nabla f_{i,h}(r_{i,h,k}^t), \\
z_{ij,k+1} &= \tfrac{1}{2}\left(z_{ij,k} - z_{ji,k} - 2\rho\,x_{j,k+1}\right).
\end{aligned}
$$

where $x_{i,k}$ is agent $i$'s model, $\phi_{i,k}^t$ the local iterate over $\tau$ inner steps, $\gamma$ the step size, $\rho$ the ADMM penalty, $\mathcal{N}_i$ the neighbor set with degree $|\mathcal{N}_i|$, $z_{ij}$ the auxiliary (edge) variables exchanged between neighbors, $\mathcal{B}_i$ a sampled mini-batch over the $m_i$ local samples, and $r_{i,h,k}^t$ the SAGA memory holding the iterate at which $\nabla f_{i,h}$ was last evaluated.

Reference: Xiaoxing Ren, Nicola Bastianello, Karl H. Johansson, Thomas Parisini, "Communication-Efficient Stochastic Distributed Learning", arXiv 2025. https://arxiv.org/abs/2501.13516

---
[Back to the Canon](../README.md)
