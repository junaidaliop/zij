# Top-DP

Implements Top-DP, a topology-aware differentially private optimizer for decentralized training.

Top-DP runs differentially private decentralized SGD, where each agent mixes its own clipped local update with a neighbor's parameters and injects Gaussian noise for privacy. Its key idea is topology-aware noise reduction: because a neighbor's parameter $\tilde{x}_k$ already carries that neighbor's privacy noise $G_k$, agent $i$ can subtract the inherited noise variance and add only the residual amount $G_i^j$ needed to reach its target noise level. A time-aware decay shrinks the noise scale over training to improve the utility-privacy trade-off.

$$
\begin{aligned}
\bar{g}(\tilde{x}_i, \xi_s) &= \frac{g(\tilde{x}_i, \xi_s)}{\max\!\left(1,\ \frac{\lVert g(\tilde{x}_i, \xi_s)\rVert_2}{C}\right)} \\
\sigma_i^j &= \sqrt{\sigma_i^2 - (1-\alpha)^2 \sigma_k^2} \\
G_i^j &\sim \mathcal{N}\!\left(0,\ (\sigma_i^j)^2 C^2 \mathbf{I}\right) \\
\tilde{x}_i^j &= \alpha\, \tilde{x}_i + (1-\alpha)\, \tilde{x}_k^i - \lambda\, \bar{g}(\tilde{x}_i, \xi_s) + G_i^j \\
\sigma_{t,i} &= \sigma_{0,i}\, \gamma^{\lfloor t/\mathrm{period} \rfloor}
\end{aligned}
$$

where $\tilde{x}_i$ is agent $i$'s noised local estimate, $\tilde{x}_k^i$ the parameter received from neighbor $k$, $\alpha \in [0,1]$ the mixing weight between local and neighbor contributions, $\lambda$ the learning rate, $g(\tilde{x}_i, \xi_s)$ the stochastic gradient on sample $\xi_s$, $C$ the $\ell_2$ clipping threshold, $\sigma_i$ and $\sigma_k$ the target noise scales of agents $i$ and $k$, $\sigma_i^j$ the reduced (residual) noise scale that exploits the neighbor's inherited noise, $G_i^j$ the added Gaussian noise, and $\gamma \in (0,1)$ the per-period decay factor of the noise scale at step $t$.

Reference: Shangwei Guo, Tianwei Zhang, Guowen Xu, Han Yu, Tao Xiang, Yang Liu, "Topology-aware Differential Privacy for Decentralized Image Classification", arXiv 2020. https://arxiv.org/abs/2006.07817

---
[Back to the Canon](../index.md)
