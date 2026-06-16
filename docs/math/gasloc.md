# GASLoC

Implements GASLoC, a decentralized low-communication method that fuses gossip averaging with the outer optimizer of local-update training.

GASLoC runs each worker $i$ through $H_i$ local inner steps, then forms a pseudo-gradient $g_t^i$ equal to the drift of its parameters over the inner phase. Instead of a global all-reduce, workers exchange parameters only with sparse neighbors in a communication graph: the outer step applies the proposed update $\theta_t + \eta g_t$ and then subtracts a gossip term built from the weighted graph Laplacian $\Lambda$, so each worker is pulled toward its peers. This generalizes the DiLoCo outer optimizer to arbitrary topologies and supports randomized one- or two-peer matchings.

An accelerated variant adds a momentum term on the post-local-update iterates, which provably improves the dependence on the spectral gap $\chi$ of the gossip matrix:

$$
\begin{aligned}
\theta_{t,0}^i &= \theta_t^i, \qquad \theta_{t,h+1}^i = \theta_{t,h}^i - \beta\,\nabla f(\theta_{t,h}^i), \quad h = 0,\dots,H_i-1 \\
g_t^i &= \theta_{t,H_i}^i - \theta_{t,0}^i \\
\theta_{t+1} &= \theta_t + \eta\, g_t - \alpha\,\Lambda(\theta_t + \eta\, g_t) + \gamma\big[(\theta_t + \eta\, g_t) - (\theta_{t-1} + \eta\, g_{t-1})\big]
\end{aligned}
$$

where $\theta_t^i$ are worker $i$'s parameters at outer step $t$, $g_t^i$ is its pseudo-gradient (the inner-phase drift), $\beta$ is the inner learning rate, $\eta$ is the outer learning rate, $H_i$ is the number of local steps on worker $i$, $\Lambda = \tfrac{1}{2}\sum_{(i,j)\in\mathcal{E}}\lambda_{ij}(e_i-e_j)(e_i-e_j)^\top$ is the weighted graph Laplacian of the communication graph, $\alpha$ is the gossip step size, and $\gamma$ is the momentum parameter applied to the post-update iterates ($\gamma = 0$ recovers the non-accelerated form). Stacked over workers, $\theta_t$ and $g_t$ collect all $\theta_t^i$ and $g_t^i$.

Reference: Pietro Cagnasso, Eugene Belilovsky, Edouard Oyallon, "Unifying Local Communications and Local Updates for LLM Pretraining", arXiv 2026. https://arxiv.org/abs/2606.11081

---
[Back to the Canon](../index.md)
