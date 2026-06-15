# SPARQ-SGD

Implements SPARQ-SGD, a decentralized SGD with event-triggered, compressed (sparsified and quantized) communication.

Each node runs plain local SGD between communication rounds. At designated synchronization indices it checks an event trigger: it communicates with neighbors only when its model has drifted far enough from the last broadcast estimate. When triggered, the node sends a compressed difference $\mathcal{C}(\cdot)$ relative to neighbors' stored estimates, the estimates are refreshed incrementally, and a gossip consensus step mixes the iterates over the doubly stochastic network matrix. This couples local steps, lazy communication, and lossy compression while matching the convergence rate of vanilla decentralized SGD.

$$
\begin{aligned}
g_i^{(t)} &= \nabla F_i(x_i^{(t)}, \xi_i^{(t)}), \\
x_i^{(t+1/2)} &= x_i^{(t)} - \eta_t\, g_i^{(t)}, \\
q_i^{(t)} &= \begin{cases} \mathcal{C}\big(x_i^{(t+1/2)} - \hat{x}_i^{(t)}\big), & (t{+}1)\in\mathcal{I}_t,\ \|x_i^{(t+1/2)} - \hat{x}_i^{(t)}\|_2^2 > c_t\eta_t^2, \\ 0, & \text{otherwise}, \end{cases} \\
\hat{x}_j^{(t+1)} &= \hat{x}_j^{(t)} + q_j^{(t)}, \qquad j\in\{i\}\cup\mathcal{N}_i, \\
x_i^{(t+1)} &= x_i^{(t+1/2)} + \gamma \sum_{j\in\mathcal{N}_i} w_{ij}\big(\hat{x}_j^{(t+1)} - \hat{x}_i^{(t+1)}\big).
\end{aligned}
$$

where $x_i$ is node $i$'s model, $\hat{x}_j$ the locally stored estimate of node $j$'s model, $\eta_t$ the learning rate, $\gamma$ the consensus step size, $\mathcal{C}$ a compression operator with $\mathbb{E}\|x - \mathcal{C}(x)\|_2^2 \le (1-\omega)\|x\|_2^2$, $\mathcal{N}_i$ the neighbors of $i$, $W=(w_{ij})$ the symmetric doubly stochastic mixing matrix, $\mathcal{I}_t$ the set of indices where the trigger is checked, and $c_t \le c_0 t^{1-\varepsilon}$ the increasing threshold sequence.

Reference: Navjot Singh, Deepesh Data, Jemin George, Suhas Diggavi, "SPARQ-SGD: Event-Triggered and Compressed Communication in Decentralized Stochastic Optimization", arXiv preprint 2019. https://arxiv.org/abs/1910.14280

---
[Back to the Canon](../README.md)
