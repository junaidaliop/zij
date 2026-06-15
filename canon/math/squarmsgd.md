# SQuARM-SGD

Implements SQuARM-SGD, decentralized momentum SGD with sparsified-quantized, event-triggered communication.

Each node $i$ runs local Nesterov-momentum SGD between synchronization rounds and keeps compressed estimates $\hat{x}_j$ of its neighbors' parameters. At a synchronization index a node communicates only if its parameters have drifted past a triggering threshold; it then sends a compressed change, every node refreshes its neighbor estimates, and a gossip consensus step mixes the iterates over the connectivity graph. Compression composes sparsification with stochastic quantization, and error feedback is realized implicitly through the accumulated estimates $\hat{x}$.

$$
\begin{aligned}
v_i^{(t)} &= \beta\, v_i^{(t-1)} + g_i^{(t)}, \\
x_i^{(t+\frac{1}{2})} &= x_i^{(t)} - \eta\,(\beta\, v_i^{(t)} + g_i^{(t)}), \\
\hat{x}_j^{(t+1)} &= \hat{x}_j^{(t)} + C\!\left(x_j^{(t+\frac{1}{2})} - \hat{x}_j^{(t)}\right) \quad \text{if } \left\lVert x_j^{(t+\frac{1}{2})} - \hat{x}_j^{(t)} \right\rVert_2^2 > c_t\, \eta^2, \\
x_i^{(t+1)} &= x_i^{(t+\frac{1}{2})} + \gamma \sum_{j \in N_i} w_{ij}\left(\hat{x}_j^{(t+1)} - \hat{x}_i^{(t+1)}\right),
\end{aligned}
$$

where $g_t = \nabla F_i(x_i^{(t)}, \xi_i^{(t)})$ is the stochastic gradient, $\beta$ the momentum coefficient, $\eta$ the learning rate, $\gamma$ the consensus step-size, $C(\cdot)$ the sparsify-then-quantize compression operator, $\hat{x}_j$ the neighbor estimates, $c_t$ the triggering threshold, $N_i$ the neighbors of node $i$, and $w_{ij}$ the entries of the doubly stochastic mixing matrix $W$. The consensus and estimate updates fire only at synchronization indices $t+1 \in \mathcal{I}_T$; otherwise $x_i^{(t+1)} = x_i^{(t+\frac{1}{2})}$ and $\hat{x}_i^{(t+1)} = \hat{x}_i^{(t)}$.

Reference: Navjot Singh, Deepesh Data, Jemin George, Suhas Diggavi, "SQuARM-SGD: Communication-Efficient Momentum SGD for Decentralized Optimization", IEEE Journal on Selected Areas in Information Theory 2021. https://arxiv.org/abs/2005.07041

---
[Back to the Canon](../README.md)
