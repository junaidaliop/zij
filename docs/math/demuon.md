# DeMuon

Implements DeMuon, a decentralized Muon for matrix optimization over networks.

DeMuon runs Muon across $N$ workers connected by a communication graph, with no central server. Each worker $i$ keeps a local momentum estimate $M_i$, a gradient-tracking variable $V_i$ that follows the network-wide gradient, and a local iterate $X_i$. Mixing with neighbors through the doubly stochastic matrix $W$ drives the iterates toward consensus, while the matrix-sign (orthogonalization) step is applied to the tracked gradient so the structure-aware Muon update is preserved in the distributed setting.

$$
\begin{aligned}
M_i^k &= (1-\theta)\, M_i^{k-1} + \theta\, g_i^k \\
V_i^k &= \sum_{j=1}^{N} w_{ij}\bigl(V_j^{k-1} + M_j^k - M_j^{k-1}\bigr) \\
X_i^{k+1} &= \sum_{j=1}^{N} w_{ij}\bigl(X_j^k - \eta\, \mathrm{msgn}(V_j^k)\bigr)
\end{aligned}
$$

where $X_i \in \mathbb{R}^{m\times n}$ is worker $i$'s parameter matrix, $g_i^k$ its stochastic gradient, $\theta \in (0,1)$ the momentum weight, $\eta$ the step size, $w_{ij}$ the entries of the mixing matrix $W$ (nonzero only for neighbors), and $\mathrm{msgn}(M) = UV^\top$ from the reduced SVD $M = U\Sigma V^\top$ is the matrix sign that orthogonalizes the tracked gradient.

Reference: Chuan He, Shuyi Ren, Jingwei Mao, Erik G. Larsson, "DeMuon: A Decentralized Muon for Matrix Optimization over Graphs", arXiv 2025. https://arxiv.org/abs/2510.01377

---
[Back to the Canon](../index.md)
