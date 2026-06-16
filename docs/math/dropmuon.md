# Drop-Muon

Implements Drop-Muon, a Muon-style method that updates only a random subset of layers at each step.

Drop-Muon challenges the convention of refreshing every layer at every iteration. At each step it samples a subset $S^k$ of the network's layers from a distribution $\mathcal{D}$, updates only those layers, and freezes the rest (momentum and weights held fixed). Active layers receive the standard layer-wise momentum followed by a non-Euclidean step through the sharp operator $(\cdot)^\sharp$, which under the spectral norm is exactly Muon's orthogonalization $U V^\top$ from the SVD $M = U \Sigma V^\top$. The default sampling scheme, Randomized Progressive Training, draws a starting index $s^k$ and sets $S^k = \{s^k, \dots, b\}$, so the cheaply-computed downstream gradients from backpropagation are reused while earlier layers stay frozen, cutting per-iteration cost.

$$
\begin{aligned}
i \notin S^k:\quad & M_i^k = M_i^{k-1}, \qquad X_i^{k+1} = X_i^k \\
i \in S^k:\quad & M_i^k = (1 - \beta_i)\,M_i^{k-1} + \beta_i\,\nabla_i f(X^k; \xi^k) \\
& X_i^{k+1} = X_i^k - \gamma_i^k\,(M_i^k)^\sharp \\
\text{where}\quad & M^\sharp = \mathrm{arg\,max}_{X \in \mathcal{X}}\,\Big\{ \langle M, X\rangle - \tfrac{1}{2}\lVert X\rVert^2 \Big\}
\end{aligned}
$$

where $X_i^k$ are the weights of layer $i$ at iteration $k$, $M_i^k$ its momentum buffer, $\nabla_i f(X^k;\xi^k)$ the stochastic gradient for layer $i$, $\beta_i \in [0,1)$ the per-layer momentum, $\gamma_i^k > 0$ the per-layer step size, $S^k \sim \mathcal{D}$ the active subset, and $(\cdot)^\sharp$ the sharp operator for the norm $\lVert\cdot\rVert$ on the layer's space $\mathcal{X}$ (the identity under the Euclidean norm, and the orthogonalization $UV^\top$ under the spectral norm, recovering Muon).

Reference: Kaja Gruntkowska, Yassine Maziane, Zheng Qu, Peter Richtárik, "Drop-Muon: Update Less, Converge Faster", arXiv 2025. https://arxiv.org/abs/2510.02239

---
[Back to the Canon](../index.md)
