# RSO

Implements RSO (Randomized Subspace Optimization), a memory-efficient method that trains large models by repeatedly solving low-dimensional subproblems.

At each outer step a random projection matrix $P_k$ maps a small variable $B$ back into the full parameter space, and the loss is minimized over $B$ together with a proximal penalty. Because the inner optimizer (typically Adam) only ever sees the reduced variable $B$, both the optimizer states and the activation gradients are kept low-dimensional, while the full weights are still updated through $P_k$. The subproblem is solved only approximately, started from $B = 0$ each outer iteration.

$$
\begin{aligned}
P_k &\sim \mathcal{N}\!\left(0, \tfrac{1}{r}\right), \\
\tilde{B}_k &\approx \arg\min_{B}\ \Big\{ f(\theta_k + P_k B) + \tfrac{1}{2\eta_k}\lVert B \rVert^2 \Big\}, \\
\theta_{k+1} &= \theta_k + P_k\,\tilde{B}_k.
\end{aligned}
$$

where $\theta_k$ are the full weights, $P_k$ is a random projection ($r$ the subspace dimension), $B$ is the low-dimensional subspace variable solved from $B=0$ with a standard inner optimizer, $f$ is the training loss, and $\eta_k = 1/(2\hat{L})$ controls the proximal term.

Reference: Yiming Chen, Yuan Zhang, Yin Liu, Kun Yuan, Zaiwen Wen, "A Memory Efficient Randomized Subspace Optimization Method for Training Large Language Models", arXiv 2025. https://arxiv.org/abs/2502.07222

---
[Back to the Canon](../index.md)
