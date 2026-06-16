# SinkGD

Implements SinkGD, stateless gradient descent that Sinkhorn-normalizes each weight-matrix gradient before the step.

SinkGD treats the per-layer gradient as a matrix and applies a few iterations of a row/column rescaling (a Sinkhorn-like procedure, SR-Sinkhorn) so that, in the limit, every row and every column has a fixed $\ell_2$ norm. Each pass divides the matrix by its row norms (scaled by $\sqrt{n}$), then by its column norms (scaled by $\sqrt{m}$); $L$ such alternating passes give the normalized gradient $\hat g_t$, which is then used for a plain SGD step. The method keeps no optimizer state, so its memory footprint matches SGD while reaching Adam-comparable performance on LLM training.

$$
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\theta_t) \in \mathbb{R}^{m\times n}, \qquad X^{(0)} = g_t \\
X^{(\ell)} &= \sqrt{m}\,\Big(\sqrt{n}\, Q(X^{(\ell-1)})^{-1} X^{(\ell-1)}\Big) R\!\big(\sqrt{n}\, Q(X^{(\ell-1)})^{-1} X^{(\ell-1)}\big)^{-1}, \quad \ell = 1,\dots,L \\
\hat g_t &= X^{(L)} \\
\theta_{t+1} &= \theta_t - \eta_t\, \hat g_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the gradient reshaped to an $m\times n$ matrix, $\eta_t$ the learning rate, $L$ the number of SR-Sinkhorn iterations, $Q(X) = \mathrm{diag}(\lVert X_{1,:}\rVert_2,\dots,\lVert X_{m,:}\rVert_2)$ the diagonal of row $\ell_2$ norms, and $R(X) = \mathrm{diag}(\lVert X_{:,1}\rVert_2,\dots,\lVert X_{:,n}\rVert_2)$ the diagonal of column $\ell_2$ norms. SinkGD maintains no moments, weight decay, or $\epsilon$ term.

Reference: Meyer Scetbon, Chao Ma, Wenbo Gong, Edward Meeds, "Gradient Multi-Normalization for Stateless and Scalable LLM Training", arXiv 2025. https://arxiv.org/abs/2502.06742

---
[Back to the Canon](../index.md)
