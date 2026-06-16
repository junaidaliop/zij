# ProjFactor (VLoRP)

Implements ProjFactor, the optimizer for Various-Grained Low-Rank Projection (VLoRP) that keeps a projected first moment and an Adafactor-style factored second moment in the subspace.

VLoRP reshapes the gradient of a weight matrix $W \in \mathbb{R}^{n \times m}$ into $[nc,\, m/c]$ and right-multiplies it by a random projection $\tilde P \in \mathbb{R}^{(m/c) \times r}$ with entries drawn from $\mathcal N(0, 1/r)$, trading off rank $r$ against granularity $c$ at a fixed compression budget. ProjFactor stores the momentum $\tilde m^s$ in the low-rank subspace and tracks only the row and column sums of the squared back-projected gradient, so the optimizer state grows as $O(n + m)$ rather than $O(nm)$. At each step the momentum is projected back to the original space, divided by the rank-1 reconstruction of the second moment, reshaped, and applied to $W$.

$$
\begin{aligned}
\tilde G_t^s &= \mathrm{Reshape}(g_t,\, [nc,\, m/c])\, \tilde P, \qquad \tilde p_{ij} \sim \mathcal N(0,\, 1/r) \\
\tilde m_t^s &= \beta_1 \tilde m_{t-1}^s + (1 - \beta_1)\, \tilde G_t^s \\
\tilde v_t^{ro} &= \beta_2 \tilde v_{t-1}^{ro} + (1 - \beta_2)\, (\tilde G_t^s \tilde P^{\top})^{\odot 2}\, \mathbf{1}_m \\
\tilde v_t^{co} &= \beta_2 \tilde v_{t-1}^{co} + (1 - \beta_2)\, \mathbf{1}_n^{\top} (\tilde G_t^s \tilde P^{\top})^{\odot 2} \\
\hat v_t^o &= \frac{\tilde v_t^{ro}\, \tilde v_t^{co}}{\mathbf{1}_n^{\top} \tilde v_t^{ro}} \\
\Delta_t^o &= \mathrm{Reshape}\!\left( \frac{\tilde m_t^s \tilde P^{\top}}{\sqrt{\hat v_t^o + \epsilon}},\, [n,\, m] \right) \\
\theta_t &= \theta_{t-1} - \eta\, \frac{1 - \beta_2^t}{1 - \beta_1^t}\, \Delta_t^o
\end{aligned}
$$

where $\theta$ (the matrix $W$) are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $\tilde P$ the random down-projection of rank $r$, $\tilde m_t^s$ the first moment kept in the $r$-dimensional subspace, $\tilde v_t^{ro}, \tilde v_t^{co}$ the row and column second-moment factors of the back-projected gradient $\tilde G_t^s \tilde P^{\top}$, $\hat v_t^o$ their rank-1 reconstruction, $\beta_1, \beta_2$ the decay rates, $\epsilon$ a stability constant, $\odot 2$ elementwise squaring, $\mathbf 1$ all-ones vectors, $n, m$ the matrix dimensions, $c$ the granularity, and $(1 - \beta_2^t)/(1 - \beta_1^t)$ the bias-correction scalar at step $t$.

Reference: Yezhen Wang, Zhouhao Yang, Brian K. Chen, Fanyi Pu, Bo Li, Tianyu Gao, Kenji Kawaguchi, "Memory-Efficient LLM Training by Various-Grained Low-Rank Projection of Gradients", ICML 2025. https://arxiv.org/abs/2505.01744

---
[Back to the Canon](../index.md)
