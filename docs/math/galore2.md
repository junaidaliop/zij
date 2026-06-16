# GaLore 2

Implements GaLore 2, a scalable realization of gradient low-rank projection for large-scale LLM pre-training.

GaLore keeps the optimizer state small by running Adam inside a periodically refreshed low-rank subspace of the gradient. Every $T$ steps a projection $P_t$ is taken from the leading singular vectors of the gradient $G_t$; the gradient is projected down to rank $r$, the Adam moments are kept only in that subspace, and the resulting update is projected back to full rank, scaled by $\alpha$, and applied to the weights. Between refreshes the previous projector is reused ($P_t = P_{t-1}$).

GaLore 2 leaves this update rule unchanged and instead makes it practical at scale: the costly exact SVD is replaced by fast randomized SVD (Halko et al., up to about 15x faster with no accuracy loss), and the optimizer is fused with FSDP so each layer's weight update runs as soon as its gradient is ready, minimizing the gradient memory held during the backward pass. The subspace orientation depends on the parameter shape: for $m \le n$ the left singular vectors are used, for $m > n$ the right singular vectors.

$$
\begin{aligned}
P_t &= \begin{cases} U[:, :r] & m \le n \\ \big(V[:, :r]\big)^\top & m > n \end{cases}, \quad U S V^\top = \mathrm{SVD}(G_t)\ \text{ if } t \bmod T = 0,\ \text{ else } P_t = P_{t-1}, \\
R_t &= P_t^\top G_t, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, R_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, R_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}, \\
N_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}, \qquad \tilde{G}_t = \alpha\, P_t N_t, \\
\theta_t &= \theta_{t-1} - \eta\, \tilde{G}_t.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $G_t$ the full-rank gradient of shape $m \times n$, $P_t$ the rank-$r$ projection matrix refreshed every $T$ steps via (randomized) SVD, $R_t$ the projected low-rank gradient, $m_t,v_t$ its first and second Adam moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected forms, $N_t$ the normalized low-rank update, $\alpha$ the projection scale, and $\epsilon$ a stability constant.

Reference: DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, Jiawei Zhao, "GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection", arXiv 2025. https://arxiv.org/abs/2504.20437

---
[Back to the Canon](../index.md)
