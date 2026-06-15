# SCALE

Implements SCALE (Stochastic Column-normAlized Last-layer momEntum), a minimalist memory-efficient optimizer for LLM pretraining.

SCALE strips Adam down to two ingredients. It drops second-order moments entirely and keeps first-order momentum only for the last layer, while every other layer uses the raw stochastic gradient. The update direction for each weight matrix is then column-normalized: each column is divided by its own L2 norm. This corresponds to steepest descent under the $\|\cdot\|_{1\to 2}$ operator norm, which removes the per-parameter adaptive state of Adam yet preserves most of its conditioning benefit at near-SGD memory cost.

For a layer $l$ with weight matrix $\theta_l$ and gradient $g_t$, the update is

$$
\begin{aligned}
m_t &= \begin{cases} \beta\, m_{t-1} + (1-\beta)\, g_t, & l = L \\ g_t, & l \neq L \end{cases} \\
\theta_{t+1} &= \theta_t - \eta\, C(m_t), \qquad C(M)_{:,j} = \frac{M_{:,j}}{\lVert M_{:,j}\rVert_2}
\end{aligned}
$$

where $\theta_t$ are the layer parameters, $\eta$ is the learning rate, $g_t$ the stochastic gradient, $m_t$ the first-order momentum, $\beta$ the momentum decay, $L$ the index of the last layer, and $C(\cdot)$ the column-wise normalization that divides each column $M_{:,j}$ by its Euclidean norm.

Reference: Athanasios Glentis, Jiaxiang Li, Andi Han, Mingyi Hong, "A Minimalist Optimizer Design for LLM Pretraining", arXiv 2025. https://arxiv.org/abs/2506.16659

---
[Back to the Canon](../README.md)
