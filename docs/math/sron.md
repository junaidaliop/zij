# SRON

Implements SRON (SGD with Row-wise Normalization), a state-free optimizer that rescales each row of a 2-D gradient by its own root-mean-square before a plain SGD step.

The method is motivated by the observation that row-wise gradient scales in LLM training, especially in the attention module, vary widely: flattening the gradient into a single vector for global normalization lets one large dimension suppress updates in the others. Instead of storing any moments, SRON forms a diagonal matrix $V_t$ from the per-row second moment of the current gradient and left-multiplies the gradient by it, so each row is normalized to a comparable magnitude. With no first- or second-moment buffers, it eliminates optimizer state memory entirely.

$$
\begin{aligned}
G_t &= \frac{1}{b}\sum_{i=1}^{b} \nabla_{W} f_i(W_t) \in \mathbb{R}^{m\times n}, \\
(V_t)_{i,i} &= \left( \sqrt{\tfrac{1}{n}\sum_{j=1}^{n} (G_t)_{i,j}^{2}} \; + \epsilon \right)^{-1}, \\
\tilde{G}_t &= V_t\, G_t, \\
W_{t+1} &= W_t - \alpha\, \eta_t\, \tilde{G}_t .
\end{aligned}
$$

where $W$ is the weight matrix with rows indexed by $i$ and columns by $j$, $G_t$ is the batch gradient, $V_t=\mathrm{diag}(\cdot)$ is the diagonal row-wise normalizer (one entry per row), $\eta_t$ is the learning rate, $\alpha$ is a scaling coefficient, and $\epsilon>0$ is a numerical-stability constant.

Reference: Ziqing Wen, Yanqi Shi, Jiahuan Wang, Ping Luo, Linbo Qiao, Dongsheng Li, Tao Sun, "SRON: State-free LLM Training via Row-wise Gradient Normalization", ICLR 2026 submission. https://openreview.net/forum?id=BtQLBWr6zI

---
[Back to the Canon](../index.md)
