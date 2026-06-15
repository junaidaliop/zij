# RMNP

Implements RMNP (Row-Momentum Normalized Preconditioning), a matrix-based optimizer that replaces Muon's Newton-Schulz orthogonalization with cheap row-wise normalization.

For a weight matrix, RMNP maintains a heavy-ball momentum estimate of the gradient and then normalizes each of its rows to unit Euclidean length before stepping. This row normalization plays the role of the preconditioner: it bounds the per-row update magnitude without the spectral whitening of Newton-Schulz, dropping the per-step cost from $O(mn\min(m,n))$ to $O(mn)$ while keeping competitive performance on large-model pretraining. Non-matrix parameters are typically left to AdamW.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t, \\
d_t &= \mathrm{diag}\!\big(m_t m_t^{\top}\big)^{-1/2} m_t, \qquad [d_t]_{i,:} = \frac{[m_t]_{i,:}}{\lVert [m_t]_{i,:} \rVert_2}, \\
\theta_t &= \theta_{t-1} - \eta\, d_t.
\end{aligned}
$$

where $\theta$ is the weight matrix, $g_t$ its gradient, $m_t$ the momentum estimate, $\beta$ the momentum coefficient, $\eta$ the learning rate, and $[\cdot]_{i,:}$ denotes the $i$-th row, so $d_t$ is $m_t$ with every row rescaled to unit $\ell_2$ norm.

Reference: Shenyang Deng, Zhuoli Ouyang, Tianyu Pang, Zihang Liu, Ruochen Jin, Shuhua Yu, Yaoqing Yang, "RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization", ICML 2026. https://arxiv.org/abs/2603.20527

---
[Back to the Canon](../README.md)
