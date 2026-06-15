# MLorc

Implements MLorc, memory-efficient adaptation that low-rank compresses and reconstructs the optimizer momentum rather than the gradient.

MLorc (Momentum Low-rank compression) keeps Adam's full-rank gradient signal but stores only a rank-$r$ factorization of the first and second moments. At each step the previous moments are reconstructed from their stored factors, updated with the current full gradient $g_t$, then recompressed by randomized SVD before the parameter update. Because the second moment must stay non-negative after a low-rank approximation, the reconstructed $\tilde{v}_{t-1}$ is rectified by a ReLU and shifted by the average magnitude of the discarded negative entries.

The matrix-shaped weight $W$ is updated by the standard bias-corrected AdamW rule using the reconstructed moments:

$$
\begin{aligned}
\tilde{m}_{t-1} &= m_{u,t-1}\, m_{s,t-1}\, m_{v,t-1}^{\top}, \qquad \tilde{v}_{t-1} = v_{u,t-1}\, v_{s,t-1}\, v_{v,t-1}^{\top} \\
\tilde{v}_{t-1} &\leftarrow \mathrm{ReLU}(\tilde{v}_{t-1}) + \big|\, \mathrm{mean}(\tilde{v}_{t-1} - \mathrm{ReLU}(\tilde{v}_{t-1}))\, \big| \\
m_t &= \beta_1\, \tilde{m}_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2\, \tilde{v}_{t-1} + (1-\beta_2)\, g_t^{2} \\
(m_{u,t}, m_{s,t}, m_{v,t}) &= \mathrm{RSVD}(m_t, r, p), \qquad (v_{u,t}, v_{s,t}, v_{v,t}) = \mathrm{RSVD}(v_t, r, p) \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
W_t &= W_{t-1} - \alpha\left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, W_{t-1} \right)
\end{aligned}
$$

where $W$ is a weight matrix, $\alpha$ the learning rate, $g_t$ the full-rank gradient, $m_t$/$v_t$ the first and second moments, $\beta_1,\beta_2$ their decay rates, $\lambda$ the weight decay, $\epsilon$ the stability constant, $r$ the target rank and $p$ the oversampling parameter of the randomized SVD $\mathrm{RSVD}$, and $(\cdot_u, \cdot_s, \cdot_v)$ the stored left-factor, singular, and right-factor matrices whose product reconstructs a moment. The ReLU shift averages only over the negative (discarded) entries; non-negative entries are left unchanged.

Reference: Wei Shen, Yaxiang Zhang, Minhui Huang, Mengfan Xu, Jiawei Zhang, Cong Shen, "MLorc: Momentum Low-rank Compression for Large Language Model Adaptation", arXiv 2025. https://arxiv.org/abs/2506.01897

---
[Back to the Canon](../README.md)
