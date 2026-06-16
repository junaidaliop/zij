# Q-GaLore

Implements Q-GaLore, quantized low-rank-gradient training with INT8 weights, INT4 projection matrices, and a lazy layer-adaptive subspace.

Q-GaLore builds on GaLore, which keeps the optimizer state small by running Adam in a periodically refreshed low-rank subspace of the gradient. Every $T$ steps the projection $P_t$ is taken from the top-$r$ left singular vectors of the gradient $G_t$; the gradient is projected down, updated by Adam in that subspace, then projected back and scaled before being applied to the weights. Q-GaLore adds three things on top: weights are held in INT8 and projection matrices in INT4 (block size $256$), so the full-precision copy is never materialized; weight updates use stochastic rounding $\mathcal{F}_{\mathrm{SR}}$ so that quantization is unbiased and accumulated small updates are not lost; and the SVD interval for each layer is doubled whenever its successive projections stay aligned (cosine similarity $\ge 40\%$ over $k$ checks), confining costly re-projection to layers that are still exploring.

$$
\begin{aligned}
P_t &= U[:, :r], \quad U S V^\top = \mathrm{SVD}(G_t) \ \text{ if } t \bmod T = 0, \ \text{ else } P_t = P_{t-1}, \\
R_t &= P_t^\top G_t, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, R_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, R_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \\
N_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}, \\
\tilde{G}_t &= \alpha\, P_t N_t, \\
\theta_t &= \mathcal{F}_{\mathrm{SR}}\!\left(\theta_{t-1} - \gamma\, \tilde{G}_t\right), \quad
\mathcal{F}_{\mathrm{SR}}(x) = \begin{cases} \lceil x \rceil & \text{with prob. } x - \lfloor x \rfloor \\ \lfloor x \rfloor & \text{with prob. } \lceil x \rceil - x \end{cases}
\end{aligned}
$$

where $\theta$ are the INT8 weights, $G_t$ the full gradient, $P_t$ the INT4 projection matrix of rank $r$ refreshed every $T$ steps, $R_t$ the projected low-rank gradient, $m_t,v_t$ its first and second Adam moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected forms, $N_t$ the normalized low-rank update, $\alpha$ the projection scale, $\gamma$ the learning rate, $\epsilon$ a stability constant, and $\mathcal{F}_{\mathrm{SR}}$ the stochastic-rounding operator, which is unbiased ($\mathbb{E}[\mathcal{F}_{\mathrm{SR}}(x)] = x$).

Reference: Zhenyu Zhang, Ajay Jaiswal, Lu Yin, Shiwei Liu, Jiawei Zhao, Yuandong Tian, Zhangyang Wang, "Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients", arXiv 2024. https://arxiv.org/abs/2407.08296

---
[Back to the Canon](../index.md)
