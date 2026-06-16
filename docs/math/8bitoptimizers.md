# 8-bit Optimizers

Implements 8-bit Optimizers, stateful optimizers (Adam, Momentum) whose moments are stored in 8 bits via block-wise dynamic quantization.

The optimizer state (the running moments $m_t$ and $v_t$) is the dominant memory cost of stateful methods. Here each state tensor is kept in 8 bits and only its non-quantized 32-bit form is materialized transiently during the update. The state tensor is split into contiguous blocks of $B = 2048$ elements; each block is normalized by its own absolute maximum and mapped to an 8-bit index against a fixed dynamic-quantization codebook $Q^{\mathrm{map}}$. Block-wise normalization confines outliers to a single block, so it does not degrade the precision of all other blocks. At each step the 8-bit states are dequantized, the standard 32-bit update is performed, and the new states are quantized back to 8 bits.

For the $b$-th block with normalization constant $N_b$, quantization, dequantization, and the (Adam) update applied in dequantized space are

$$
\begin{aligned}
N_b &= \max(|T_b|), \\
T_{bi}^{Q} &= \arg\min_{0 \le j \le 2^{n}} \left| Q_j^{\mathrm{map}} - \frac{T_{bi}}{N_b} \right|, \qquad 0 < i < B, \\
T_{bi}^{D} &= Q^{\mathrm{map}}\!\left(T_{bi}^{Q}\right)\cdot N_b, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2, \\
\theta_t &= \theta_{t-1} - \gamma\,\frac{m_t}{\sqrt{v_t}+\epsilon},
\end{aligned}
$$

where $T_b$ is a block of an optimizer state tensor, $N_b$ its block-wise normalization constant, $T_{bi}^{Q}$ the stored 8-bit index, $Q^{\mathrm{map}}$ the dynamic-quantization codebook with $2^n=256$ entries, $T_{bi}^{D}$ the dequantized 32-bit value used for the update, $g_t$ the gradient, $m_t,v_t$ the dequantized first and second moments, $\beta_1,\beta_2$ the decay rates, $\gamma$ the learning rate, and $\epsilon$ a stability constant; $m_t$ and $v_t$ (or, for 8-bit Momentum, the single momentum buffer) are re-quantized to 8 bits after the update.

Reference: Tim Dettmers, Mike Lewis, Sam Shleifer, Luke Zettlemoyer, "8-bit Optimizers via Block-wise Quantization", ICLR 2022. https://arxiv.org/abs/2110.02861

---
[Back to the Canon](../index.md)
