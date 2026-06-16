# 4-bit-Muon-GRASP

Implements 4-bit-Muon-GRASP, a memory-efficient Muon that stores the momentum state in 4 bits via subspace-preserving grid quantization.

Muon orthogonalizes the momentum matrix through Newton-Schulz iterations, but that orthogonalization sharply amplifies the quantization error of the momentum buffer, with the damage concentrated in the top singular subspace. GRASP (GRid And Subspace Preserving) splits the momentum $M_t$ into a top-rank factor $P_t R_t^\top$, kept at 8 bits, and a residual $M_{\mathrm{res},t}$, compressed to 4 bits. The top factor is tracked cheaply with a single warm-started power iteration. Both pieces are stored with grid quantization, which normalizes each block entry by the smaller of its row and column maxima so that outliers spanning either dimension are bounded tightly.

$$
\begin{aligned}
M_t &= \mu M_{t-1} + g_t, \qquad M_{t-1} = M_{\mathrm{res},t-1} + P_{t-1} R_{t-1}^\top, \\
P_t, R_t &= \mathrm{PowerIter}(M_t, Q_t), \qquad Q_t = \mathrm{ColNorm}(R_{t-1}), \quad P_t R_t^\top \approx M_{\mathrm{top}}, \\
M_{\mathrm{res},t} &= M_t - P_t R_t^\top, \qquad M^q_{\mathrm{res},t} = \mathrm{QUANT}_4(M_{\mathrm{res},t}), \quad P^q_t = \mathrm{QUANT}_8(P_t), \quad R^q_t = \mathrm{QUANT}_8(R_t), \\
O_t &= \mathrm{NewtonSchulz}(M_t, T), \\
W_t &= W_{t-1} - \eta_t \left( O_t + \lambda W_{t-1} \right), \\
N_{\mathrm{grid}}(x_{i,j}) &= \frac{x_{i,j}}{\min\!\left( \mathrm{scale}^r_i, \, \mathrm{scale}^c_j \right)}, \qquad \mathrm{scale}^r_i = \max_j |x_{i,j}|, \quad \mathrm{scale}^c_j = \max_i |x_{i,j}|.
\end{aligned}
$$

where $W$ are the weights, $g_t = \nabla L_t(W_{t-1})$ the gradient, $M_t$ the momentum, $\mu$ the momentum decay, $\eta_t$ the learning rate, $\lambda$ the weight decay, and $T$ the Newton-Schulz iteration count; $P_t \in \mathbb{R}^{m\times k}$, $R_t \in \mathbb{R}^{n\times k}$ are the rank-$k$ top singular factors obtained by one power-iteration step ($P \leftarrow \mathrm{Orthogonalize}(M_t Q_t)$, $R \leftarrow M_t^\top P$), $M_{\mathrm{res},t}$ is the residual subspace, $\mathrm{QUANT}_b$ a $b$-bit quantizer, and $\mathrm{scale}^r_i, \mathrm{scale}^c_j$ the per-row and per-column maxima within each $s\times s$ block used by grid quantization.

Reference: Huaijin Wu, Bingrui Li, Yebin Yang, Yi Tu, Zhanpeng Zhou, Jianfei Chen, Junchi Yan, "Achieving low-bit Muon through subspace preservation and grid quantization", ICLR 2026. https://openreview.net/forum?id=g2l9bg9DWx

---
[Back to the Canon](../index.md)
