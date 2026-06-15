# LoQT

Implements LoQT, low-rank quantized training that trains large models from scratch using quantized weights and periodically merged low-rank adapters.

LoQT keeps the full weight matrix quantized and frozen, and learns a low-rank update $PB$ on top of it. The projection $P$ is set to the top-$r$ left singular vectors of the weight gradient $G^W$, which acts as a fixed low-dimensional subspace; only the small factor $B$ accumulates an optimizer state and is trained, with its gradient obtained by projecting the full gradient through $P^{\top}$. At an exponentially growing schedule of merge steps $T_i$ the adapter is folded into the dequantized weight and re-quantized, the subspace $P$ is recomputed from the new gradient, and $B$ is reinitialized to compensate the quantization error of the merged weight via the pseudo-inverse of $P$. This restricts optimizer-state memory to the rank-$r$ factor while the base weight and projection stay in low precision.

$$
\begin{aligned}
G_t^{W} &= U \Sigma V^{\top}, \qquad P \leftarrow U[:,{:}r] \\
G_t^{B} &= P^{\top} G_t^{W}, \qquad B_{t+1} = B_t - \eta\, \rho(G_t^{B}) \\
W_{T_i} &= W_{T_{i-1}} + P_{T_{i-1}} B_{T_{i-1}}, \qquad Q_{T_i} = q_{\mathrm{NF4}}(W_{T_i}) \\
B_{T_i} &\leftarrow P_{T_i}^{+}\big(W_{T_i} - Q_{T_i}\big) \\
(T_i)_{i=0}^{\infty} &= \big(\tau + \psi^{\,i}\big)_{i=0}^{\infty}
\end{aligned}
$$

where $\theta$ are the model parameters factored as $W = Q + PB$, $W$ the full-precision weight, $G_t^{W}$ its gradient, $U\Sigma V^{\top}$ the SVD of that gradient, $P$ the rank-$r$ projection (top-$r$ left singular vectors, frozen between merges), $B$ the trained low-rank factor, $G_t^{B}$ its projected gradient, $\rho$ the inner optimizer (Adam) supplying the per-coordinate step, $\eta$ the learning rate, $q_{\mathrm{NF4}}$ the NF4 quantizer and $Q$ the quantized base weight, $P^{+}$ the Moore-Penrose pseudo-inverse used to reinitialize $B$ so that $PB$ cancels the quantization error $W-Q$, $\{T_i\}$ the exponentially increasing merge schedule with initial gap $\tau$ and growth factor $\psi$.

Reference: Sebastian Loeschcke, Mads Toftrup, Michael J. Kastoryano, Serge Belongie, Vésteinn Snæbjarnarson, "LoQT: Low-Rank Adapters for Quantized Pretraining", NeurIPS 2024. https://arxiv.org/abs/2405.16528

---
[Back to the Canon](../README.md)
