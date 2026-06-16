# 4-bit Optimizers

Implements 4-bit Optimizers, AdamW with first- and second-moment states stored at 4-bit precision.

The base update is ordinary AdamW; the only change is that the optimizer states $m_t$ and $v_t$ are kept in a 4-bit compressed form between steps. A state is compressed by first normalizing it (block-wise for $m_t$, and a rank-1 scheme for the matrix-shaped $v_t$ that divides by the smaller of the per-row and per-column maxima), then mapping the normalized value to the nearest 4-bit codeword. Before each step the state is dequantized back to floating point. The second moment uses a linear codebook that excludes zero, since codewords near zero would otherwise blow up the $1/\sqrt{v_t}$ direction.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \\
q_j &= Q(x_j) = M\big(N(x_j)\big), \qquad \tilde{x}_j = N^{-1}\big(T(q_j)\big) \\
N_{\mathrm{block}}(x_j) &= \frac{x_j}{\max\{\,|x_i| : 1+B\lfloor j/B\rfloor \le i \le B(\lfloor j/B\rfloor+1)\,\}} \\
N_{\mathrm{rank\text{-}1}}(x_{i,j}) &= \frac{x_{i,j}}{\min\{r_i, c_j\}}, \qquad r_i = \max_j |x_{i,j}|,\ \ c_j = \max_i |x_{i,j}|
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t/v_t$ the first/second moments, $\beta_1,\beta_2$ the decay rates, $\epsilon$ the stability constant, $Q$ the quantizer composing normalization $N$ and codebook mapping $M$, $T$ the dequantization map (linear $T(i)=(i+1)/2^b$ for $v_t$, dynamic-exponent for $m_t$, $b=4$ bits), $B$ the block size (128), and $r_i,c_j$ the per-row and per-column maxima used by the rank-1 normalization of the matrix-shaped second moment.

Reference: Bingrui Li, Jianfei Chen, Jun Zhu, "Memory Efficient Optimizers with 4-bit States", NeurIPS 2023. https://arxiv.org/abs/2309.01507

---
[Back to the Canon](../index.md)
