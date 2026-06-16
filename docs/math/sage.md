# SAGE

Implements SAGE, a sign-based optimizer with a bounded $O(d)$ adaptive scale for memory-efficient LLM training.

SAGE keeps the sign-momentum update direction of Lion but replaces Lion's fixed unit step scale with a per-dimension damper $H_t \in [0, 1]$. The damper is built from a column-wise mean-absolute-gradient statistic (per feature dimension for embeddings, element-wise otherwise), tracked by a single $O(d)$ exponential moving average rather than AdamW's $O(Vd)$ second-moment state. Dimensions whose smoothed magnitude exceeds the layer's RMS reference are scaled down, taming high-variance gradients while the scale stays provably bounded by $1$.

$$
\begin{aligned}
\theta_{t-1} &\leftarrow \theta_{t-1}\,(1 - \eta_t \lambda) \\
S_t &\leftarrow \beta_2 S_{t-1} + (1-\beta_2)\, s_t, \qquad \hat{S}_t = \frac{S_t}{1-\beta_2^{\,t}} \\
\sigma_{\mathrm{rms}} &= \sqrt{\tfrac{1}{d}\textstyle\sum_j (\hat{S}_t)_j^2}, \qquad \gamma_{\mathrm{rms}} = \sqrt{\tfrac{1}{d}\textstyle\sum_j (s_t)_j^2} \\
(H_t)_j &= \min\!\left(\frac{\sigma_{\mathrm{rms}}}{(\hat{S}_t)_j + \epsilon},\; \frac{\gamma_{\mathrm{rms}}}{(s_t)_j + \epsilon},\; 1 \right) \\
C_t &= \mathrm{sign}\!\big(\beta_1 m_{t-1} + (1-\beta_1)\, g_t\big) \\
\theta_t &\leftarrow \theta_{t-1} - \eta_t\, (C_t \odot H_t) \\
m_t &\leftarrow \beta_2 m_{t-1} + (1-\beta_2)\, g_t
\end{aligned}
$$

where $s_t$ is the gradient magnitude snapshot ($(s_t)_j = \tfrac{1}{V}\sum_i |g_{t,ij}|$ over the vocabulary axis for embeddings, $|g_t|$ element-wise otherwise), $S_t$ is its EMA with bias-corrected value $\hat{S}_t$, $H_t$ is the bounded adaptive scale, $C_t$ the sign-momentum direction, $m_t$ the momentum buffer, $\eta_t$ the learning rate, $\lambda$ the weight decay, $\beta_1,\beta_2$ the decay rates, $\epsilon$ a stability constant, $V$ the vocabulary size, and $d$ the feature dimension.

Reference: Wooin Lee, Hyuntae Kim, "SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization", arXiv 2026. https://arxiv.org/abs/2604.07663

---
[Back to the Canon](../index.md)
