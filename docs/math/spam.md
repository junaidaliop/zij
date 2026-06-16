# SPAM

Implements SPAM, Spike-Aware Adam with momentum reset for stable training.

SPAM augments Adam with two stabilizing mechanisms. Spike-aware
clipping caps any gradient coordinate whose squared value exceeds a
multiple of its running second moment, replacing it with a magnitude
bounded by that second moment:


$$
\begin{aligned}
g_{t,i} &\leftarrow \mathrm{sign}(g_{t,i})\,
    \sqrt{\tau\, v_{t-1,i}}
    &&\text{if } g_{t,i}^2 > \tau\, v_{t-1,i} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\theta_t &= \theta_{t-1} - \gamma\, \phi_t\,
    \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta_t$ are the parameters, $\tau$ is the fixed
spike-detection `threshold` (default 5000, never updated), $\hat{m}_t$
and $\hat{v}_t$ are the bias-corrected moments, and $\phi_t$
is a cosine warmup factor. Every `update_proj_gap` steps the moments
$m, v$ are reset to zero and the warmup restarts, which clears
accumulated momentum after a spike. For two-dimensional parameters a
random binary mask of fraction `density` selects the coordinates that
keep momentum (sparse momentum), and the mask is resampled at each reset.

Reference: Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang
Wang, Shiwei Liu, "SPAM: Spike-Aware Adam with Momentum Reset for
Stable LLM Training", ICLR 2025.
https://arxiv.org/abs/2501.06842

---
[Back to the Canon](../index.md)
