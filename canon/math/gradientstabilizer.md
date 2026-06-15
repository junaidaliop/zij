# GradientStabilizer

Implements GradientStabilizer (AdaGN), the adaptive gradient-norm scaling from Stable-SPAM.

The idea is to stabilize the gradient before it enters an Adam-style update by rescaling it according to its historical $\ell_2$-norm statistics. The current gradient is normalized to a unit direction and then rescaled by a smoothed estimate of its typical magnitude, with the magnitude estimate formed from exponential moving averages of the gradient norm and its square. This damps gradient-norm spikes that destabilize low-precision training while preserving direction.

$$
\begin{aligned}
g_{\mathrm{norm}} &= \lVert g_t \rVert_2 \\
m_{\mathrm{norm},t} &= \gamma_1\, m_{\mathrm{norm},t-1} + (1-\gamma_1)\, g_{\mathrm{norm}} \\
v_{\mathrm{norm},t} &= \gamma_2\, v_{\mathrm{norm},t-1} + (1-\gamma_2)\, g_{\mathrm{norm}}^2 \\
\hat{m}_{\mathrm{norm},t} &= \frac{m_{\mathrm{norm},t}}{1-\gamma_1^{\,t}}, \qquad
\hat{v}_{\mathrm{norm},t} = \frac{v_{\mathrm{norm},t}}{1-\gamma_2^{\,t}} \\
\hat{g}_t &= \frac{g_t}{g_{\mathrm{norm}}}\cdot\frac{\hat{m}_{\mathrm{norm},t}}{\sqrt{\hat{v}_{\mathrm{norm},t}}+\epsilon}
\end{aligned}
$$

where $g_t$ is the gradient, $g_{\mathrm{norm}}$ its $\ell_2$ norm, $m_{\mathrm{norm}}$ and $v_{\mathrm{norm}}$ are scalar exponential moving averages of the norm and squared norm, $\gamma_1,\gamma_2$ their decay rates (defaults $0.7$ and $0.9$), $\hat{m}_{\mathrm{norm}},\hat{v}_{\mathrm{norm}}$ their bias-corrected values, $\epsilon$ a stability constant, and $\hat{g}_t$ the stabilized gradient passed to the subsequent Adam update.

Reference: Tianjin Huang, Haotian Hu, Zhenyu Zhang, Gaojie Jin, Xiang Li, Li Shen, Tianlong Chen, Lu Liu, Qingsong Wen, Zhangyang Wang, Shiwei Liu, "Stable-SPAM: How to Train in 4-Bit More Stably than 16-Bit Adam", ICML 2025. https://arxiv.org/abs/2502.17055

---
[Back to the Canon](../README.md)
