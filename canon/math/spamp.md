# SPAMP

Implements SPAMP, a per-layer gradient shaping rule that generalizes hard clipping into smooth, statistics-driven update-magnitude control.

SPAMP (Statistical Per-layer Adaptive Modulation and Projection) tracks an exponential moving average of each layer's gradient norm to form a dynamic threshold, then applies a sign-preserving power transform whose exponent depends on how far the current gradient sits above that threshold. Smaller exponents compress large gradients while leaving small ones nearly untouched, replacing the discontinuous behavior of clipping with a differentiable softening. A final norm-bounded projection caps the shaped gradient at the threshold before the parameter step.

$$
\begin{aligned}
\tau_t &= \beta\,\tau_{t-1} + (1-\beta)\,\lVert g_t \rVert \\
\alpha_t &= h\!\left(\frac{\lVert g_t \rVert}{\tau_t}\right) \\
\tilde{g}_t &= \mathrm{sign}(g_t)\,\lvert g_t \rvert^{\alpha_t} \\
\tilde{g}_t &\leftarrow \frac{\tau_t}{\lVert \tilde{g}_t \rVert}\,\tilde{g}_t \quad \text{if } \lVert \tilde{g}_t \rVert > \tau_t \\
\theta_{t+1} &= \theta_t - \eta\,\tilde{g}_t
\end{aligned}
$$

where the relations are applied per layer, $\tau_t$ is the per-layer dynamic threshold, $\beta \in [0.9, 0.999]$ smooths the gradient-norm estimate, $\alpha_t \in [0.7, 1.0]$ is the shaping exponent produced by a fixed decreasing function $h(\cdot)$ of the normalized gradient magnitude, $\mathrm{sign}$ and $\lvert\cdot\rvert^{\alpha_t}$ act elementwise, and $\eta$ is the learning rate.

Reference: Haochen You, Baojing Liu, "Gradient Shaping Beyond Clipping: A Functional Perspective on Update Magnitude Control", ACM Multimedia Asia 2025. https://arxiv.org/abs/2510.01578

---
[Back to the Canon](../README.md)
