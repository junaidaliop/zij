# AlphaGrad

Implements AlphaGrad, a memory-efficient optimizer that L2-normalizes each layer's gradient and applies a tanh smooth clipping controlled by a steepness parameter.

The raw gradient of each parameter group (layer) is first normalized to unit scale by its own L2 norm, removing magnitude information and making the update invariant to gradient scale. The normalized gradient is then passed through $\tanh(\alpha \cdot \tilde g_t)$, which bounds every component to $(-1, 1)$: small $\alpha$ keeps the response nearly linear, while large $\alpha$ saturates into a sign-like clip. The descent step uses this bounded direction, optionally with classical momentum.

$$
\begin{aligned}
\tilde g_t &= \frac{g_t}{\lVert g_t \rVert_2 + \epsilon} \\
g_t' &= \tanh(\alpha \, \tilde g_t) \\
v_t &= \gamma \, v_{t-1} + \eta \, g_t' \\
\theta_t &= \theta_{t-1} - v_t
\end{aligned}
$$

where $g_t$ is the per-layer gradient, $\lVert \cdot \rVert_2$ its L2 norm, $\epsilon$ a stability constant, $\alpha$ the layer-wise steepness parameter, $\eta$ the learning rate, $\gamma$ the momentum factor, and $v_t$ the velocity. Without momentum, set $\gamma = 0$ so that $\theta_t = \theta_{t-1} - \eta \, g_t'$.

Reference: Soham Sane, "AlphaGrad: Non-Linear Gradient Normalization Optimizer", arXiv 2025. https://arxiv.org/abs/2504.16020

---
[Back to the Canon](../index.md)
