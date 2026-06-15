# Aida

Implements Aida, an AdaBelief variant that suppresses the adaptive stepsize range via layerwise mutual vector projections.

Aida follows AdaBelief's idea of tracking the variance of the gradient about its momentum, but first refines the momentum $m_t$ and gradient $g_t$ within each layer through $K$ rounds of mutual projection. Each round projects one vector onto the other so the two progressively align in direction; replacing the raw $(m_t - g_t)^2$ belief term with $(m_t^{(K)} - g_t^{(K)})^2$ narrows the spread of the second moment $v_t$ and thus the range of the effective per-coordinate stepsize. Setting $K=0$ recovers AdaBelief.

For each layer, with $(m_t^{(0)}, g_t^{(0)}) = (m_t, g_t)$, the update is

$$
\begin{aligned}
g_t &= \nabla f(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
m_t^{(k)} &= \frac{\langle g_t^{(k-1)}, m_t^{(k-1)} \rangle}{\lVert g_t^{(k-1)} \rVert_2^2 + \xi}\, g_t^{(k-1)}, \quad k = 1,\dots,K \\
g_t^{(k)} &= \frac{\langle g_t^{(k-1)}, m_t^{(k-1)} \rangle}{\lVert m_t^{(k-1)} \rVert_2^2 + \xi}\, m_t^{(k-1)}, \quad k = 1,\dots,K \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\bigl(m_t^{(K)} - g_t^{(K)}\bigr)^2 + \epsilon \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first and second moments, $\beta_1,\beta_2$ the decay rates, $\epsilon$ a stability constant, $\xi$ a small constant guarding the projection denominators, the projections run per layer, and $K$ is the number of mutual projection iterations ($K=2$ in practice).

Reference: Guoqiang Zhang, Kenta Niwa, W. Bastiaan Kleijn, "A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range", arXiv 2022. https://arxiv.org/abs/2203.13273

---
[Back to the Canon](../README.md)
