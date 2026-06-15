# AdaGO

Implements AdaGO, an adaptive-stepsize variant of Muon that scales orthogonalized momentum by an AdaGrad-style gradient-norm accumulator.

AdaGO keeps Muon's orthogonalized update direction but replaces the fixed step with an adaptive one. It accumulates squared gradient norms, clamped by a constant $\gamma$ to bound the influence of large gradients, and divides the learning rate by the resulting accumulator. The update direction $O_t$ is obtained by orthogonalizing the momentum: if $M_t = U\Sigma V^\top$ is the reduced SVD, then $\mathrm{Orth}(M_t) = UV^\top$ (in practice approximated by Newton–Schulz iterations).

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + (1-\mu)\, g_t \\
v_t^2 &= v_{t-1}^2 + \min\{\|g_t\|^2,\ \gamma^2\} \\
o_t &= \mathrm{Orth}(m_t) \\
\alpha_t &= \max\!\left\{\epsilon,\ \frac{\eta\,\min\{\|g_t\|,\ \gamma\}}{v_t}\right\} \\
\theta_t &= \theta_{t-1} - \alpha_t\, o_t
\end{aligned}
$$

where $\theta$ are the parameters (a matrix), $\eta$ the base learning rate, $g_t$ the gradient, $m_t$ the momentum with decay $\mu$, $v_t = \sqrt{v_t^2}$ the accumulated clamped gradient norm, $\gamma$ the clamping constant, $\mathrm{Orth}(\cdot)$ the orthogonal polar factor, $\alpha_t$ the adaptive stepsize, and $\epsilon$ a stability floor.

Reference: Minxin Zhang, Yuxuan Liu, Hayden Schaeffer, "AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates", arXiv 2025. https://arxiv.org/abs/2509.02981

---
[Back to the Canon](../README.md)
