# Conda

Implements Conda (Column-Normalized Adam), a subspace optimizer that combines Muon-style orthogonal projection with Adam-style coordinate-wise adaptivity.

The first moment $m_t$ is accumulated as in Adam, then periodically an SVD of $m_t$ supplies a left-singular basis $U_t$ that defines a low-dimensional column subspace, refreshed only every $T$ steps and reused in between. Both the momentum and the raw gradient are projected into this subspace, and a second moment $v_t$ is maintained on the projected gradient. Normalizing the projected momentum column-wise by $\sqrt{v_t}$ recovers Adam's per-coordinate adaptivity inside the conditioned subspace before mapping the update back with $U_t$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
U_t &= \mathrm{SVD}_U(m_t) \quad \text{if } t \bmod T = 0,\; \text{else } U_t = U_{t-1} \\
m_t' &= U_t^\top m_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,(U_t^\top g_t)^2 \\
\hat{m}_t' &= \frac{m_t'}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\, U_t \frac{\hat{m}_t'}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

where $\theta$ are the (matrix-shaped) parameters, $g_t$ the gradient, $\eta$ the learning rate, $m_t$/$v_t$ the first and second moments, $\beta_1,\beta_2$ the decay rates, $\epsilon$ the stability constant, $U_t$ the left singular vectors of $m_t$ refreshed every $T$ subspace-update steps, and $(\cdot)^2$, division, and $\sqrt{\cdot}$ are element-wise.

Reference: Junjie Wang, Pan Zhou, Yiming Dong, Huan Li, Jia Li, Xun Zhou, Qicheng Lao, Cong Fang, Zhouchen Lin, "Conda: Column-Normalized Adam for Training Large Language Models Faster", arXiv 2025. https://arxiv.org/abs/2509.24218

---
[Back to the Canon](../index.md)
