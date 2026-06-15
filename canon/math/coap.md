# COAP

Implements COAP, Adam in a low-rank subspace defined by a correlation-aware gradient projection.

COAP cuts optimizer-state memory by projecting each gradient $g_t$ down to a low-rank space, running the Adam moment updates entirely in that compact space, then projecting the resulting step back to full rank. Unlike methods that recompute the projection by full SVD at every interval, COAP refreshes the projection matrix $P_t$ cheaply with a correlation-aware objective that ties the new subspace to the prior first moment, so the trajectory stays smooth and abrupt subspace switches are avoided; a low-cost SVD is applied only occasionally to recalibrate.

$$
\begin{aligned}
g_t^{\mathrm{proj}} &= g_t P_t \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t^{\mathrm{proj}} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, (g_t^{\mathrm{proj}})^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\Delta_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \\
\theta_t &= \theta_{t-1} - \eta\, \Delta_t P_t^{\top}
\end{aligned}
$$

where $g_t \in \mathbb{R}^{m \times n}$ is the gradient, $P_t \in \mathbb{R}^{n \times r}$ is the rank-$r$ projection matrix, $g_t^{\mathrm{proj}}$ and the moments $m_t, v_t$ live in the low-rank space $\mathbb{R}^{m \times r}$, $\eta$ is the learning rate, $\beta_1, \beta_2$ are the moment decay rates, and $\epsilon$ is the stability constant. $P_t$ is updated by minimizing $\mathrm{MSE}(\hat{g}, g)\,(1 - \mathrm{CosSim}(\hat{m}, g))$ with $\hat{g} = g P P^{\top}$ and $\hat{m} = m^{\mathrm{proj}} P^{\top}$, and is periodically recalibrated via a reduced QR plus SVD of $g_t P_{t-1}$.

Reference: Jinqi Xiao, Shen Sang, Tiancheng Zhi, Jing Liu, Qing Yan, Linjie Luo, Bo Yuan, "COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection", arXiv 2024. https://arxiv.org/abs/2412.00071

---
[Back to the Canon](../README.md)
