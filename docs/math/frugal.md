# FRUGAL

Implements FRUGAL, full-rank training that splits each gradient into a low-dimensional state-full subspace and its state-free complement.

FRUGAL is a memory-efficient framework rather than a single rule. For every parameter, a projector $P_k$ selects a subspace (a block of columns, an SVD-rank-$r$ basis, or a random subset of coordinates, refreshed every $T$ steps). The projected part is optimized by a stateful method that keeps moments only on that small subspace, while the complementary residual is updated by a stateless method, so the overall step stays full-rank without storing full-size moments. The default instantiation uses Adam on the state-full part and signSGD on the state-free part.

For the default FRUGAL(Adam, signSGD) with $g_t=\nabla\mathcal L(\theta_t)$, projection $g^{\mathrm{full}}_t=P_t(g_t)$, residual $g^{\mathrm{free}}_t=g_t-P_t^{-1}(g^{\mathrm{full}}_t)$, and moments carried in the projected coordinates:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g^{\mathrm{full}}_t,\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,(g^{\mathrm{full}}_t)^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}},\\
u^{\mathrm{full}}_t &= -\,\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\qquad
u^{\mathrm{free}}_t = -\,\eta\,\mathrm{sign}\!\left(g^{\mathrm{free}}_t\right),\\
\theta_{t+1} &= \theta_t + P_t^{-1}\!\left(u^{\mathrm{full}}_t\right) + u^{\mathrm{free}}_t.
\end{aligned}
$$

where $P_t$ is the projector onto the state-full subspace at step $t$ (so $P_t^{-1}$ embeds back to the full space), $g^{\mathrm{full}}_t$ and $g^{\mathrm{free}}_t$ are the projected and residual gradients, $m_t,v_t$ are the first and second moments kept only on the state-full coordinates, $\eta$ is the learning rate, $\beta_1,\beta_2$ are the Adam decays, $\epsilon$ is the stability constant, and $\mathrm{sign}$ applies signSGD to the complement. When the projector is refreshed, the stored state is re-projected as $s\leftarrow P_t(P_{t-1}^{-1}(s))$. Density $\rho$ controls the fraction of coordinates in the state-full subspace, recovering Adam at $\rho=1$ and signSGD at $\rho=0$.

Reference: Philip Zmushko, Aleksandr Beznosikov, Martin Takáč, Samuel Horváth, "FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training", arXiv 2024. https://arxiv.org/abs/2411.07837

---
[Back to the Canon](../index.md)
