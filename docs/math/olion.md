# OLion

Implements OLion, an optimizer that intersects the spectral bias of Muon with the $\ell_\infty$ bias of Lion.

OLion forms a Nesterov-mixed momentum, orthogonalizes it through Newton-Schulz iteration to obtain a spectral (semi-orthogonal) direction, then applies an entrywise sign to push the update toward the $\ell_\infty$ ball. The signed matrix is rescaled by an RMS-alignment factor so its magnitude matches a unit-RMS step, and the parameters are updated with decoupled weight decay.

$$
\begin{aligned}
m_t &= \beta_2 m_{t-1} + (1 - \beta_2) g_t \\
\tilde{g}_t &= (1 - \beta_1) g_t + \beta_1 m_t \\
q_t &= \mathrm{NewtonSchulz}(\tilde{g}_t, K) \\
s_t &= \mathrm{sign}(q_t) \\
\gamma_t &= 0.2 \cdot \frac{\sqrt{d_1 d_2}}{\lVert s_t \rVert_F} \\
\theta_{t+1} &= \theta_t - \eta_t \gamma_t s_t - \lambda \eta_t \theta_t
\end{aligned}
$$

where $\theta$ are the parameters (a $d_1 \times d_2$ matrix), $\eta_t$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $\beta_1$ the Nesterov-mixing weight, $\beta_2$ the momentum decay, $\lambda$ the weight decay, $K$ the number of Newton-Schulz iterations, $\mathrm{sign}$ the entrywise sign, and $\gamma_t \approx 0.2$ the RMS-alignment scale ($\lVert \cdot \rVert_F$ is the Frobenius norm).

Reference: Zixiao Wang, Yifei Shen, Huishuai Zhang, "OLion: Approaching the Hadamard Ideal by Intersecting Spectral and $\ell_\infty$ Implicit Biases", arXiv 2025. https://arxiv.org/abs/2602.01105

---
[Back to the Canon](../index.md)
