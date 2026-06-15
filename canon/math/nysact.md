# NysAct

Implements NysAct, a scalable preconditioned gradient method built on a Nyström approximation of the layerwise activation covariance.

For each layer $l$ NysAct maintains an exponential moving average of a sketched activation covariance $A_t S$, where the sketch $S$ (uniform column sampling or Gaussian) projects the $d_{l-1}\times d_{l-1}$ covariance down to a thin $d_{l-1}\times r$ matrix. A damped, eigenvalue-shifted Nyström factorization of this sketch yields a positive-definite preconditioner whose inverse $C_{\mathrm{nys}}^{-1}$ is applied to the gradient. Working in the sketched $r$-dimensional space keeps the cost linear in the layer width while still capturing curvature, giving second-order-style preconditioning at near first-order memory.

$$
\begin{aligned}
\tilde{C}_t &= \beta_2\,\tilde{C}_{t-1} + (1-\beta_2)\,A_t S, \\
\hat{C}_t &= \tilde{C}_t / \bigl(1-\beta_2^{\lfloor t/\tau\rfloor}\bigr) + \rho\,S, \\
C_{\mathrm{nys},t}^{-1} &= U\,\tilde{\Sigma}^{-1}U^{\top} + \tfrac{1}{\rho}\bigl(I - U U^{\top}\bigr), \\
m_t &= \beta_1\,m_{t-1} - \eta\,\mathrm{vec}\!\bigl(g_t\,C_{\mathrm{nys},t}^{-1}\bigr), \\
\theta_t &= \theta_{t-1} + m_t,
\end{aligned}
$$

where $A_t$ is the layer activation matrix, $S$ the random sketch of rank $r$, $\beta_2$ the covariance EMA decay with update period $\tau$, $\rho$ the damping factor, $U,\tilde{\Sigma}$ the eigenvalue-shifted Nyström factors of the damped sketch $\hat{C}_t$ (so $C_{\mathrm{nys},t}$ is symmetric positive definite), $g_t$ the layer gradient, $\eta$ the learning rate, and $\beta_1$ the momentum coefficient.

Reference: Hyunseok Seung, Jaewoo Lee, Hyunsuk Ko, "NysAct: A Scalable Preconditioned Gradient Descent using Nyström Approximation", IEEE BigData 2024 (extended version, arXiv 2025). https://arxiv.org/abs/2506.08360

---
[Back to the Canon](../README.md)
