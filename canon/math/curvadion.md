# CurvaDion

Implements CurvaDion, a curvature-adaptive variant of Dion that gates distributed orthonormalization on a momentum-change signal.

Dion maintains a momentum buffer and updates parameters with an orthonormalized low-rank factorization of that buffer, obtained by power iteration: $M_t \approx P_t R_t^\top$ with $P_t$ orthogonalized and $Q_t$ the column-normalized factor, applied with the spectral scaling $\sqrt{m/n}$. CurvaDion observes that the expensive synchronization (all-reduce plus orthogonalization across workers) is only worthwhile in high-curvature regions. It tracks the relative maximum momentum change per layer, $\mathrm{RMMC}_\ell(t)$, and triggers a full synchronized Dion step only when the global maximum exceeds a threshold $\tau$; otherwise each worker takes a cheap local gradient step.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + g_t, \\
\mathrm{RMMC}_\ell(t) &= \frac{\bigl|\,\lVert m_{\ell,t}\rVert - \lVert m_{\ell,t-1}\rVert\,\bigr|}{\lVert m_{\ell,t-1}\rVert}, \\
P_t, R_t &= \mathrm{PowerIter}(m_t;\, Q_{t-1}), \quad P_t = \mathrm{Orthogonalize}(P_t), \quad Q_t = \mathrm{ColumnNormalize}(R_t), \\
\theta_t &=
\begin{cases}
\theta_{t-1} - \eta\,\sqrt{m/n}\; P_t Q_t^\top, & \max_\ell \mathrm{RMMC}_\ell(t) > \tau, \\
\theta_{t-1} - \eta_{\mathrm{local}}\, g_t, & \max_\ell \mathrm{RMMC}_\ell(t) \le \tau.
\end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate (with $\eta_{\mathrm{local}}$ used for the cheap local step), $g_t$ the gradient, $m_t$ the momentum buffer with coefficient $\mu$, $P_t/R_t/Q_t$ the low-rank factors from power iteration, $\sqrt{m/n}$ the shape-dependent spectral scaling for an $m \times n$ matrix, and $\tau$ the curvature-synchronization threshold on the relative maximum momentum change $\mathrm{RMMC}_\ell$.

Reference: Anonymous Authors, "CurvaDion: Curvature-Adaptive Distributed Orthonormalization", MLSys 2026 (under review). https://arxiv.org/abs/2512.13728

---
[Back to the Canon](../README.md)
