# PowerSGD

Implements PowerSGD, low-rank gradient compression via one step of power iteration for communication-efficient distributed SGD.

PowerSGD reshapes each gradient tensor into a matrix $M$ and approximates it by a rank-$r$ factorization $\hat{M} = P Q^\top$, computed with a single subspace-iteration step that reuses the right factor $Q$ across optimizer steps as a warm start. Only the low-rank factors are exchanged: $P$ is aggregated across workers with all-reduce, then orthogonalized, and $Q$ is recomputed and all-reduced. To stay unbiased over time, the compression residual is kept in an error buffer $e_t$ that is added back to the gradient before the next compression.

$$
\begin{aligned}
\Delta_t &= g_t + e_t \\
P &= \mathrm{all\_reduce\_mean}(\Delta_t\, Q) \\
\hat{P} &= \mathrm{orthogonalize}(P) \\
Q &= \mathrm{all\_reduce\_mean}(\Delta_t^\top \hat{P}) \\
\Delta_t' &= \hat{P} Q^\top \\
e_t &= \Delta_t - \Delta_t' \\
m_t &= \lambda\, m_{t-1} + \Delta_t' \\
\theta_t &= \theta_{t-1} - \gamma\,(\Delta_t' + m_t)
\end{aligned}
$$

where $g_t$ is the local gradient reshaped to a matrix, $e_t$ the error-feedback buffer, $Q$ the right factor carried over between steps, $\hat{P} Q^\top$ the decompressed (all-reduced) gradient, $\lambda$ the momentum factor, $\gamma$ the learning rate, and $\theta$ the parameters.

Reference: Thijs Vogels, Sai Praneeth Karimireddy, Martin Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019. https://arxiv.org/abs/1905.13727

---
[Back to the Canon](../README.md)
