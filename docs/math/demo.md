# DeMo

Implements DeMo, decoupled momentum optimization that compresses the synchronized update so each worker transmits only a sparse fast-moving component.

DeMo keeps a decoupled local momentum buffer that is allowed to drift across accelerators. Each step it accumulates the learning-rate-scaled gradient into the buffer, then extracts the fast-moving components via a fixed orthonormal transform (a chunked DCT) followed by top-$k$ sparsification. The extracted component is subtracted from the buffer, so what remains acts as an error-feedback residual that is carried forward; only the sparse top-$k$ amplitudes and indices are all-gathered across workers. The synchronized sparse coefficients are decoded with the inverse transform to form the global update, and parameters are stepped with the sign of that update (sign-SGD).

$$
\begin{aligned}
m_t &= \beta \, m_{t-1} + \eta \, g_t \\
q_t &= \mathrm{TopK}_k\big(\mathrm{DCT}(m_t)\big) \\
m_t &\leftarrow m_t - \mathrm{IDCT}(q_t) \\
Q_t &= \mathrm{AllGather}(q_t) \\
u_t &= \mathrm{IDCT}(Q_t) \\
\theta_t &= \theta_{t-1} - \eta \, \mathrm{sign}(u_t)
\end{aligned}
$$

where $m_t$ is the decoupled momentum buffer (initialized to $0$), $\beta$ is the momentum/compression decay, $\eta$ is the learning rate, $g_t$ is the local gradient, $\mathrm{DCT}/\mathrm{IDCT}$ are the chunked discrete cosine transform and its inverse, $\mathrm{TopK}_k$ keeps the $k$ largest-magnitude coefficients (zeroing the rest), $q_t$ is the local sparse fast component, $Q_t$ aggregates the sparse coefficients gathered from all workers, and $u_t$ is the reconstructed global update. Decoupled weight decay $\lambda$ is applied multiplicatively as $\theta \leftarrow (1 - \eta\lambda)\theta$ before the momentum update.

Reference: Bowen Peng, Jeffrey Quesnelle, Diederik P. Kingma, "DeMo: Decoupled Momentum Optimization", ICLR 2025. https://arxiv.org/abs/2411.19870

---
[Back to the Canon](../index.md)
