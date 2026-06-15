# Pion

Implements Pion, a drop-in replacement for Muon that swaps uniform spectral whitening for a two-stage high-pass Promotion+Suppression spectral filter.

Pion targets settings where Muon's uniform whitening of the momentum buffer hurts, such as vision-language-action training and reinforcement learning. It keeps Muon's structure of momentum followed by a matrix-sign step $\mathrm{msign}(\cdot)$ computed by Newton-Schulz iterations, but redefines $\mathrm{msign}$ as a composed high-pass filter. After Frobenius normalization, a promotion polynomial is iterated $k_p$ times to lift the dominant singular directions, then a suppression polynomial is iterated $k_s = k - k_p$ times to damp small (noise) singular values, reshaping each singular value $\sigma \in [0,1]$ rather than flattening the whole spectrum to one.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + g_t \\
X_0 &= \frac{m_t}{\|m_t\|_F + \epsilon} \\
X &\leftarrow a_p X + b_p\, X X^\top X + c_p\, X (X^\top X)^2 \quad (k_p \text{ times}) \\
X &\leftarrow a_s X + b_s\, X X^\top X + c_s\, X (X^\top X)^2 \quad (k_s \text{ times}) \\
\theta_t &= \theta_{t-1} - \eta\, \mathrm{msign}(m_t)
\end{aligned}
$$

where $\mathrm{msign}(m_t)$ is the result $X$ of the iterations started from $X_0$, the promotion coefficients are $(a_p, b_p, c_p) = (1.875, -1.25, 0.375)$ giving $f_p(\sigma) = 1.875\,\sigma - 1.25\,\sigma^3 + 0.375\,\sigma^5$, the suppression coefficients are $(a_s, b_s, c_s) = (0, 2.5, -1.5)$ giving $f_s(\sigma) = 2.5\,\sigma^3 - 1.5\,\sigma^5$, $g_t$ is the gradient, $m_t$ the momentum buffer, $\mu$ the momentum coefficient, $\eta$ the learning rate, $\epsilon$ a small constant, and $\theta$ the matrix-shaped parameters.

Reference: Chongyu Fan, Gaowen Liu, Mingyi Hong, Ramana Rao Kompella, Sijia Liu, "Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR", arXiv 2026. https://arxiv.org/abs/2605.19282

---
[Back to the Canon](../README.md)
