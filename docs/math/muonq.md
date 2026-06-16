# MuonQ

Implements MuonQ, a low-bit quantized variant of Muon that preserves the orthogonal update direction under companding quantization.

MuonQ keeps Muon's polar-factor update but makes it memory-efficient by normalizing the momentum and quantizing a structural decomposition of it. The momentum is normalized to unit Frobenius norm, then split by power iteration into an orthonormal factor $U_t$, a small core $S_t$, and a residual $R_t$. Each piece is compressed with $\mu$-law companding followed by uniform $b$-bit quantization, which concentrates precision where the directional signal lives. The parameter update is the orthogonal polar factor of the normalized momentum, applied via Newton-Schulz iteration, so the step direction is preserved despite the low-bit storage.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + \frac{g_t}{\lVert g_t\rVert_F} \\
\bar{m}_t &= \frac{m_t}{\lVert m_t\rVert_F} \\
U_t &= \mathrm{orth}\!\left(\bar{m}_t V_{t-1}^{\top}\right), \quad S_t = U_t^{\top}\bar{m}_t, \quad R_t = \bar{m}_t - U_t S_t \\
\theta_t &= \theta_{t-1} - \eta\, \mathrm{polar}(\bar{m}_t)
\end{aligned}
$$

where $\theta$ are the (matrix-shaped) parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $\bar{m}_t$ its Frobenius-normalized form, $\beta$ the momentum decay, and $\lVert\cdot\rVert_F$ the Frobenius norm; $\mathrm{orth}(\cdot)$ orthonormalizes its argument and $\mathrm{polar}(\cdot)$ returns the orthogonal polar factor via Newton-Schulz iteration (coefficients $(a,b,c)=(3.4445,-4.7750,2.0315)$); $U_t, S_t, R_t$ are the orthonormal, core, and residual factors that are stored after $b$-bit companding quantization $\mathrm{CQuant}_b$.

Reference: Yupeng Su, Ruijie Zhang, Ziyue Liu, Yequan Zhao, Zheng Zhang, "MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization", arXiv 2026. https://arxiv.org/abs/2605.11396

---
[Back to the Canon](../index.md)
