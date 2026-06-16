# FOGO

Implements FOGO, an orthogonalized-momentum optimizer that suppresses catastrophic forgetting by detecting and correcting gradient interference.

FOGO extends Muon-style spectral orthogonalization with two momentum streams: a slow buffer (decay $\beta_s$) that holds persistent update directions and a fast buffer (decay $\beta_f$) that tracks recent change, with $\beta_s > \beta_f$. Each stream is orthogonalized by a Newton-Schulz iteration, then fused by spherical interpolation so that dominant mini-batch gradients no longer overwrite rare-but-useful directions. The fused update is rescaled to a fixed root-mean-square magnitude before being applied with decoupled weight decay.

$$
\begin{aligned}
m_t^{(s)} &= \beta_s\, m_{t-1}^{(s)} + g_t, &\quad m_t^{(f)} &= \beta_f\, m_{t-1}^{(f)} + g_t \\
O_t^{(s)} &= \mathrm{NewtonSchulz}\big(m_t^{(s)}\big), &\quad O_t^{(f)} &= \mathrm{NewtonSchulz}\big(m_t^{(f)}\big) \\
\hat{O}_t &= \mathrm{Slerp}\big(O_t^{(s)}, O_t^{(f)};\, \xi\big) \\
\gamma_t &= \sigma\,\frac{\sqrt{mn}}{\lVert \hat{O}_t \rVert_F + \epsilon} \\
\theta_t &= (1 - \eta\lambda)\,\theta_{t-1} - \eta\,\gamma_t\,\hat{O}_t
\end{aligned}
$$

where $g_t$ is the gradient, $m_t^{(s)}, m_t^{(f)}$ are the slow and fast momentum buffers with decays $\beta_s > \beta_f$, $\mathrm{NewtonSchulz}(\cdot)$ is the iterative orthogonalization (approximate polar factor) used by Muon, $\mathrm{Slerp}(\cdot;\xi)$ is spherical interpolation with mixing weight $\xi$, $\sigma$ is a target scale and $\sqrt{mn}$ the dimensions of the $m \times n$ parameter matrix, $\lVert\cdot\rVert_F$ the Frobenius norm, $\eta$ the learning rate, $\lambda$ the weight decay, and $\epsilon$ a small constant. A random-projection codebook stores past orthogonalized directions and adds a proximal correction to $\hat{O}_t$ to resolve interference with previously learned tasks.

Reference: Toan Nguyen, Yang Liu, Trung Le, Celso De Melo, Flora D. Salim, "FOGO: Forgetting-aware Orthogonalization Optimizer", arXiv 2026. https://arxiv.org/abs/2606.10406

---
[Back to the Canon](../index.md)
