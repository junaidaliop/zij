# Rose

Implements Rose (Range-Of-Slice Equilibration), a stateless optimizer that scales each gradient slice by its own value range.

Rose holds no moment buffers. For every parameter matrix it treats the leading axis as the "neuron" axis and reduces over the remaining axes, computing a per-slice range $R = |\max(g)| - \min(g)$ that acts as a local, online preconditioner. When these per-slice ranges disagree across neurons the optimizer trusts them less: a trust gate $\tau$, formed from the mean and spread of the ranges, interpolates the denominator between each slice's local range (full detail) and the global mean range (maximum noise resistance). Weight decay is decoupled and coupled to the learning-rate schedule, and degenerate tensors fall back to signed/range-normalized SGD.

$$
\begin{aligned}
g_t &= g_t - \mathrm{mean}_{\text{slice}}(g_t) \quad (\text{optional centralization})\\
R &= |\max_{\text{slice}}(g_t)| - \min_{\text{slice}}(g_t)\\
\mu &= \mathrm{mean}(R), \qquad \sigma = \mathrm{std}(R)\\
\tau &= \frac{\mu}{\sigma + \mu}\\
d &= \mu + \tau\,(R - \mu)\\
\theta_t &= \bigl(1 - \tfrac{\eta_t}{\eta_{\mathrm{ref}}}\,\lambda\bigr)\,\theta_{t-1} - \eta_t\,\frac{g_t}{d}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate (with reference value $\eta_{\mathrm{ref}}$), $g_t$ the gradient, $\lambda$ the weight decay; the max/min/mean reductions run over all axes except the leading one, $\mu,\sigma$ are taken over the per-slice ranges $R$, $\tau$ is the trust gate, and $d$ is the blended denominator (zeros are replaced by $1$, giving an SGD fallback).

Reference: Matthew E. Kieren, "Rose: Range-Of-Slice Equilibration optimizer", Zenodo 2026. https://doi.org/10.5281/zenodo.19589764

---
[Back to the Canon](../index.md)
