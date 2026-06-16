# DP-MicroAdam

Implements DP-MicroAdam, a differentially private and memory-efficient Adam variant built on MicroAdam's sparse error-feedback mechanism.

DP-MicroAdam combines DP-SGD-style privacy (per-sample $\ell_2$ clipping followed by Gaussian noise) with MicroAdam's frugal state: instead of storing full first and second moments, it keeps a ring buffer of the $m$ most recent top-$k$ sparse gradients and reconstructs bias-corrected moments on the fly. An error-feedback term, stored in a quantized buffer, accumulates the information dropped by sparsification and noise so the optimizer remains an unbiased compressor over time.

Each step clips and noises the per-sample gradients to form a private estimate $g_t$, adds back the decompressed error feedback, applies top-$k$ sparsification, re-quantizes the residual into the error buffer, then forms Adam moments from the sparse-gradient window and takes the update:

$$
\begin{aligned}
g_t &= \frac{1}{B}\left(\sum_{i=1}^{B}\mathrm{clip}\!\left(\nabla f(\theta_t, d_i),\, C\right) + \zeta_t\right), \quad \zeta_t \sim \mathcal{N}(0, \sigma^2 C^2 I) \\
a_t &= g_t + Q^{-1}(e_t) \\
(\mathcal{I}_t, \mathcal{V}_t) &= T_k(|a_t|), \qquad a_t[\mathcal{I}_t] \leftarrow 0 \\
e_{t+1} &= Q(a_t) \\
\hat{m}_t &= \frac{(1-\beta_1)}{1-\beta_1^{t}}\sum_{i} \beta_1^{\,r_i}\, \mathcal{V}_i, \qquad
\hat{v}_t = \frac{(1-\beta_2)}{1-\beta_2^{t}}\sum_{i} \beta_2^{\,r_i}\, \mathcal{V}_i^2 \\
\theta_{t+1} &= \theta_t - \eta_t\,\frac{\hat{m}_t}{\epsilon + \sqrt{\hat{v}_t}}
\end{aligned}
$$

where $\mathrm{clip}(g, C) = g\cdot\min(1, C/\lVert g\rVert_2)$ is the per-sample clip to threshold $C$, $\zeta_t$ is the privacy noise with multiplier $\sigma$, $B$ is the batch size, $Q$ and $Q^{-1}$ are the (de)quantization operators for the error-feedback buffer $e_t$, $T_k$ selects the $k$ largest-magnitude coordinates (indices $\mathcal{I}_t$, values $\mathcal{V}_t$), the moments are accumulated over the ring buffer of the last $m$ sparse gradients with per-slot age exponent $r_i$, $\beta_1,\beta_2$ are the decay rates, $\eta_t$ is the learning rate, and $\epsilon$ is the stability constant.

Reference: Mihaela Hudişteanu, Nikita P. Kalinin, Edwige Cyffers, "DP-MicroAdam: Private and Frugal Algorithm for Training and Fine-tuning", PPML Workshop at EurIPS 2025 / WiML Workshop at NeurIPS 2025. https://arxiv.org/abs/2511.20509

---
[Back to the Canon](../index.md)
