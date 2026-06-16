# SMA-DP-SGD

Implements SMA-DP-SGD, a differentially private optimizer that augments DP-SGD with a spectral, memory-aware release built from past privatized updates.

Within each parameter group, per-example gradients are clipped to norm $C$ and summed over a Poisson subsample, exactly as in DP-SGD. Instead of releasing this sum directly, the method mixes it with a fractional memory state $\nu_{t-1}$ assembled from previously privatized releases, weighted with a power-law decay of fractional order $\alpha$ and a spectral tempering factor. The memory contribution is gated by its cosine alignment $\Gamma_t$ with the private trend $\mu_{t-1}$ and rescaled by $\Psi_t$ to match that trend's norm, then Gaussian noise is added so only privatized quantities are ever reused. Setting $\beta = 1$ recovers standard group-wise DP-SGD.

$$
\begin{aligned}
\bar g_t(x_i) &= g_t(x_i) \big/ \max\!\big(1, \lVert g_t(x_i)\rVert_2 / C\big), \qquad s_t = \sum_{x_i \in S_t} \bar g_t(x_i) \\
a_{t,j} &= (j+1)^{\alpha-1}\, e^{-\lambda_t j}, \qquad \hat a_{t,j} = a_{t,j} \Big/ \sum_{\ell=1}^{M_t} a_{t,\ell}, \qquad \nu_{t-1} = \sum_{j=1}^{M_t} \hat a_{t,j}\, \tilde s_{t-j} \\
\Gamma_t &= \max\!\Big(0,\ \tfrac{\langle \mu_{t-1}, \nu_{t-1}\rangle}{\lVert \mu_{t-1}\rVert_2\, \lVert \nu_{t-1}\rVert_2 + \epsilon}\Big), \qquad \Psi_t = \min\!\Big(\xi_{\max},\ \tfrac{\lVert \mu_{t-1}\rVert_2}{\lVert \nu_{t-1}\rVert_2 + \epsilon}\Big) \\
r_t &= \beta\, s_t + (1-\beta)\, \omega_t\, \Gamma_t\, \Psi_t\, \nu_{t-1}, \qquad \tilde s_t = r_t + Z_t, \quad Z_t \sim \mathcal{N}\!\big(0, \sigma^2 C^2 I\big) \\
\theta_{t+1} &= \theta_t - \eta\, \tilde s_t / L
\end{aligned}
$$

where $g_t(x_i)$ is the per-example gradient, $C$ the clipping norm, $s_t$ the clipped subsample sum over $S_t$, $\alpha \in (0,1]$ the fractional order, $\lambda_t = 1 - e^{-c_\lambda d_t}$ the spectral tempering factor with deviation $d_t = \max(0, \rho_{\min}-\rho_t, \rho_t-\rho_{\max})$ of the layer spectral exponent $\rho_t$, $M_t$ the memory length over privatized releases $\tilde s_{t-j}$, $\mu_{t-1}$ the private exponential trend, $\omega_t = 1 - e^{-t/\tau}$ a warm-up coefficient, $\beta \in (0,1]$ the memory mixing weight, $\xi_{\max}$ the norm-matching cap, $\sigma$ the noise multiplier, $L$ the batch size, $\eta$ the learning rate, and $\epsilon$ a stability constant.

Reference: Mohammad Partohaghighi, Roummel P. Marcia, "SMA-DP: Spectral Memory-Aware Differential Privacy for Deep Learning", arXiv 2026. https://arxiv.org/abs/2605.20450

---
[Back to the Canon](../index.md)
