# DOPPLER

Implements DOPPLER, a differentially private optimizer that applies a low-pass filter to privatized gradients to suppress injected DP noise.

In differentially private training, each per-sample gradient is clipped to a norm bound and isotropic Gaussian noise is added to guarantee privacy, which degrades the signal-to-noise ratio of the update. DOPPLER observes that the useful gradient signal concentrates in low frequencies while the DP noise is broadband, so it passes the noisy gradient through a linear recursive (IIR) low-pass filter parameterized by coefficients $\{a_\tau\}$ and $\{b_\tau\}$. A scalar bias-correction term keeps the filtered moment unbiased at the start of training, after which the parameters take a plain gradient step on the filtered direction.

$$
\begin{aligned}
g_t &= \frac{1}{B}\sum_{\xi_i \in \mathcal{B}_t} \mathrm{clip}\!\left(\nabla f(\theta_t;\xi_i),\, C\right) + w_t, \qquad w_t \sim \mathcal{N}(0,\sigma_{\mathrm{DP}}^2 I), \\
\mathrm{clip}(v, C) &= \min\!\left\{1,\, \frac{C}{\lVert v \rVert}\right\} v, \\
m_t &= -\sum_{\tau=1}^{n_a} a_\tau\, m_{t-\tau} + \sum_{\tau=0}^{n_b} b_\tau\, g_{t-\tau}, \\
c_t &= -\sum_{\tau=1}^{n_a} a_\tau\, c_{t-\tau} + \sum_{\tau=0}^{n_b} b_\tau, \\
\hat{m}_t &= \frac{m_t}{c_t}, \\
\theta_{t+1} &= \theta_t - \eta\, \hat{m}_t.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta$ the learning rate, $g_t$ the privatized minibatch gradient over batch $\mathcal{B}_t$ of size $B$, $C$ the per-sample clipping bound, $\sigma_{\mathrm{DP}}$ the DP noise scale, $\{a_\tau\},\{b_\tau\}$ the filter coefficients with orders $n_a,n_b$, $m_t$ the filtered gradient, and $c_t$ the scalar bias-correction factor (with $c_t = 0$ for $t \le 0$).

Reference: Xinwei Zhang, Zhiqi Bu, Mingyi Hong, Meisam Razaviyayn, "DOPPLER: Differentially Private Optimizers with Low-pass Filter for Noise Reduction", arXiv 2024. https://arxiv.org/abs/2408.13460

---
[Back to the Canon](../README.md)
