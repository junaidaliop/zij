# StoSignSGD

Implements StoSignSGD, a sign-based optimizer that injects unbiased structural stochasticity into the sign operator.

SignSGD discards gradient magnitude and is known to diverge on non-smooth objectives. StoSignSGD replaces the deterministic sign with a stochastic sign operator: each coordinate is perturbed by uniform noise scaled by a per-coordinate buffer before taking the sign, so that in expectation the update recovers an anisotropic, magnitude-aware (preconditioned SGD) direction while still transmitting only one bit per coordinate. The buffer $G_t$ tracks the running coordinate-wise maximum of the (momentum-smoothed) gradient magnitudes, setting the normalization level. The practical implementation adds heavy-ball momentum and decoupled weight decay.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
G_t &= \max\!\left(G_{t-1},\, |m_t|\right), \\
n_t &\sim \mathrm{Unif}[-1, 1]^d, \\
\theta_{t+1} &= \theta_t - \eta_t \, \mathrm{sign}\!\left(m_t + G_t \odot n_t\right) - \eta_t \lambda \theta_t.
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the stochastic gradient, $m_t$ is the momentum buffer with decay $\beta_1$, $G_t$ is the coordinate-wise max buffer, $n_t$ is uniform noise drawn independently per coordinate, $\odot$ is the elementwise product, $\eta_t$ is the learning rate, and $\lambda$ is the weight decay. The stochastic sign operator is $\mathrm{sign}(m_t + G_t \odot n_t)$, which is unbiased after coordinate-wise rescaling: $\mathbb{E}[\mathrm{sign}(x + G \odot n)] = x / G$.

Reference: Dingzhi Yu, Rui Pan, Yuxing Liu, Tong Zhang, "StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models", arXiv 2026. https://arxiv.org/abs/2604.15416

---
[Back to the Canon](../index.md)
