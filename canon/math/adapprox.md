# Adapprox

Implements Adapprox, an Adam variant that stores the second moment as a randomized low-rank approximation to cut memory.

Adapprox keeps the full second-moment matrix $V_t$ only implicitly: at each step it reconstructs the previous estimate from low-rank factors $Q_{t-1}U_{t-1}^{T}$, applies the usual exponential decay against $g_t^2$, then re-factors the result via adaptive streamlined randomized subspace iteration (AS-RSI) into new factors $Q_t, U_t$ of adaptively chosen rank. The denominator $\sqrt{V_t}$ used in the update is formed from this approximation, so only the factors are stored rather than the dense matrix.

The update otherwise follows Adam with decoupled weight decay: it omits Adam's bias-correction terms, applies an RMS clip to the per-step direction, and treats the first moment (when $\beta_1>0$) as a running average of the clipped update rather than of the raw gradient.

$$
\begin{aligned}
V_t &= \beta_2\, Q_{t-1}U_{t-1}^{T} + (1-\beta_2)\, g_t^2 \\
Q_t, U_t &= \mathrm{AS\text{-}RSI}(V_t) \\
\hat m_t &= \frac{g_t}{\sqrt{V_t}+\epsilon} \\
\hat m_t &\leftarrow \frac{\hat m_t}{\max\!\left(1,\ \mathrm{RMS}(\hat m_t)/d\right)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\hat m_t \quad (\text{if } \beta_1>0) \\
\theta_t &= \theta_{t-1} - \eta\,(m_t + \lambda\,\theta_{t-1})
\end{aligned}
$$

where $V_t \approx Q_t U_t^{T}$ is the low-rank second moment with factors $Q_t,U_t$ of adaptive rank from AS-RSI, $g_t$ is the gradient, $\hat m_t$ the clipped update direction, $\mathrm{RMS}(x)=\lVert x\rVert_F/\sqrt{mn}$ for an $m\times n$ matrix, $d$ the clip threshold, $\eta$ the learning rate, $\beta_1,\beta_2$ the decay rates, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant.

Reference: Pengxiang Zhao, Ping Li, Yingjie Gu, Yi Zheng, Stephan Ludger Kölker, Zhefeng Wang, Xiaoming Yuan, "Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices", arXiv 2024. https://arxiv.org/abs/2403.14958

---
[Back to the Canon](../README.md)
