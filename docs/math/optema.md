# OptEMA

Implements OptEMA, an EMA-based optimizer with a closed-loop, Lipschitz-free adaptive stepsize.

OptEMA keeps the Adam-style first- and second-moment EMA backbone but replaces the preset coefficients and stepsize with trajectory-dependent quantities driven by a Corrected AdaGrad-Norm statistic $\rho_t$ and the running maximum gradient norm $\hat g_t$. Two symmetric variants are studied: OptEMA-M makes the first-moment coefficient adaptive ($\alpha_t = \rho_t$) with a fixed second-moment decay, while OptEMA-V makes the second-moment coefficient adaptive ($\beta_t = \rho_t$) with a fixed first-moment decay. The Corrected AdaGrad-Norm numerator averages historical squared gradient norms, which tempers AdaGrad-Norm's premature decay and yields a noise-adaptive rate that collapses to the deterministic optimum when the noise vanishes.

$$
\begin{aligned}
\rho_t &= \frac{\sqrt{1 + \tfrac{\tau}{t}\sum_{i=1}^{t}\|g_i\|^2}}{1 + \sum_{i=1}^{t}\|g_i\|^2}, \qquad \hat g_t = \max_{1\le i\le t}\|g_i\| \\
(\alpha_t,\beta_t) &= \begin{cases}(\rho_t,\ \beta) & \text{OptEMA-M}\\ (\alpha,\ \rho_t) & \text{OptEMA-V}\end{cases} \\
m_t &= (1-\alpha_t)\,m_{t-1} + \alpha_t\,g_t, \qquad v_t = (1-\beta_t)\,v_{t-1} + \beta_t\,g_t^2 \\
\gamma_t &= \begin{cases}\min\!\Big(\dfrac{\alpha_t}{1+\mu\hat g_t^4},\ \big(1+\sum_{j=1}^{t}\|m_j\|^2/\alpha_j\big)^{-1/2}\Big) & \text{OptEMA-M}\\ \dfrac{1}{1+\mu\hat g_t^4}\big(1+\sum_{j=1}^{t}\|m_j\|^2\big)^{-1/2} & \text{OptEMA-V}\end{cases} \\
\theta_{t+1} &= \theta_t - \eta\,\gamma_t\cdot\frac{m_t}{\varepsilon + \sqrt{v_t}}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate (default $1$), $g_t$ the stochastic gradient, $m_t$/$v_t$ the first/second moment EMAs, $\alpha_t,\beta_t$ the EMA coefficients, $\gamma_t$ the closed-loop effective stepsize, $\rho_t$ the Corrected AdaGrad-Norm statistic with $\tau\in[0,1]$ (default $\tau=1$), $\hat g_t$ the running max gradient norm, $\alpha,\beta\in(0,1)$ the fixed decays, and $\mu,\varepsilon$ small stabilizers.

Reference: Ganzhao Yuan, "OptEMA: Adaptive Exponential Moving Average for Stochastic Optimization with Zero-Noise Optimality", arXiv 2026. https://arxiv.org/abs/2603.09923

---
[Back to the Canon](../index.md)
