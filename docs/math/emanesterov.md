# EMA-Nesterov

Implements EMA-Nesterov, a wrapper that replaces Nesterov's one-step lookahead with an exponential moving average of the optimization trajectory.

Standard Nesterov acceleration extrapolates along the most recent update, which amplifies high-frequency noise in stochastic deep learning. EMA-Nesterov instead maintains a low-pass-filtered direction $m_t$, the EMA of successive parameter increments, and takes the base optimizer step from the lookahead point $\theta_t + \beta_t m_t$. The base optimizer $\mathcal{A}_t$ (Adam, SOAP, Muon, etc.) is treated as a black box, so the method is optimizer-agnostic; bias correction and weight decay are handled inside $\mathcal{A}_t$.

$$
\begin{aligned}
\theta_{t+1} &= \mathcal{A}_t\!\left(\theta_t + \beta_t\, m_t\right), \\
m_{t+1} &= \gamma\, m_t + (1-\gamma)\,(\theta_{t+1} - \theta_t).
\end{aligned}
$$

where $\theta$ are the parameters, $\mathcal{A}_t$ is the base optimizer step applied at the lookahead position, $m_t$ is the EMA of parameter increments, $\gamma \in [0,1)$ is the EMA decay rate, and $\beta_t \ge 0$ is the (scheduled) lookahead step size, set to $0$ during warm-up and the final decay phase.

Reference: Chung-Yiu Yau, Dawei Li, Athanasios Glentis, Valentyn Boreiko, Hoi-To Wai, Mingyi Hong, "EMA-Nesterov: Stabilizing Nesterov's Lookahead for Accelerated Deep Learning Optimization", arXiv 2026. https://arxiv.org/abs/2605.25395

---
[Back to the Canon](../index.md)
