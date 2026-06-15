# SPAM

Implements SPAM, spike-aware Adam with periodic momentum reset for stable LLM training.

Large language model training is destabilized by rare gradient spikes whose magnitude dwarfs the running average. SPAM detects such spikes on the fly using each coordinate's second moment as a scale, then clips the offending gradient back to a bounded magnitude while preserving its sign. To further limit the influence of any spike that does slip through, the optimizer periodically resets the first and second moments to zero every $\Delta T$ steps, followed by a short cosine warmup.

A coordinate is flagged as a spike when $g_t^2 / v_{t-1} > \theta$; flagged coordinates are rescaled to $\mathrm{sign}(g_t)\sqrt{\theta\, v_{t-1}}$ before the usual Adam moments are formed.

$$
\begin{aligned}
\tilde{g}_t &= \begin{cases} \mathrm{sign}(g_t)\,\sqrt{\theta\, v_{t-1}} & \text{if } g_t^2 / v_{t-1} > \theta \\ g_t & \text{otherwise} \end{cases} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde{g}_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde{g}_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \gamma \cdot s_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

where $\theta_t$ are the parameters, $\gamma$ the learning rate, $g_t$ the raw gradient, $\tilde{g}_t$ the spike-clipped gradient, $m_t,v_t$ the Adam moments, $\beta_1,\beta_2$ their decay rates, $\theta$ the spike-detection threshold, $\epsilon$ the stability constant, and $s_t$ a cosine warmup scale applied for $N$ steps after each reset; every $\Delta T$ iterations $m_t$ and $v_t$ are reset to zero.

Reference: Tianjin Huang, Ziquan Zhu, Lu Liu, Zhangyang Wang, Shiwei Liu, "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training", arXiv 2025. https://arxiv.org/abs/2501.06842

---
[Back to the Canon](../README.md)
