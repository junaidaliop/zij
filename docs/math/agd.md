# AGD

Implements AGD, an auto-switchable optimizer that builds its preconditioner from the stepwise difference of bias-corrected gradient moments.

AGD forms the preconditioner from $s_t$, the difference between consecutive bias-corrected first moments, rather than from the raw gradient. A $\max$ in the denominator gates the per-coordinate behavior: where the accumulated squared difference $b_t$ is large the step is adaptive (Adam-like), and where it falls below the threshold set by $\delta$ the update reduces to scaled momentum (SGD-like), so the optimizer switches automatically between the two regimes per coordinate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
s_t &= \frac{m_t}{1 - \beta_1^{t}} - \frac{m_{t-1}}{1 - \beta_1^{t-1}} \\
b_t &= \beta_2 b_{t-1} + (1 - \beta_2)\, s_t^2 \\
\theta_{t+1} &= \theta_t - \gamma \, \frac{\sqrt{1 - \beta_2^{t}}}{1 - \beta_1^{t}} \cdot \frac{m_t}{\max\!\left(\sqrt{b_t},\, \delta \sqrt{1 - \beta_2^{t}}\right)}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $g_t$ the gradient, $m_t$ the first moment, $s_t$ the stepwise difference of bias-corrected moments (with $s_1 = m_1 / (1 - \beta_1)$), $b_t$ the second moment of $s_t$, $\beta_1,\beta_2$ the decay rates, and $\delta$ the threshold controlling the SGD-to-adaptive switch.

Reference: Yun Yue, Zhiling Ye, Jiadi Jiang, Yongchao Liu, Ke Zhang, "AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix", NeurIPS 2023. https://arxiv.org/abs/2312.01658

---
[Back to the Canon](../index.md)
