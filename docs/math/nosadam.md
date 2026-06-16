# NosAdam

Implements NosAdam (Nostalgic Adam), an Adam variant that weights past squared gradients more heavily when forming the adaptive learning rate.

Standard Adam uses an exponential moving average of squared gradients, which assigns more weight to recent gradients. NosAdam instead defines a coefficient sequence $b_k \ge 0$ with partial sums $B_t = \sum_{k=1}^{t} b_k$ and sets a time-varying decay $\beta_{2,t} = B_{t-1}/B_t$. This makes the second-moment estimate a weighted average $v_t = \tfrac{1}{B_t}\sum_{k=1}^{t} b_k\, g_k^2$ with non-increasing weights in $k$, so older gradients retain influence. The NosAdam-HH variant uses the hyperharmonic series $b_k = k^{-\gamma}$ with $\gamma \ge 0$.

$$
\begin{aligned}
g_t &= \nabla f_t(\theta_{t-1}) \\
B_t &= \sum_{k=1}^{t} b_k, \quad \beta_{2,t} = \frac{B_{t-1}}{B_t} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_{2,t}\, v_{t-1} + (1-\beta_{2,t})\, g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta\, m_t}{\sqrt{v_t}}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the first moment with decay $\beta_1$, $v_t$ the second moment, $b_k \ge 0$ the weighting coefficients with cumulative sum $B_t$ (and $B_0 = 0$), and $\beta_{2,t} = B_{t-1}/B_t$ the induced second-moment decay; for NosAdam-HH, $b_k = k^{-\gamma}$ with $\gamma \ge 0$.

Reference: Haiwen Huang, Chang Wang, Bin Dong, "Nostalgic Adam: Weighting more of the past gradients when designing the adaptive learning rate", arXiv 2018. https://arxiv.org/abs/1805.07557

---
[Back to the Canon](../index.md)
