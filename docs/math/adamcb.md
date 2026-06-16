# AdamCB

Implements AdamCB, Adam with adaptive batch selection driven by a combinatorial bandit.

AdamCB replaces uniform mini-batch sampling with an adaptive scheme: at each step it draws $K$ distinct samples without replacement, where the selection probabilities are maintained by a combinatorial-bandit rule that favors high-influence samples while keeping a uniform exploration floor. The resulting gradient is importance-weighted so it stays unbiased, then fed into an AMSGrad-style Adam update with a time-decaying first-moment rate $\beta_{1,t}$ and a matching correction inside the running maximum.

$$
\begin{aligned}
g_t &= \frac{1}{K}\sum_{j\in J_t}\frac{g_{j,t}}{n\,p_{j,t}} \\
m_t &= \beta_{1,t}\,m_{t-1} + (1-\beta_{1,t})\,g_t \\
v_t &= \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2 \\
\hat{v}_t &= \max\left\{\frac{(1-\beta_{1,t})^2}{(1-\beta_{1,t-1})^2}\,\hat{v}_{t-1},\; v_t\right\} \\
\theta_{t+1} &= \theta_t - \alpha_t\,\frac{m_t}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

where $J_t$ is the set of $K$ sampled indices, $p_{j,t}$ their selection probabilities, $n$ the dataset size, $g_{j,t}$ the per-sample gradient, $\beta_{1,t}=\beta_1\lambda^{t-1}$ with $\lambda\in(0,1)$ the decaying first-moment rate, $\beta_2$ the second-moment rate, $\hat{v}_t$ the AMSGrad running maximum (the $\max$ applies for $t\ge 2$), $\alpha_t=\alpha/\sqrt{t}$ the step size, and $\epsilon$ a stability constant.

Reference: Gyu Yeol Kim, Min-hwan Oh, "Adam Optimization with Adaptive Batch Selection", arXiv 2025. https://arxiv.org/abs/2512.06795

---
[Back to the Canon](../index.md)
