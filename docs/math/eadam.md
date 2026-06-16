# EAdam

Implements EAdam, an Adam variant that accumulates the stability constant into the second moment instead of adding it to the denominator.

EAdam observes that where $\epsilon$ enters the update changes Adam's behavior. Rather than appending $\epsilon$ to $\sqrt{\hat{v}_t}$ once per step, EAdam adds $\epsilon$ directly to $v_t$ at every iteration before bias correction. Because $v_t$ carries over between steps, this $\epsilon$ compounds through training, producing an effective constant that grows roughly like $t\epsilon$ early on and acts as an automatically scaled, time-varying regularizer in the denominator.

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^{\,2} \\
v_t &\leftarrow v_t + \epsilon \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}},\qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \alpha\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first and second moment estimates with bias-corrected forms $\hat{m}_t$/$\hat{v}_t$, $\beta_1,\beta_2$ the exponential decay rates, and $\epsilon$ the stability constant added to $v_t$ each step (so it accumulates), with no further $\epsilon$ in the final denominator. Defaults $\alpha=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

Reference: Wei Yuan, Kai-Xin Gao, "EAdam Optimizer: How $\epsilon$ Impact Adam", arXiv 2020. https://arxiv.org/abs/2011.02150

---
[Back to the Canon](../index.md)
