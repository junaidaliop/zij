# AMSGrad

Implements AMSGrad, a variant of Adam that uses a non-increasing second-moment scaling to fix Adam's convergence failures.

Reddi et al. show that Adam can fail to converge even on simple convex problems because the exponential moving average of squared gradients can shrink the effective second moment, occasionally yielding large, destabilizing steps. AMSGrad keeps a running maximum $\hat{v}_t$ of the second-moment estimate and divides by that instead of $v_t$. This guarantees the per-coordinate step size is monotonically non-increasing, restoring convergence while keeping the same time and memory cost as Adam.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{v}_t &= \max(\hat{v}_{t-1}, v_t) \\
\theta_t &= \theta_{t-1} - \eta \, \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first- and second-moment estimates, $\hat{v}_t$ the elementwise running maximum of $v_t$, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ a small stability constant.

Reference: Sashank J. Reddi, Satyen Kale, Sanjiv Kumar, "On the Convergence of Adam and Beyond", ICLR 2018. https://arxiv.org/abs/1904.09237

---
[Back to the Canon](../index.md)
