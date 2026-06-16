# AdamD

Implements AdamD, Adam with bias correction applied only to the second moment.

AdamD retains Adam's exponential moving averages of the gradient and squared gradient, but drops the first-moment bias-correction term $1-\beta_1^t$ entirely, keeping only the well-justified second-moment correction $\sqrt{1-\beta_2^t}$ folded into the step size. Because the early uncorrected first moment $m_t$ is small, this yields conservative, monotonically increasing effective step sizes during the first steps of training and removes the need for learning-rate warmup.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\eta_t &= \eta \sqrt{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first- and second-moment estimates, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ a stability constant; note the absence of any $1-\beta_1^t$ correction on $m_t$.

Reference: John St John, "AdamD: Improved bias-correction in Adam", 2021. https://arxiv.org/abs/2110.10828

---
[Back to the Canon](../index.md)
