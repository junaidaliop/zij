# HomeAdam

Implements HomeAdam, an Adam variant that conditionally "goes home" to momentum SGD when the second-moment estimate becomes too small.

Adam and AdamW converge quickly but generalize worse than SGD. HomeAdam keeps the usual first/second moment estimates with bias correction, but drops the square root on the second moment and inspects its smallest coordinate at each step. When every coordinate of $\hat{v}_t$ stays above a threshold $\tau$ the adaptive (Adam-style) step is used; once any coordinate falls below $\tau$ the adaptive denominator is deemed unreliable and the step "returns home" to a plain momentum-SGD update. This switch yields a provably smaller generalization error than Adam while retaining a fast convergence rate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad \hat{m}_t = \frac{m_t}{1-\beta_1^t} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &=
\begin{cases}
\theta_{t-1} - \eta\!\left( \dfrac{\hat{m}_t}{\hat{v}_t + \epsilon} + \lambda\,\theta_{t-1} \right), & \min_{1 \le j \le d} (\hat{v}_t)_j \ge \tau \\
\theta_{t-1} - \eta\!\left( \hat{m}_t + \lambda\,\theta_{t-1} \right), & \text{otherwise}
\end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first/second moment estimates with bias-corrected forms $\hat{m}_t,\hat{v}_t$, $\beta_1,\beta_2$ the decay rates, $\lambda$ the weight decay, $\epsilon$ a stability constant, $\tau > 0$ the home-switching threshold, and $d$ the parameter dimension; note the square root on $\hat{v}_t$ is removed.

Reference: Feihu Huang, Guanyi Zhang, Songcan Chen, "HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain Better Provable Generalization", ICML 2025. https://arxiv.org/abs/2603.02649

---
[Back to the Canon](../index.md)
