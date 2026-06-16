# FADAS

Implements FADAS, federated adaptive asynchronous optimization with an Adam-like server update and delay-adaptive learning rate.

Each client $i$ runs local SGD from the broadcast model and returns the model-update difference $\Delta_t^i = x_{t-\tau,K}^i - x_{t-\tau}$, which the server treats as a pseudo-gradient. The server keeps a buffer of size $M$: it accumulates incoming differences into $\Delta_t$ and, once $M$ updates arrive, averages them and applies an AMSGrad-style adaptive step. To stay robust to stragglers, the global learning rate is scaled down whenever the maximum staleness $\tau_t^{\max}$ in a round exceeds a delay threshold $\tau_c$, shrinking the step in proportion to $1/\tau_t^{\max}$.

$$
\begin{aligned}
\Delta_t &= \frac{1}{M}\sum_{i\in\mathcal{M}_t}\Delta_t^i, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\Delta_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\Delta_t \odot \Delta_t, \\
\hat{v}_t &= \max(\hat{v}_{t-1},\, v_t), \\
\eta_t &= \begin{cases} \eta & \text{if } \tau_t^{\max} \le \tau_c, \\ \min\!\left(\eta,\ \dfrac{1}{\tau_t^{\max}}\right) & \text{if } \tau_t^{\max} > \tau_c, \end{cases} \\
x_{t+1} &= x_t + \eta_t\,\frac{m_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
$$

where $\theta$ (here $x$) are the global model parameters, $\eta$ the base global learning rate, $\eta_l$ the local learning rate, $\Delta_t^i$ the buffered model-update difference from client $i$, $M$ the buffer size, $\mathcal{M}_t$ the clients contributing at round $t$, $m_t$ and $v_t$ the first and second pseudo-gradient moments, $\hat{v}_t$ the running maximum second moment, $\beta_1,\beta_2$ the decay rates, $\tau_t^{\max}$ the maximum client delay in the round, $\tau_c$ the delay threshold, and $\epsilon$ a stability constant.

Reference: Yujia Wang, Shiqiang Wang, Songtao Lu, Jinghui Chen, "FADAS: Towards Federated Adaptive Asynchronous Optimization", ICML 2024. https://arxiv.org/abs/2407.18365

---
[Back to the Canon](../index.md)
