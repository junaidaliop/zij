# SlowMo

Implements SlowMo, a slow-momentum wrapper that periodically averages workers and applies an outer momentum step.

SlowMo is a framework for communication-efficient distributed training. Each worker runs $\tau$ steps of a base optimizer (e.g. local SGD) on its own data, after which the worker parameters are exactly averaged. The averaged result is treated as a single "fast" update direction, and an outer loop maintains a slow momentum buffer that is applied to the global parameters with a separate slow learning rate. This decouples the inner optimization from the infrequent synchronization and recovers the accuracy lost by reducing communication.

$$
\begin{aligned}
\theta_{t,k+1}^{(i)} &= \theta_{t,k}^{(i)} - \gamma_t\, d_{t,k}^{(i)}, \qquad k = 0,\dots,\tau-1 \\
\bar{\theta}_{t,\tau} &= \frac{1}{m}\sum_{i=1}^{m} \theta_{t,\tau}^{(i)} \\
u_{t+1} &= \beta\, u_t + \frac{1}{\gamma_t}\left(\theta_{t,0} - \bar{\theta}_{t,\tau}\right) \\
\theta_{t+1,0} &= \theta_{t,0} - \alpha\, \gamma_t\, u_{t+1}
\end{aligned}
$$

where $\theta_{t,k}^{(i)}$ are worker $i$'s parameters at inner step $k$, $\gamma_t$ is the base (fast) learning rate, $d_{t,k}^{(i)}$ is the base optimizer's update direction (the gradient for SGD), $m$ is the number of workers, $\tau$ is the number of inner steps per round, $u_t$ is the slow momentum buffer, $\beta$ is the slow momentum factor, and $\alpha$ is the slow learning rate.

Reference: Wang, Tantia, Ballas, Rabbat, "SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum", ICLR 2020. https://arxiv.org/abs/1910.00643

---
[Back to the Canon](../index.md)
