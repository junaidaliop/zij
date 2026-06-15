# FedAC

Implements FedAc (Federated Accelerated SGD), a provably accelerated variant of Federated Averaging built on a generalized Nesterov scheme.

Each of the $M$ workers maintains three coupled sequences: a main iterate $w_t$, an aggregated "ag" iterate $w_t^{ag}$, and a "middle" point $w_t^{md}$ where the gradient is queried. Every worker runs the accelerated SGD update locally for $K$ steps; at each synchronization round the main and aggregated iterates are averaged across workers and broadcast back. The four hyperparameters $\alpha,\beta,\gamma,\eta$ decouple acceleration from stability, which is what enables the communication savings over FedAvg.

For worker $m$ at local step $t$ (gradient $g_t = \nabla f(w_t^{md}; \xi_t^m)$):

$$
\begin{aligned}
w_t^{md} &= \tfrac{1}{\beta}\, w_t + \left(1-\tfrac{1}{\beta}\right) w_t^{ag} \\
w_{t+1}^{ag} &= w_t^{md} - \eta\, g_t \\
w_{t+1} &= \left(1-\tfrac{1}{\alpha}\right) w_t + \tfrac{1}{\alpha}\, w_t^{md} - \gamma\, g_t \\
\text{if } t \bmod K = 0:\quad w_{t+1} &\leftarrow \tfrac{1}{M}\sum_{m=1}^{M} w_{t+1}^{m}, \qquad w_{t+1}^{ag} \leftarrow \tfrac{1}{M}\sum_{m=1}^{M} w_{t+1}^{ag,m}
\end{aligned}
$$

where $\eta$ is the learning rate for the aggregated step, $\gamma$ the (larger) learning rate for the main step, and $\alpha,\beta$ the coupling coefficients. For a $\mu$-strongly-convex objective the FedAc-I choice sets $\gamma = \max\!\left(\eta, \sqrt{\eta/(\mu K)}\right)$, $\alpha = 1/(\gamma\mu)$, and $\beta = \alpha + 1$, with $\eta \in (0, 1/L]$.

Reference: Honglin Yuan, Tengyu Ma, "Federated Accelerated Stochastic Gradient Descent", NeurIPS 2020. https://arxiv.org/abs/2006.08950

---
[Back to the Canon](../README.md)
