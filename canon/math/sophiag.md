# SophiaG

Implements Sophia (Gauss-Newton-Bartlett variant), a second-order
clipped stochastic optimizer.

Sophia preconditions the gradient with a moving average of a light-weight
diagonal Hessian estimate and clips the result element-wise, which bounds
the worst-case update along any coordinate. With first moment $m_t$,
diagonal Hessian estimate $h_t$, learning rate $\eta$, decay
rates $\beta_1$, $\beta_2$, and pre-conditioner coefficient
$\rho$ (the paper's $\gamma$), with the per-coordinate clip
applied to magnitude 1:


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
h_t &= \beta_2 h_{t-k} + (1 - \beta_2)\, \hat{h}_t \\
\theta_t &= (1 - \eta\lambda)\, \theta_{t-1} - \eta\,
    \mathrm{clip}\!\left(
        \frac{m_t}{\rho\, B\, h_t + \epsilon},\, 1
    \right)
\end{aligned}
$$

where $\lambda$ is the decoupled `weight_decay` and $B$ is the
`bs` (batch size) passed to `step`. The Hessian estimate
$h_t$ is refreshed every $k$ steps by `update_hessian`.
The Gauss-Newton-Bartlett estimator forms $\hat{h}_t$ from the
per-coordinate squared gradient of a loss evaluated on labels sampled from
the model's own predictive distribution; the batch-size factor $B$ is
applied here in the denominator rather than folded into
$\hat{h}_t$, following the official implementation. The clip operates
per coordinate, so the effective step never exceeds $\eta$ in
magnitude.


**Note:** Sophia requires a periodic Hessian refresh. Call

`update_hessian` every `k` steps after a backward pass on a sampled
loss (a closure), then call `step`. The `bs` argument to
`step` is the batch size used to scale the estimator. Until the first
`update_hessian` call the estimate is zero and every update saturates
the clip, reducing the step to $-\eta\,\mathrm{sign}(m_t)$.

Reference: Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma,
"Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
Pre-training", ICLR 2024.
https://arxiv.org/abs/2305.14342

---
[Back to the Canon](../README.md)
