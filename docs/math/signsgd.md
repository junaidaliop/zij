# SignSGD

Implements SignSGD, sign-based stochastic gradient descent.

SignSGD compresses each gradient to its element-wise sign before applying
the update, which keeps only one bit per coordinate. With `momentum` set
to zero the update is the plain sign of the gradient:


$$
\theta_t = \theta_{t-1} - \gamma \mathrm{sign}(g_t)
$$

With a positive momentum coefficient the method becomes Signum, which takes
the sign of an exponential moving average of the gradients:


$$
\begin{aligned}
m_t &= \beta m_{t-1} + (1 - \beta)\, g_t \\
\theta_t &= \theta_{t-1} - \gamma \mathrm{sign}(m_t)
\end{aligned}
$$

where $m_t$ is the momentum buffer, $\gamma$ the learning rate,
and $\beta$ the momentum coefficient. Decoupled weight decay scales the
parameters by $(1 - \gamma \lambda)$ before the update; with
`weight_decouple=False` the weight decay $\lambda$ is instead added
to the gradient as an L2 penalty.

Reference: Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli,
Anima Anandkumar, "signSGD: Compressed Optimisation for Non-Convex
Problems", ICML 2018.
https://arxiv.org/abs/1802.04434

---
[Back to the Canon](../index.md)
