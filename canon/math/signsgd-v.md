# signSGD

Implements signSGD, gradient descent that uses only the sign of each gradient coordinate.

signSGD compresses the gradient to a single bit per coordinate, taking a fixed step in the direction given by the elementwise sign. This makes the step magnitude independent of the gradient scale and slashes communication cost in the distributed setting. The Signum variant first accumulates an exponential moving average of the gradient and applies the sign to that momentum buffer, which trades variance for a small bias in the stochastic gradient estimate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
\theta_t &= \theta_{t-1} - \eta \, \mathrm{sign}(m_t)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the stochastic gradient, $m_t$ the momentum buffer, and $\beta_1$ the momentum decay; plain signSGD is the special case $\beta_1 = 0$, giving $\theta_t = \theta_{t-1} - \eta \, \mathrm{sign}(g_t)$, and $\mathrm{sign}(\cdot)$ is applied elementwise.

Reference: Bernstein, Wang, Azizzadenesheli, Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018. https://arxiv.org/abs/1802.04434

---
[Back to the Canon](../README.md)
