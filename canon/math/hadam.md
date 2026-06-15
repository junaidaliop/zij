# HAdam

Implements HAdam, an Adam variant that replaces the second moment with an arbitrary even-order moment of the gradient.

HAdam generalizes Adam by tracking the $k$-th raw moment of the stochastic gradient instead of just the second moment, with $k = 2d$ restricted to even integers. The denominator then uses the $k$-th root of this moment, and the bias-correction factor on the squared-moment term is likewise taken to the $1/k$ power. Even-order moments match or improve on Adam, whereas odd-order moments break the boundedness of the effective step size and lead to divergence.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
V_t &= \beta_2 V_{t-1} + (1-\beta_2) g_t^{k} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{(1-\beta_2^t)^{1/k}}{1-\beta_1^t} \, \frac{m_t}{V_t^{1/k} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the gradient, $m_t$ is the first moment, $V_t$ is the $k$-th moment, $\beta_1,\beta_2$ are the decay rates, $\epsilon$ is the stability constant, and $k = 2d$ for $d \in \{1, 2, \dots\}$ (Adam is recovered at $k=2$).

Reference: Zhanhong Jiang, Aditya Balu, Sin Yong Tan, Young M. Lee, Chinmay Hegde, Soumik Sarkar, "On Higher-order Moments in Adam", arXiv 2019. https://arxiv.org/abs/1910.06878

---
[Back to the Canon](../README.md)
