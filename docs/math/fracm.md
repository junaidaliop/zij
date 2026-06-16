# FracM

Implements FracM, SGD with momentum whose update is driven by a fractional-order difference of the momentum and gradient.

FracM replaces the integer-order difference used in classical SGD with momentum (SGDM) by a Grünwald-Letnikov (G-L) fractional-order difference of order $\alpha \in (0,1)$. Because the fractional difference accumulates a weighted history of past states, the resulting update carries the memory and nonlocality of fractional calculus, which the authors report helps escape shallow local minima and speeds up training. A short-memory truncation (a fixed number $K$ of past terms, about ten in the paper) keeps the per-step cost bounded.

Starting from the SGDM recursion $m_t = \mu\, m_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta\, m_t$, FracM applies the G-L fractional difference $\Delta^{\alpha}$ to the momentum/gradient sequence rather than the ordinary first difference, giving

$$
\begin{aligned}
\Delta^{\alpha} x_t &= \sum_{k=0}^{K} (-1)^k \binom{\alpha}{k}\, x_{t-k}, \qquad \binom{\alpha}{k} = \frac{\Gamma(\alpha+1)}{\Gamma(k+1)\,\Gamma(\alpha-k+1)}, \\
m_t &= \mu\, m_{t-1} + \Delta^{\alpha} g_t, \\
\theta_t &= \theta_{t-1} - \eta\, m_t.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum buffer, $\mu$ the momentum coefficient, $\alpha$ the fractional order, $K$ the short-memory length, $\binom{\alpha}{k}$ the generalized binomial coefficient written through the Gamma function $\Gamma(\cdot)$, and $\Delta^{\alpha}$ the truncated Grünwald-Letnikov fractional difference.

Reference: Z. Yu, G. Sun, J. Lv, "A fractional-order momentum optimization approach of deep neural networks", Neural Computing and Applications 2022. https://doi.org/10.1007/s00521-021-06765-2

---
[Back to the Canon](../index.md)
