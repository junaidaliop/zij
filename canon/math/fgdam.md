# FGDAM

Implements FGDAM (fractional gradient descent with adaptive momentum), BP-network training that combines a Grünwald-Letnikov fractional gradient with a momentum term whose coefficient adapts each step.

FGDAM replaces the integer-order gradient of standard backpropagation with the Grünwald-Letnikov (G-L) fractional gradient of order $\alpha$, which accumulates a weighted history of past weight states and so carries the memory and nonlocality of fractional calculus. To this fractional descent step the method adds a heavy-ball momentum term, but instead of a fixed coefficient it uses an adaptive one: the momentum weight is recomputed each iteration from the current fractional gradient and the previous weight change, which the authors report stabilizes training, helps escape local minima, and enlarges the usable range of the learning rate.

Writing $g_t^{(\alpha)}$ for the G-L fractional gradient of the loss at $\theta_t$ and $\Delta\theta_{t} = \theta_t - \theta_{t-1}$ for the previous weight change, one iteration is

$$
\begin{aligned}
g_t^{(\alpha)} &= \sum_{k=0}^{K} \frac{(-1)^k\,\Gamma(\alpha+1)}{\Gamma(k+1)\,\Gamma(\alpha-k+1)}\, \nabla_\theta E\big(\theta_{t-k}\big), \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{(\alpha)} + \mu_t\, \Delta\theta_{t}, \\
\mu_t &= \mu\big(g_t^{(\alpha)},\, \Delta\theta_{t}\big),
\end{aligned}
$$

where $\theta$ are the network weights, $E$ the loss, $\eta > 0$ the learning rate, $\alpha$ the fractional order, $g_t^{(\alpha)}$ the truncated G-L fractional gradient over a short memory of $K$ past steps, $\Gamma(\cdot)$ the Gamma function, $\Delta\theta_t$ the previous weight increment, and $\mu_t$ the adaptive momentum coefficient set each step as a function of the current fractional gradient and $\Delta\theta_t$.

Reference: Xiaohui Han, Jianping Dong, "Applications of fractional gradient descent method with adaptive momentum in BP neural networks", Applied Mathematics and Computation 2023. https://doi.org/10.1016/j.amc.2023.127944

---
[Back to the Canon](../README.md)
