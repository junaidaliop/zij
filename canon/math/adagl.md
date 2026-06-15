# AdaGL

Implements AdaGL, an Adam-style optimizer that replaces the gradient with a Grünwald–Letnikov fractional-order approximated gradient and modulates the step size by short-term gradient change.

The fractional-order gradient is a truncated Grünwald–Letnikov series over the last ten gradients, injecting long-term memory and global curvature into the moment estimates. A step size control coefficient $C_t$, built from a scaled-and-shifted softsign of the instantaneous gradient change, adapts the effective learning rate in real time: when the gradient barely changes (likely a flat minimum) the step shrinks toward exploration, and when it changes sharply (likely a sharp minimum or saddle) the step stays large enough to escape. The two pieces are combined in an Adam-style bias-corrected update.

$$
\begin{aligned}
\nabla^{\alpha} L(\theta_{t-1}) &= \sum_{j=0}^{10} (-1)^{j}\, \frac{\Gamma(\alpha+1)}{\Gamma(j+1)\,\Gamma(\alpha-j+1)}\, g_{t-j} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, \nabla^{\alpha} L(\theta_{t-1}) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, \bigl(\nabla^{\alpha} L(\theta_{t-1})\bigr)^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
C_t &= 1.1 - \frac{1}{2\,(1 + |g_{t-1} - g_t|)} \\
\theta_t &= \theta_{t-1} - \frac{\eta\, C_t\, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\nabla^{\alpha} L$ is the order-$\alpha$ Grünwald–Letnikov approximated gradient truncated to 10 terms, $\Gamma$ is the gamma function, $\alpha$ is the fractional order, $g_t$ is the current gradient, $m_t,v_t$ are the first and second moments of the fractional gradient with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias corrections, $C_t \in [0.6, 1.1)$ is the step size control coefficient, $\eta$ is the learning rate, and $\epsilon$ is a numerical stability constant.

Reference: Shuang Chen, Changlun Zhang, Haibing Mu, "An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on G–L Fractional-Order Derivative", Neural Processing Letters 2024. https://doi.org/10.1007/s11063-024-11571-7

---
[Back to the Canon](../README.md)
