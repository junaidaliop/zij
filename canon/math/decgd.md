# DecGD

Implements DecGD, an adaptive method built from a decomposition of the gradient.

The loss is wrapped as $L(\theta) = \sqrt{f(\theta) + c}$ with $c > 0$, so that $g_t = \nabla f(\theta_t) = 2 L(\theta_t)\, \nabla L(\theta_t)$. DecGD applies momentum to the decomposed gradient $\nabla L$ rather than to $\nabla f$, and maintains a loss-based vector $v_t$ that accumulates the inner product of this momentum with successive parameter increments. The update scales the step by $v_t$ elementwise, yielding a per-coordinate adaptive rate informed by the loss landscape; an optional AMS-style monotone variant keeps the running minimum of $v_t$.

$$
\begin{aligned}
d_t &= \frac{\nabla f(\theta_t)}{2\sqrt{f(\theta_t) + c}} \\
m_t &= \gamma\, m_{t-1} + d_t \\
v_t &= v_{t-1} + m_t \odot (\theta_t - \theta_{t-1}) \\
v_t^{*} &= \min(v_{t-1}^{*},\, v_t) \quad \text{(AMS variant)}, \qquad v_t^{*} = v_t \quad \text{(otherwise)} \\
\theta_{t+1} &= \theta_t - 2\eta\, v_t^{*} \odot m_t
\end{aligned}
$$

where $d_t$ is the decomposed (scaled) gradient $\nabla L$, $f(\theta_t)$ is the loss, $c > 0$ a stabilizing constant, $\gamma \in (0,1)$ the momentum coefficient, $m_t$ the momentum on $d_t$, $v_t$ the loss-based vector (initialized $v_0 = \sqrt{f(\theta_1) + c}$), $\odot$ elementwise product, and $\eta$ the learning rate.

Reference: Zhou Shao, Tong Lin, "A New Adaptive Gradient Method with Gradient Decomposition", arXiv 2021. https://arxiv.org/abs/2107.08377

---
[Back to the Canon](../README.md)
