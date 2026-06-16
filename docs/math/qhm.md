# QHM

Implements quasi-hyperbolic momentum (QHM), a discounted interpolation
between plain SGD and momentum.


$$
\begin{aligned}
   g_{t+1} &= \beta\, g_t + (1 - \beta)\, \nabla_t \\
   \theta_{t+1} &= \theta_t
       - \alpha \left[ (1 - \nu)\, \nabla_t + \nu\, g_{t+1} \right]
\end{aligned}
$$

where $\alpha$ is the learning rate, $\beta$ the momentum
factor, $\nu$ the immediate discount factor that interpolates between
plain SGD ($\nu = 0$) and momentum ($\nu = 1$), $g_t$ the
momentum buffer, and $\nabla_t$ the gradient at $\theta_t$.


**Note:** QHM uses dampened momentum. When converting from plain momentum to QHM, scale the learning rate by $1 / (1 - \beta)$: momentum with $\alpha = 0.1$ and $\beta = 0.9$ corresponds to QHM with $\alpha = 1.0$.

Reference: Jerry Ma, Denis Yarats, "Quasi-hyperbolic momentum and Adam for
deep learning", ICLR 2019.
https://arxiv.org/abs/1810.06801

---
[Back to the Canon](../index.md)
