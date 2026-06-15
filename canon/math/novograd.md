# NovoGrad

Implements NovoGrad, an Adam variant with layer-wise adaptive moments.

NovoGrad keeps a single scalar second moment per layer, the running average
of the squared gradient norm, and normalizes the gradient by it before the
first moment accumulation. Weight decay is folded into the first moment, so
a layer with a large gradient norm receives a proportionally smaller update.


$$
\begin{aligned}
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \lVert g_t \rVert^2          \\
     m_t &= \beta_1 m_{t-1} +
         \left( \frac{g_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_{t-1}
         \right)                                                         \\
     \theta_t &= \theta_{t-1} - \eta \,
         \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \, m_t
\end{aligned}
$$

The second moment is initialized to $\lVert g_1 \rVert^2$ and the
norm is taken over each parameter tensor (the layer). With decoupled weight
decay the $\lambda \theta$ term is applied directly to the parameter
rather than through the first moment, matching AdamW. The implementation
applies Adam-style bias correction to the step size,
$\eta \, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$, unlike the paper
which removes bias through initialization.

Reference: Boris Ginsburg, Patrice Castonguay, Oleksii Hrinchuk,
Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, Huyen Nguyen,
Yang Zhang, Jonathan M. Cohen, "Stochastic Gradient Methods with Layer-wise
Adaptive Moments for Training of Deep Networks", arXiv 2019.
https://arxiv.org/abs/1905.11286

---
[Back to the Canon](../README.md)
