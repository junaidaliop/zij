# AdamW

Implements AdamW, Adam with weight decay decoupled from the gradient
update.

In standard Adam, L2 regularization is folded into the gradient, so the
weight decay is rescaled by the per-coordinate adaptive learning rate and
its effect on parameters with large second moments is suppressed. AdamW
removes the decay term from the gradient and instead subtracts
$\eta \lambda \theta_{t-1}$ directly from the parameters at each step. The
adaptive moment estimates $m_t$ and $v_t$ are therefore computed from the
raw gradient, and the regularization acts uniformly across coordinates.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}
\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
gradient, $m_t$ and $v_t$ are the first and second moment estimates,
$\beta_1, \beta_2$ are the decay rates, $\lambda$ is the weight decay
coefficient, and $\epsilon$ is a numerical-stability term.

Reference: Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay
Regularization", ICLR 2019.
https://arxiv.org/abs/1711.05101

---
[Back to the Canon](../README.md)
