# Adamax

Implements AdaMax, a variant of Adam in which the second moment is
replaced by an exponentially weighted infinity norm of the gradients.

Adam scales the step by the $\ell_2$ norm of past gradients through the
second moment $v_t$. AdaMax generalizes this to the $\ell_p$ norm and
takes the limit $p \to \infty$, which collapses the accumulator into a
running maximum: $u_t$ tracks the largest recent gradient magnitude with
exponential decay. Because $u_t$ is a max rather than a sum, it needs no
bias correction, and only the first moment is bias-corrected, folded into
the effective learning rate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
u_t &= \max\!\left(\beta_2\, u_{t-1},\; |g_t| + \epsilon\right) \\
\theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} \cdot \frac{m_t}{u_t}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is
the gradient, $m_t$ is the first moment, $u_t$ is the exponentially
weighted infinity norm, $\beta_1, \beta_2$ are the decay rates, and
$\epsilon$ is a numerical-stability term.

Reference: Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic
Optimization", ICLR 2015.
https://arxiv.org/abs/1412.6980

---
[Back to the Canon](../README.md)
