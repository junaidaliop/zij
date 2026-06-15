# AdaBelief

Implements AdaBelief, an Adam variant that scales the step size by the
belief in the observed gradient.

AdaBelief replaces Adam's second moment $v_t$ (the running average of
$g_t^2$) with $s_t$, the running average of the squared deviation
of the gradient from its own first moment $(g_t - m_t)^2$. A small
deviation signals a trustworthy gradient direction and yields a large step;
a large deviation yields a small step.


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
s_t &= \beta_2 s_{t-1} + (1 - \beta_2) (g_t - m_t)^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{s}_t = \frac{s_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\eta\, \hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate,
$g_t$ is the gradient, $m_t$ and $s_t$ are the first moment
and the belief in the gradient, and $\beta_1, \beta_2$ are the decay
rates of the moving averages.

The equations above describe the `rectify=False` path. The default
`rectify=True` instead applies the RAdam variance rectification: when the
length of the approximated moving average is large enough the step is
rescaled by the RAdam factor and the denominator uses the un-bias-corrected
$\sqrt{s_t}$, otherwise it reduces to an SGD-like step on
$\hat{m}_t$. Following the official implementation, $\epsilon$
is added to $s_t$ before the square root and again after it.

Reference: Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar Tatikonda,
Nicha Dvornek, Xenophon Papademetris, James S. Duncan, "AdaBelief Optimizer:
Adapting Stepsizes by the Belief in Observed Gradients", NeurIPS 2020.
https://arxiv.org/abs/2010.07468

---
[Back to the Canon](../README.md)
