# AdaFisher

Implements AdaFisher, an adaptive optimizer that preconditions the gradient with a diagonal block-Kronecker approximation of the Fisher Information Matrix.

AdaFisher replaces Adam's second-moment denominator with a curvature estimate drawn from the Fisher Information Matrix (FIM). For each layer the FIM is approximated by a Kronecker product of two factors built from the layer's activations and pre-activation gradients; AdaFisher keeps only the diagonals of these factors, Min-Max normalizes them, and forms their Kronecker product as a cheap per-parameter curvature vector. The factors are tracked with an exponential moving average, and the preconditioned step uses the bias-corrected momentum divided by this diagonal Fisher rather than by the square root of the gradient second moment.

$$
\begin{aligned}
\mathcal{H}_{D} &\leftarrow \gamma_1\, \mathcal{H}_{D} + (1-\gamma_2)\, \hat{\mathcal{H}}_{D} \\
\mathcal{S}_{D} &\leftarrow \gamma_1\, \mathcal{S}_{D} + (1-\gamma_2)\, \hat{\mathcal{S}}_{D} \\
\tilde{F}_{D} &= \mathcal{H}'_{D} \otimes \mathcal{S}'_{D} + \lambda \\
m_t &= \frac{\beta\, m_{t-1} + (1-\beta)\, g_t}{1 - \beta^{t}} \\
\theta_{t} &= \theta_{t-1} - \eta\left( \tilde{F}_{D}^{-1} m_t + \kappa\, \theta_{t-1} \right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the bias-corrected first moment with decay $\beta$, and $\tilde{F}_{D}$ the diagonal Fisher approximation. $\mathcal{H}_{D}$ and $\mathcal{S}_{D}$ are the diagonals of the Kronecker factors (from pre-activation gradients and activations), tracked by an EMA with decay factors $\gamma_1,\gamma_2$; $\mathcal{H}'_{D}, \mathcal{S}'_{D}$ are their Min-Max normalized forms; $\otimes$ is the Kronecker product; $\lambda$ is a Tikhonov damping constant; and $\kappa$ is the decoupled weight-decay coefficient (AdaFisherW, omit the $\kappa\,\theta$ term for plain AdaFisher). Note the absence of any square root on the denominator, in contrast to Adam.

Reference: Damien Martins Gomes, Yanlei Zhang, Eugene Belilovsky, Guy Wolf, Mahdi S. Hosseini, "AdaFisher: Adaptive Second Order Optimization via Fisher Information", ICLR 2025. https://arxiv.org/abs/2405.16397

---
[Back to the Canon](../README.md)
