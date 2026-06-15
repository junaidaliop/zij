# Distributed Shampoo

Implements Distributed Shampoo, a distributed data-parallel realization of the Shampoo full-matrix adaptive preconditioner.

For each layer with parameter matrix $\theta$ and gradient $g_t$, Shampoo maintains two Kronecker factor matrices that accumulate the left ($g_t g_t^\top$) and right ($g_t^\top g_t$) gradient covariances. The preconditioned direction is obtained by applying inverse fourth roots of these factors to the gradient. To make the method robust as a drop-in replacement, the Shampoo direction is rescaled by a grafted method (e.g. AdaGrad or Adam), which lends its step size while Shampoo supplies the direction; momentum and decoupled weight decay are then applied. For an order-$k$ tensor the inverse power generalizes to $-1/(2k)$.

$$
\begin{aligned}
L_t &= \beta_2 L_{t-1} + (1 - \beta_2)\, g_t g_t^\top, \qquad R_t = \beta_2 R_{t-1} + (1 - \beta_2)\, g_t^\top g_t \\
\bar{L}_t &= \left(\tfrac{L_t}{1 - \beta_2^{t}}\right)^{-1/4}, \qquad \bar{R}_t = \left(\tfrac{R_t}{1 - \beta_2^{t}}\right)^{-1/4} \\
P_t &= \bar{L}_t\, g_t\, \bar{R}_t \\
\tilde{P}_t &= \|P_t^{\mathrm{graft}}\|_F \,\frac{P_t}{\|P_t\|_F} \\
m_t &= \mu\, m_{t-1} + \tilde{P}_t \\
\theta_t &= \theta_{t-1} - \gamma \lambda\, \theta_{t-1} \\
\theta_t &= \theta_t - \gamma\, m_t
\end{aligned}
$$

where $L_t, R_t$ are the left/right Kronecker factors, $\bar{L}_t, \bar{R}_t$ their (bias-corrected) inverse fourth roots, $P_t$ the raw Shampoo direction, $P_t^{\mathrm{graft}}$ the search direction of the grafting method, $\|\cdot\|_F$ the Frobenius norm, $\mu$ the momentum, $\lambda$ the decoupled weight decay, $\gamma$ the learning rate, $\beta_2$ the factor EMA decay, and $\epsilon$ a small ridge added to each factor for numerical stability.

Reference: Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, Michael Rabbat, "A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale", arXiv 2023. https://arxiv.org/abs/2309.06497

---
[Back to the Canon](../README.md)
