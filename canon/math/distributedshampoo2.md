# Distributed Shampoo

Implements Distributed Shampoo, a full-matrix Kronecker-factored preconditioner with learning-rate grafting for at-scale data-parallel training.

For a matrix parameter $\theta\in\mathbb{R}^{m\times n}$, Shampoo keeps left and right second-order statistics $L_t\in\mathbb{R}^{m\times m}$ and $R_t\in\mathbb{R}^{n\times n}$ as exponential moving averages of $g_t g_t^{\top}$ and $g_t^{\top} g_t$, and preconditions the gradient by their inverse fourth roots. Because the raw Shampoo direction has no natural scale, the step length is grafted from a diagonal AdaGrad/Adam preconditioner: the Shampoo direction is rescaled to the Frobenius norm of the grafting direction. Nesterov-style momentum and decoupled (AdamW) weight decay are then applied to produce the update.

$$
\begin{aligned}
L_t &= \beta_2 L_{t-1} + (1-\beta_2)\, g_t g_t^{\top} + \epsilon I_m, \qquad R_t = \beta_2 R_{t-1} + (1-\beta_2)\, g_t^{\top} g_t + \epsilon I_n \\
A_t &= \beta_2 A_{t-1} + (1-\beta_2)\, g_t \odot g_t \\
P_t^{\mathrm{shampoo}} &= L_t^{-1/4}\, g_t\, R_t^{-1/4}, \qquad P_t^{\mathrm{graft}} = \frac{g_t}{\sqrt{A_t} + \epsilon} \\
P_t &= \frac{\lVert P_t^{\mathrm{graft}} \rVert_F}{\lVert P_t^{\mathrm{shampoo}} \rVert_F}\; P_t^{\mathrm{shampoo}} \\
M_t &= \mu M_{t-1} + P_t \\
\theta_t &= \theta_{t-1} - \eta\, M_t - \eta\, \lambda\, \theta_{t-1}
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the gradient reshaped to a matrix, $\eta$ the learning rate, $\beta_2$ the second-moment decay, $\mu$ the momentum coefficient, $\lambda$ the decoupled weight-decay coefficient, $\epsilon$ the regularization/damping constant, $L_t,R_t$ the left and right Kronecker factors with $L_t^{-1/4},R_t^{-1/4}$ their inverse fourth roots (via eigendecomposition), $A_t$ the diagonal grafting state, $\odot$ the elementwise product, $\lVert\cdot\rVert_F$ the Frobenius norm, and $M_t$ the momentum buffer.

Reference: Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, Michael Rabbat, "A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale", ACM Transactions on Mathematical Software 2023. https://arxiv.org/abs/2309.06497

---
[Back to the Canon](../README.md)
