# IO-Adam

Implements IO-Adam, a memory-efficient Adam variant that tracks the input and output gradients instead of the full first and second moments.

For a linear layer $y = Wx$, the weight gradient factorizes as the outer product $\nabla_W L = (\nabla_y L)x^\top$. IO-Adam exploits this by storing the input $x$ and the output gradient $\nabla_y L$ rather than the $n\times m$ moment matrices. The first moment is updated in a fused step from the input/output gradients, and the second moment is reconstructed from two factors $C_W \in \mathbb{R}^{m\times b}$ (squared input) and $R_W \in \mathbb{R}^{n\times b}$ (squared output gradient), whose columns are updated alternately so that the product mixes fewer cross-batch terms.

$$
\begin{aligned}
M_W^t &= \beta_1 M_W^{t-1} + (1-\beta_1)\,(\nabla_{Y_t} L)\,X_t^\top \\
C_W^t &= \beta_2 C_W^{t-1} + (1-\beta_2)\,X_t^{2}\,(1_{bs} e_{(t \bmod b)}^\top) \\
R_W^t &= \beta_2 R_W^{t-1} + (1-\beta_2)\,(\nabla_{Y_t} L)^{2}\,(1_{bs} e_{(t \bmod b)}^\top) \\
V_W^t &= C_W^t \, R_W^{t\top} \\
\hat{M}_W^t &= M_W^t / (1-\beta_1^t), \qquad \hat{V}_W^t = V_W^t / (1-\beta_2^t)^2 \\
W^t &= W^{t-1} - \alpha\, \hat{M}_W^t \big/ \big(\sqrt{\hat{V}_W^t} + \epsilon\big)
\end{aligned}
$$

where $X_t$ is the layer input and $\nabla_{Y_t} L$ the output gradient at step $t$, $bs$ the batch size, $b$ the buffer width, $1_{bs}$ an all-ones vector, $e_{(t \bmod b)}$ the standard basis vector selecting the column to update, $\alpha$ the step size, $\beta_1,\beta_2$ the moment decay rates, and $\epsilon$ the stability constant. Squaring is element-wise, and $V_W^t = C_W^t R_W^{t\top}$ recovers the full second-moment matrix as the outer product of the squared-input and squared-output factors.

Reference: Yiting Chen, Zongwei Huo, Junchi Yan, "IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation", ICLR 2026. https://openreview.net/forum?id=iCT5xdOlJH

---
[Back to the Canon](../README.md)
