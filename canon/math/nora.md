# Nora

Implements Nora, a scalable matrix optimizer that projects momentum onto the orthogonal complement of each weight row, then row-normalizes it.

Nora targets matrix-shaped parameters (e.g. Transformer weights). It keeps a single momentum buffer and, for each row of the weight matrix, removes the component of the momentum that is parallel to the corresponding weight row. This row-wise orthogonal projection stabilizes weight norms and angular velocities. Dividing each projected row by its own $L_2$ norm yields a scale-invariant update that approximates structured preconditioning by exploiting the row block-diagonal dominance of the Transformer Hessian, all at $O(mn)$ cost.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
m_{t,i:}^{\perp} &= m_{t,i:} - \frac{\langle m_{t,i:},\, \theta_{t,i:}\rangle}{\lVert \theta_{t,i:}\rVert_2^2}\, \theta_{t,i:} \\
d_{t,i:} &= \frac{m_{t,i:}^{\perp}}{\lVert m_{t,i:}^{\perp}\rVert_2} \\
\theta_{t+1} &= \theta_t - \eta\,\bigl(d_t + \lambda\, \theta_t\bigr)
\end{aligned}
$$

where $\theta$ is the weight matrix with $i$-th row $\theta_{t,i:}$, $g_t$ is the gradient, $m_t$ is momentum with decay $\beta$, $m_{t,i:}^{\perp}$ is the $i$-th momentum row projected onto the orthogonal complement of $\theta_{t,i:}$, $d_t$ is the row-normalized update, $\eta$ is the learning rate, and $\lambda$ is the weight decay.

Reference: Jinghui Yuan, Jiaxuan Zou, Shuo Wang, Yong Liu, Feiping Nie, "Nora: Normalized Orthogonal Row Alignment for Scalable Matrix Optimizer", arXiv 2026. https://arxiv.org/abs/2605.03769

---
[Back to the Canon](../README.md)
