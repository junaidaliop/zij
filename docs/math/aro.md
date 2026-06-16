# ARO

Implements ARO (Adaptively Rotated Optimization), a matrix optimizer that performs normed steepest descent in an adaptively rotated coordinate system.

ARO treats gradient rotation as a first-class design principle. Rather than orthogonalizing or whitening the gradient in fixed coordinates, it maintains a momentum buffer of the matrix-valued gradient and applies a base projection $f_t$ (the inner optimizer, e.g. SignGD, SinkGD, or Adam) inside a rotated frame. The rotation $R_t$ is chosen by a norm-informed policy and updated each step as the orthonormal factor of a QR decomposition, making the rotation optimizer-aware.

$$
\begin{aligned}
M_t &= \beta M_{t-1} + (1 - \beta) G_t, \\
R_t &= \mathrm{QR}\!\left( M_t\, f_t(R_{t-1}^{\top} M_t)^{\top} \right), \\
\Delta W_t &= -\eta\, R_t\, f_t(R_t^{\top} M_t), \\
W_t &= W_{t-1} + \Delta W_t.
\end{aligned}
$$

where $W$ is the weight matrix, $\eta$ the step size, $G_t$ the gradient matrix, $M_t$ the EMA momentum buffer with decay $\beta$, $R_t \in \mathrm{SO}(m)$ the rotation matrix, $f_t$ the stateful base-optimizer projection, and $\mathrm{QR}(\cdot)$ the orthonormal (Q) factor of its matrix argument.

Reference: Wenbo Gong, Javier Zazo, Qijun Luo, Puqian Wang, James Hensman, Chao Ma, "ARO: A New Lens On Matrix Optimization For Large Models", arXiv 2026. https://arxiv.org/abs/2602.09006

---
[Back to the Canon](../index.md)
