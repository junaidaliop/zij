# Cayley SGD

Implements Cayley SGD, momentum SGD constrained to the Stiefel manifold via the Cayley transform.

Cayley SGD optimizes parameters that must stay orthonormal (columns of $X$ with $X^\top X = I$). It accumulates momentum in Euclidean space, projects it onto the tangent space of the manifold as a skew-symmetric matrix $W$, and moves along the resulting curve using the Cayley transform $Y(\alpha) = (I - \tfrac{\alpha}{2}W)^{-1}(I + \tfrac{\alpha}{2}W)X$, which preserves orthonormality exactly. To avoid the matrix inverse, the transform is evaluated by a fixed-point iteration, and an adaptive step size keeps the curve approximation accurate.

$$
\begin{aligned}
m_{t+1} &\leftarrow \beta m_t - g_t \\
\hat{W}_t &= m_{t+1} X_t^\top - \tfrac{1}{2} X_t \left( X_t^\top m_{t+1} X_t^\top \right) \\
W_t &= \hat{W}_t - \hat{W}_t^\top \\
m_{t+1} &\leftarrow W_t X_t \\
\alpha &= \min\left\{ \eta,\ \frac{2q}{\lVert W_t \rVert + \epsilon} \right\} \\
Y^{0} &= X_t + \alpha\, m_{t+1} \\
Y^{i} &= X_t + \tfrac{\alpha}{2} W_t \left( X_t + Y^{i-1} \right), \quad i = 1,\dots,s \\
X_{t+1} &= Y^{s}
\end{aligned}
$$

where $X_t$ is the orthonormal parameter matrix, $g_t = G(X_t)$ is the Euclidean gradient, $m_t$ is the momentum, $\beta$ the momentum coefficient, $W_t$ the skew-symmetric tangent direction, $\eta$ the base learning rate, $q$ a step-size constant (default $0.5$), $s$ the number of fixed-point iterations (default $2$), and $\epsilon$ a small constant for stability.

Reference: Jun Li, Li Fuxin, Sinisa Todorovic, "Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform", ICLR 2020. https://arxiv.org/abs/2002.01113

---
[Back to the Canon](../README.md)
