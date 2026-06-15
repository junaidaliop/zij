# IFNSO

Implements IFNSO, an iteration-free Newton-Schulz orthogonalization for Muon-style updates.

Muon orthogonalizes the momentum matrix by repeatedly applying the odd Newton-Schulz polynomial $X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k$. IFNSO collapses this iterative loop into a single composite polynomial: it drops the insignificant terms of the unrolled iteration and fits a polynomial with learnable coefficients $a_k$, so one matrix evaluation drives all singular values toward $1$. The resulting orthogonalized momentum is then used as a Muon update.

For a matrix $X$ (the normalized momentum) the unified map is

$$
\begin{aligned}
Y &= X + \sum_{k=1}^{N-1} a_k\,(I - X X^\top)^{2^{k-1}} X + \left[\, e^{1/2}\bigl(2^{N/2}-1\bigr) - \sum_{k=1}^{N-1} a_k \right] (I - X X^\top)^{2^{N-1}} X, \\
m_t &= \beta\, m_{t-1} + g_t, \\
\theta_t &= \theta_{t-1} - \eta\, Y_t,
\end{aligned}
$$

where $X$ is the momentum $m_t$ scaled so its singular values lie in $[0,1]$, $I$ is the identity, $N$ is the polynomial depth (recommended $N=14$), $a_k$ are coefficients optimized to enforce the orthogonality constraint, $Y_t$ is the orthogonalized momentum (equivalently $Y = U\,\mathrm{diag}(f(\sigma_1),\dots,f(\sigma_m))\,V^\top$ with scalar map $f(x) = x + \sum_{k=1}^{N-1} a_k\, x(1-x^2)^{2^{k-1}} + b\, x(1-x^2)^{2^{N-1}}$), $g_t$ is the gradient, $\beta$ the momentum factor, and $\eta$ the learning rate.

Reference: Chen Hu, Qianxi Zhao, Xiaochen Yuan, Hong Zhang, Ding Yuan, Yanbin Wu, Xiying Li, "IFNSO: Iteration-Free Newton-Schulz Orthogonalization", arXiv 2026. https://arxiv.org/abs/2602.02500

---
[Back to the Canon](../README.md)
