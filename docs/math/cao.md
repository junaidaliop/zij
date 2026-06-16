# CAO

Implements CAO (Curvature-Adaptive Optimization), preconditioning gradients inside a periodically refreshed top-$k$ Hessian eigenspace.

Every few steps CAO sketches the leading $k$ eigenpairs of the Hessian using Hessian-vector products and block-Lanczos, forming a rank-$k$ spectral approximation $B_t$. A damped inverse of this sketch preconditions the gradient: directions captured by the sketch are rescaled by their curvature, while the orthogonal complement is left first-order through the damping term. This keeps second-order behavior where curvature is informative without the cost of a full Hessian.

$$
\begin{aligned}
B_t &= \sum_{i=1}^{k} \lambda_{i,t}\, v_{i,t} v_{i,t}^{\top} \\
P_t &= (B_t + \eta I)^{-1} \\
\theta_{t+1} &= \theta_t - \alpha\, P_t\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t = \nabla f(\theta_t)$ the gradient, $\alpha$ the step size, and $(\lambda_{i,t}, v_{i,t})$ the top-$k$ Hessian eigenpairs of the current spectral sketch $B_t$. The damping $\eta > 0$ makes $P_t$ scale steps by $(\lambda_{i,t} + \eta)^{-1}$ along each captured eigenvector $v_{i,t}$ and by $\eta^{-1}$ on the orthogonal complement.

Reference: Du Wenzhang, "CAO: Curvature-Adaptive Optimization via Periodic Low-Rank Hessian Sketching", arXiv preprint 2025. https://arxiv.org/abs/2511.12548

---
[Back to the Canon](../index.md)
