# SubTrack++

Implements SubTrack++, memory-efficient full-parameter LLM training that tracks the gradient subspace as a geodesic on the Grassmannian instead of recomputing the SVD.

Like GaLore, SubTrack++ keeps the optimizer state inside a low-rank subspace $S_t$ (an $m \times r$ orthonormal basis) and projects gradients into and out of it. Rather than periodically rebuilding the basis from a fresh SVD, it treats the basis as a point on the Grassmann manifold and moves it along a geodesic in the direction that reduces the subspace projection error. The error direction is obtained cheaply: a least-squares fit of the current gradient onto the old basis gives a residual $R$ in the orthogonal complement, and a rank-1 SVD of the associated tangent vector $\nabla F$ yields the geodesic's rotation. The basis is refreshed only every $k$ steps; in between it is held fixed.

At each subspace-update step the gradient $G_t$ is least-squares projected onto the previous basis, the residual feeds the rank-1 tangent estimate, and the basis rotates by the Grassmannian geodesic. The projected gradient is passed through a base optimizer (Adam) whose internal statistics adapt to the rotating subspace, then projected back to full dimension for the weight update.

$$
\begin{aligned}
G_{lr} &= \arg\min_{A}\ \lVert S_{t-1} A - G_t \rVert^2, \qquad R = G_t - S_{t-1} G_{lr} \\
\nabla F &= -2 R\, G_{lr}^{\top} \approx \widehat{U}_F\, \widehat{\Sigma}_F\, \widehat{V}_F^{\top} \\
S_t &= \begin{pmatrix} S_{t-1}\widehat{V}_F & \widehat{U}_F \end{pmatrix} \begin{pmatrix} \cos(\widehat{\Sigma}_F\, \eta) \\ \sin(\widehat{\Sigma}_F\, \eta) \end{pmatrix} \widehat{V}_F^{\top} + S_{t-1}\bigl(I - \widehat{V}_F \widehat{V}_F^{\top}\bigr) \\
\widehat{G}_t &= S_t\, \rho_t\!\left(S_t^{\top} G_t\right) \\
W_{t+1} &= W_t - \mu\, \widehat{G}_t
\end{aligned}
$$

where $W$ are the weights, $G_t$ the $m \times n$ gradient, $S_t$ the $m \times r$ orthonormal subspace basis ($r \ll m \le n$), $G_{lr}$ the least-squares coefficients, $R$ the projection residual, $\nabla F$ the tangent vector on the Grassmannian with rank-1 SVD $\widehat{U}_F \widehat{\Sigma}_F \widehat{V}_F^{\top}$, $\eta$ the geodesic step size, $\rho_t(\cdot)$ the entry-wise base-optimizer (Adam) transform applied in the subspace, $\mu$ the learning rate, and $k$ the interval between subspace updates (on non-update steps $S_t = S_{t-1}$).

Reference: Sahar Rajabi, Nayeema Nonta, Sirisha Rambhatla, "SubTrack++: Gradient Subspace Tracking for Scalable LLM Training", arXiv 2025. https://arxiv.org/abs/2502.01586

---
[Back to the Canon](../README.md)
