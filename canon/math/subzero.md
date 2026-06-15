# SubZero

Implements SubZero, a zeroth-order fine-tuning method that perturbs each weight matrix inside a layer-wise low-rank random subspace.

Classic ZO estimators such as MeZO perturb the full parameter tensor with isotropic Gaussian noise, whose variance grows with the model dimension. SubZero instead restricts each weight matrix $\theta$ of size $m\times n$ to a rank-$r$ subspace: it draws column-orthonormal factors $U$ and $V$ (via QR decomposition of Gaussian matrices), forms the perturbation $U Z V^\top$ from a small $r\times r$ Gaussian core $Z$, and estimates the gradient by a symmetric two-point finite difference along that direction. A single scalar loss difference, shared across all layers, scales the low-rank direction to give the estimate, and the subspaces are refreshed lazily every $T_0$ steps to amortize the QR cost. The result lowers gradient-estimate variance and matches inference memory, since only the small factors and one scalar are kept.

$$
\begin{aligned}
U_i &\in \mathbb{R}^{m_i\times r},\quad V_i \in \mathbb{R}^{n_i\times r}\ \ (\text{column-orthonormal, refreshed every } T_0 \text{ steps}),\quad Z_i \sim \mathcal{N}(0, I)_{r\times r}, \\
\rho_t &= \frac{\mathcal{L}(\mathcal{W}_t + \epsilon\, \tilde{\mathcal{Z}}_t; \mathcal{B}_t) - \mathcal{L}(\mathcal{W}_t - \epsilon\, \tilde{\mathcal{Z}}_t; \mathcal{B}_t)}{2\epsilon},\qquad \tilde{Z}_i = U_i Z_i V_i^\top, \\
\widehat{\nabla}\mathcal{L}(\theta_{i,t}; \mathcal{B}_t) &= \rho_t\, U_i Z_i V_i^\top, \\
\theta_{i,t+1} &= \theta_{i,t} - \eta_t\, \rho_t\, U_i Z_i V_i^\top,
\end{aligned}
$$

where $\theta_i$ is the weight matrix of layer $i$, $\mathcal{W}$ collects all such matrices, $\mathcal{B}_t$ is the minibatch, $U_i, V_i$ are the column-orthonormal subspace factors with rank $r \ll \min\{m_i, n_i\}$, $Z_i$ is the $r\times r$ Gaussian core, $\tilde{Z}_i = U_i Z_i V_i^\top$ is the low-rank perturbation direction, $\epsilon$ is the perturbation (smoothing) scale, $\rho_t$ is the scalar finite-difference coefficient evaluated jointly over all layers, $\eta_t$ is the learning rate, and $T_0$ is the lazy interval at which $U_i, V_i$ are regenerated; the base update uses plain SGD with no momentum.

Reference: Ziming Yu, Pan Zhou, Sike Wang, Jia Li, Mi Tian, Hua Huang, "Zeroth-Order Fine-Tuning of LLMs in Random Subspaces", ICCV 2025. https://arxiv.org/abs/2410.08989

---
[Back to the Canon](../README.md)
