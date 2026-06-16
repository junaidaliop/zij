# RotoGrad

Implements RotoGrad, a multitask learning method that homogenizes both the magnitude and the direction of per-task gradients on a shared feature space.

RotoGrad addresses gradient conflict in hard-parameter-sharing networks. For magnitude, it rescales each task gradient $G_k$ (the batch matrix of gradients of task loss $L_k$ with respect to the shared feature $z$) to a common target magnitude $C$, computed as a convex combination weighted by each task's relative convergence. For direction, it inserts a learned rotation matrix $R_k \in \mathrm{SO}(d)$ per task that rotates the shared feature before it enters the task head, and trains each $R_k$ to align the task's feature-space gradient with the batch-averaged direction. The network (follower) and the rotations (leader) form a Stackelberg game, so the rotations are optimized with a slower learning rate.

$$
\begin{aligned}
U_k &= \frac{G_k}{\lVert G_k \rVert}, \qquad
\alpha_k = \frac{\lVert G_k \rVert / \lVert G_k^0 \rVert}{\sum_i \lVert G_i \rVert / \lVert G_i^0 \rVert}, \qquad
C = \sum_k \alpha_k \lVert G_k \rVert \\
\theta &\leftarrow \theta - \eta\, C \sum_k U_k \\
v_n &= \frac{1}{K} \sum_k u_{n,k}, \qquad
\mathcal{L}^k_{\mathrm{rot}} = -\sum_n \big\langle R_k^{\top}\, \tilde{g}_{n,k},\; v_n \big\rangle \\
R_k &\leftarrow R_k - \eta_{\mathrm{rot}}\, \nabla_{R_k} \mathcal{L}^k_{\mathrm{rot}}, \qquad \eta_{\mathrm{rot}} = o(\eta)
\end{aligned}
$$

where $G_k$ stacks the per-sample gradients $g_{n,k} = \nabla_z L_k$ of task $k$ over the batch, $G_k^0$ is that gradient at $t=0$, $\lVert\cdot\rVert$ is the Frobenius norm, $u_{n,k}$ is the row of $U_k$ for sample $n$, $\tilde{g}_{n,k} = \nabla_{r_k} L_k$ is the gradient with respect to the rotated feature $r_k = R_k z$, $K$ is the number of tasks, $\eta$ and $\eta_{\mathrm{rot}}$ are the network and rotation learning rates, and each $R_k$ is kept on $\mathrm{SO}(d)$ by parametrizing it through the exponential map on the Lie algebra of $\mathrm{SO}(d)$.

Reference: Adrián Javaloy, Isabel Valera, "RotoGrad: Gradient Homogenization in Multitask Learning", ICLR 2022. https://arxiv.org/abs/2103.02631

---
[Back to the Canon](../index.md)
