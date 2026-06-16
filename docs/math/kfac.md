# K-FAC

Implements K-FAC, an approximate natural-gradient method using a Kronecker-factored estimate of the Fisher information matrix.

The Fisher matrix of a neural network is too large to invert directly, so K-FAC approximates it as block-diagonal over layers and factors each block as a Kronecker product of two small matrices: the covariance of a layer's inputs and the covariance of the gradients of its pre-activations. Because $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$, the per-layer natural-gradient update reduces to two small matrix solves applied to the gradient reshaped as a matrix, instead of one huge solve. A Tikhonov damping term $\lambda$ keeps the curvature estimate well-conditioned.

For layer $i$ with weights $W_i$, pre-activation gradient $g_i$ and input activations $\bar a_{i-1}$ (augmented with a homogeneous coordinate), the weight gradient is the outer product $\mathcal{D}W_i = g_i\,\bar a_{i-1}^{\top}$, and the update is:

$$
\begin{aligned}
A_{i-1} &= \mathbb{E}\!\left[\bar a_{i-1}\,\bar a_{i-1}^{\top}\right], \qquad
G_i = \mathbb{E}\!\left[g_i\,g_i^{\top}\right], \\
F_i &\approx A_{i-1} \otimes G_i, \\
U_i &= \left(G_i + \lambda I\right)^{-1}\, \nabla_{W_i} \left(A_{i-1} + \lambda I\right)^{-1}, \\
W_i &\leftarrow W_i - \eta\, U_i.
\end{aligned}
$$

where $\bar a_{i-1}$ are the (homogeneous) inputs to layer $i$, $g_i = \partial \mathcal{L}/\partial s_i$ is the gradient w.r.t. the pre-activations $s_i$, $A_{i-1}$ and $G_i$ are the Kronecker factors, $\nabla_{W_i}=\mathcal{D}W_i$ is the loss gradient reshaped as a matrix, $\lambda$ is the damping coefficient, and $\eta$ is the learning rate. Vectorizing recovers the natural-gradient form $\theta \leftarrow \theta - \eta\,F^{-1}\nabla h(\theta)$ since $\mathrm{vec}(U_i) = (A_{i-1}\otimes G_i)^{-1}\,\mathrm{vec}(\nabla_{W_i})$.

Reference: James Martens, Roger Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature", ICML 2015. https://arxiv.org/abs/1503.05671

---
[Back to the Canon](../index.md)
