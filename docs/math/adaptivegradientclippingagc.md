# Adaptive Gradient Clipping (AGC)

Implements Adaptive Gradient Clipping (AGC), gradient clipping scaled by the unit-wise ratio of gradient norm to weight norm.

AGC clips gradients based on the ratio between the norm of the gradient and the norm of the corresponding weights, applied per unit (each row of a weight matrix, or each filter of a convolution). When this ratio exceeds a threshold $\lambda$, the gradient is rescaled so the ratio equals $\lambda$; otherwise it is left unchanged. This makes the clipping threshold adapt to the scale of each layer's parameters, which lets normalizer-free networks train stably at large batch sizes.

For the $i$-th unit of layer $\ell$, with weight row $W^\ell_i$ and gradient row $G^\ell_i$:

$$
\begin{aligned}
G^\ell_i \;\rightarrow\;
\begin{cases}
\lambda \dfrac{\lVert W^\ell_i \rVert_F^\star}{\lVert G^\ell_i \rVert_F}\, G^\ell_i & \text{if } \dfrac{\lVert G^\ell_i \rVert_F}{\lVert W^\ell_i \rVert_F^\star} > \lambda, \\
G^\ell_i & \text{otherwise,}
\end{cases}
\qquad
\lVert W^\ell_i \rVert_F^\star = \max\!\big(\lVert W^\ell_i \rVert_F,\; \epsilon\big).
\end{aligned}
$$

where $\lVert \cdot \rVert_F$ is the Frobenius norm taken unit-wise (over the fan-in dimensions of each row/filter), $\lambda$ is the clipping threshold (typically $0.01$), and $\epsilon = 10^{-3}$ floors the weight norm so zero-initialized units are not always clipped. The clipped gradient is then passed to the base optimizer.

Reference: Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan, "High-Performance Large-Scale Image Recognition Without Normalization", ICML 2021. https://arxiv.org/abs/2102.06171

---
[Back to the Canon](../index.md)
