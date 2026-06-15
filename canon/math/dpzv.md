# DPZV

Implements DPZV, a zeroth-order optimizer for differentially private vertical federated learning.

DPZV removes backpropagation from vertical federated training: each client perturbs its embedding network along a random direction and sends the two perturbed embeddings to the server, which returns a scalar finite-difference of the loss. This two-point estimate plays the role of a directional gradient. To guarantee privacy, the per-sample scalar is clipped and Gaussian noise is added on the server before it is broadcast back, so each client never sees raw per-sample gradients and updates its parameters using only the noised, clipped scalar times the random direction.

$$
\begin{aligned}
\delta_{m,i}^{t} &= \frac{\tilde f(w_0, h(\theta_m + \lambda u_m;\xi_{m,i})) - \tilde f(w_0, h(\theta_m - \lambda u_m;\xi_{m,i}))}{\lambda} \\
\Delta_m^{t} &= \frac{1}{B}\sum_{i\in I_m} \mathrm{clip}_C\!\left(\delta_{m,i}^{t}\right) + z_m^{t}, \qquad z_m^{t}\sim\mathcal{N}(0,\sigma^2) \\
\theta_m &\leftarrow \theta_m - \eta\,\Delta_m^{t}\, u_m
\end{aligned}
$$

where $\theta_m$ are client $m$'s embedding-network parameters, $\eta$ the learning rate, $\lambda$ the smoothing radius, $u_m$ a direction sampled uniformly from the unit sphere, $h(\cdot)$ the client embedding map on minibatch sample $\xi_{m,i}$, $\tilde f(w_0,\cdot)$ the server loss with server model $w_0$, $\mathrm{clip}_C(\cdot)$ rescaling to norm at most $C$, $B$ the batch size, and $z_m^{t}$ Gaussian privacy noise with $\sigma = \tfrac{2C\sqrt{T}}{D\mu}$ for dataset size $D$, total steps $T$, and privacy level $\mu$.

Reference: Jianing Zhang, Evan Chen, Chaoyue Liu, Christopher G. Brinton, "DPZV: Resource Efficient ZO Optimization For Differentially Private VFL", arXiv 2025. https://arxiv.org/abs/2502.20565

---
[Back to the Canon](../README.md)
