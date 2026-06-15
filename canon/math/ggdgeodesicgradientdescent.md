# GGD (Geodesic Gradient Descent)

Implements GGD (Geodesic Gradient Descent), a learning-rate-free optimizer that walks along geodesics of the manifold induced by the objective.

Each step lifts the parameters $\theta_t$ to the point $P_t = \mathrm{concat}(\theta_t,\, L(\theta_t; x))$ on the loss hypersurface. The surface normal is $n_t = \mathrm{concat}(g_t,\, -1)$ and the descent tangent is $v_t = \mathrm{concat}(g_t,\, \lVert g_t \rVert_2^2)$. Rather than a learning rate, the method osculates the surface with a sphere of radius $R_t$ centered at $C_t = R_t\, n_t / \lVert n_t \rVert$, scales the step to at most a quarter of the sphere's arc length, and advances along the geodesic via the exponential map. The radius follows a Gaussian (RBF) schedule in the step index $t$, so the effective step size is set by geometry, not a tuned rate.

$$
\begin{aligned}
R_t &= R_0 \, \exp\!\left(-\tfrac{1}{2}\frac{(t-\mu)^2}{\sigma^2}\right), &
v_t &\leftarrow \tfrac{1}{2}\,\pi R_t \, \frac{v_t}{\lVert v_t \rVert}, \\
C_t &= R_t \, \frac{n_t}{\lVert n_t \rVert}, &
\tilde P_t &= P_t - C_t, \\
P_{t+1} &= \cos\!\left(\tfrac{\lVert v_t \rVert}{R_t}\right) \tilde P_t + \frac{R_t \sin\!\left(\tfrac{\lVert v_t \rVert}{R_t}\right)}{\lVert v_t \rVert}\, v_t + C_t, &
\theta_{t+1} &= P_{t+1}[0{:}n].
\end{aligned}
$$

where $\theta_t$ are the $n$-dimensional parameters, $g_t = \nabla_{\theta_t} L(\theta_t; x)$ the gradient, $L$ the loss, $\mathrm{concat}$ stacks into an $(n{+}1)$-dimensional vector, $R_0$ the initial sphere radius, and $\mu,\sigma$ the center and width of the radius schedule; $[0{:}n]$ takes the first $n$ components.

Reference: Liwei Hu, Guangyao Li, Wenyong Wang, Xiaoming Zhang, Yu Xiang, "Geodesic Gradient Descent: A Generic and Learning-rate-free Optimizer on Objective Function-induced Manifolds", arXiv 2026. https://arxiv.org/abs/2603.06651

---
[Back to the Canon](../README.md)
