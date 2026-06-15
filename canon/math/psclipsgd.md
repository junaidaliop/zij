# PS-Clip-SGD

Implements PS-Clip-SGD, SGD with per-sample gradient clipping for heavy-tailed noise.

Instead of clipping the aggregated mini-batch gradient, PS-Clip-SGD clips each individual sample gradient before averaging. The $k$-th sample in the batch is scaled by a factor that caps its norm at a sample-dependent threshold $\alpha k^{1/\beta}$, which yields optimal non-convex convergence rates under heavy-tailed gradient noise with bounded $p$-th moment. In practice the clipping parameters are set to $\alpha = \sigma$ and $\beta = p$.

$$
\begin{aligned}
\gamma_t^{(k)} &= \min\!\left\{1,\ \frac{\alpha\, k^{1/\beta}}{\lVert \nabla f(\theta_t, \xi_t^{(k)}) \rVert}\right\} \\
g_t &= \frac{1}{n}\sum_{k=1}^{n} \gamma_t^{(k)}\, \nabla f(\theta_t, \xi_t^{(k)}) \\
\theta_{t+1} &= \theta_t - \eta_t\, g_t
\end{aligned}
$$

where $\nabla f(\theta_t, \xi_t^{(k)})$ is the gradient on the $k$-th sample of the batch of size $n$, $\gamma_t^{(k)}$ is its per-sample clipping factor, $\alpha,\beta > 0$ are clipping parameters, and $\eta_t$ is the step size.

Reference: Davide Nobile, Philipp Grohs, "Robust and Fast Training via Per-Sample Clipping", arXiv 2026. https://arxiv.org/abs/2605.02701

---
[Back to the Canon](../README.md)
