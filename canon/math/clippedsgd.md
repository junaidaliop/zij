# clipped-SGD

Implements clipped-SGD, stochastic gradient descent with norm clipping of the stochastic gradient at each step.

To handle heavy-tailed gradient noise, each (mini-batched) stochastic gradient is rescaled so that its Euclidean norm never exceeds a clipping level $\lambda$, and the standard SGD step is taken on this clipped direction. The clipping is a no-op when the gradient norm is below $\lambda$ and otherwise projects it onto the sphere of radius $\lambda$, which controls the influence of rare large-noise samples; the reported iterate is the running average of the iterates.

$$
\begin{aligned}
\mathrm{clip}(g_t, \lambda) &= \min\!\left(1,\; \frac{\lambda}{\lVert g_t \rVert_2}\right) g_t \\
\theta_{t+1} &= \theta_t - \gamma\, \mathrm{clip}(g_t, \lambda) \\
\bar\theta_N &= \frac{1}{N}\sum_{t=0}^{N-1} \theta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma > 0$ is the stepsize, $g_t$ is the stochastic (mini-batch) gradient at $\theta_t$, $\lambda > 0$ is the clipping level, $\lVert \cdot \rVert_2$ is the Euclidean norm, and $\bar\theta_N$ is the averaged output iterate over $N$ steps.

Reference: Eduard Gorbunov, Marina Danilova, Alexander Gasnikov, "Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping", NeurIPS 2020. https://arxiv.org/abs/2005.10785

---
[Back to the Canon](../README.md)
