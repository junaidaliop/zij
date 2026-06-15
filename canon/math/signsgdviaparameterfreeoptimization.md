# Sign-SGD via Parameter-Free Optimization

Implements Sign-SGD via Parameter-Free Optimization, a tuning-free Sign-SGD whose stepsize is set automatically from observed gradient and iterate differences.

Sign-SGD updates along the sign of the gradient, but its performance hinges on a stepsize that normally depends on unknown problem constants. This method removes that dependence: at each step it estimates the local smoothness from accumulated gradient and iterate differences and an upper bound on the suboptimality gap, combining them into an adaptive stepsize $\gamma_t$. No learning rate is tuned.

$$
\begin{aligned}
\lambda_t &= \left( \sum_{i=0}^{t-1} \frac{\lVert g_{i+1} - g_i \rVert_1}{\lVert \theta_{i+1} - \theta_i \rVert_\infty} \right)^{-1/2} \\
\tilde{d}_t &= \sum_{i=0}^{t-1} \gamma_i \langle g_{i+1}, \mathrm{sign}(g_i) \rangle, \qquad d_t = \max(d_{t-1}, \tilde{d}_t) \\
\gamma_t &= \lambda_t \sqrt{d_t} \\
\theta_{t+1} &= \theta_t - \gamma_t \, \mathrm{sign}(g_t)
\end{aligned}
$$

where $\theta_t$ are the parameters, $g_t = \nabla f(\theta_t)$ the gradient, $\lambda_t$ the inverse-square-root smoothness estimate built from $\ell_1$ gradient differences over $\ell_\infty$ iterate differences, $d_t$ a monotone estimate of the suboptimality gap, and $\gamma_t$ the resulting parameter-free stepsize. An alternative replaces $\sqrt{d_t}$ with $\sqrt{f(\theta_0) - \tilde{f}}$, where $\tilde{f}$ is a known lower bound on $f(\theta^\ast)$.

Reference: Daniil Medyakov, Sergey Stanko, Gleb Molodtsov, Philip Zmushko, Grigoriy Evseev, Egor Petrov, Aleksandr Beznosikov, "Sign-SGD via Parameter-Free Optimization", arXiv 2025. https://arxiv.org/abs/2506.03725

---
[Back to the Canon](../README.md)
