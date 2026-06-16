# C-FOG

Implements C-FOG, Caputo's fractional-order gradient descent with a self-organizing per-dimension step size.

C-FOG starts from the observation that gradient sign descent rescales each coordinate's step by the inverse gradient magnitude, which keeps the search consistent near an extremum. Replacing the integer derivative by a Caputo fractional derivative and keeping only its leading term turns that idea into a multivariate update whose effective step size carries a memory of the most recent parameter displacement. The scaling factor $|\theta_t-\theta_{t-1}|^{1-\alpha}$ is applied elementwise, so each dimension self-organizes its own step: when $1<\alpha\le 2$ coordinates that moved a lot are damped and small movers are amplified, balancing the iterates and guaranteeing no divergence even for large $\eta$. The order $\alpha$ is the single extra degree of freedom; $\alpha=1$ recovers natural gradient descent and $\alpha=2$ approximates gradient sign descent, while the named C-FOG regime is $1<\alpha<2$.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta\,\nabla_\theta f(\theta_t)\,\bigl|\theta_t-\theta_{t-1}\bigr|^{1-\alpha}, \qquad (0<\alpha\le 2) \\
{}^{C}_{a}D^{p}_{t}f(t) &= \frac{1}{\Gamma(n-p)}\int_{a}^{t}(t-\tau)^{n-p-1}f^{(n)}(\tau)\,d\tau, \qquad (n-1<p\le n)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $\nabla_\theta f(\theta_t)$ is the gradient, $\alpha$ is the fractional order, $|\theta_t-\theta_{t-1}|^{1-\alpha}$ is the self-organizing scaling factor computed elementwise, and the second line is the Caputo fractional derivative of order $p$ from which the leading-term update is obtained, with $\Gamma(\cdot)$ the gamma function.

Reference: Sunfu Tan, Ni Zhang, Yifei Pu, "Self-Organizing Optimization Based on Caputo's Fractional Order Gradients", Fractal and Fractional 2024. https://doi.org/10.3390/fractalfract8080451

---
[Back to the Canon](../index.md)
