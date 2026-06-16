# Accelerated GRAAL

Implements Accelerated GRAAL, a parameter-free accelerated gradient method that fuses Nesterov coupling with the Golden Ratio adaptive line search.

The method runs a GRAAL-style explicit gradient step whose step size $\eta_k$ is set automatically from a local curvature estimate (a Bregman-to-gradient-gap ratio), so no Lipschitz constant or learning rate is tuned. Around that step it layers a Nesterov-type acceleration: a GRAAL extrapolated point is coupled with an averaged sequence, and the coupling weights $\alpha_k, \beta_k$ adapt to the accumulated step sizes $H_k = \sum_i \eta_i$. This yields the optimal accelerated convergence rate for smooth convex problems while remaining fully adaptive.

$$
\begin{aligned}
\alpha_{k+1} &= \frac{(1+\gamma)\eta_k}{H_k + (1+\gamma)\eta_k} \\
x_{k+1} &= x_k - \eta_k\, \nabla f(\tilde{x}_k) \\
\bar{x}_{k+1} &= \beta_k \tilde{x}_k + (1-\beta_k)\bar{x}_k \\
\hat{x}_{k+1} &= x_{k+1} + \theta\,(x_{k+1} - x_k) \\
\tilde{x}_{k+1} &= \alpha_{k+1}\hat{x}_{k+1} + (1-\alpha_{k+1})\bar{x}_{k+1} \\
\lambda_{k+1} &= \min\!\big\{\Lambda(\bar{x}_{k+1};\tilde{x}_k),\ \Lambda(\bar{x}_{k+1};\tilde{x}_{k+1})\big\} \\
\eta_{k+1} &= \min\!\left\{(1+\gamma)\eta_k,\ \frac{\nu\, H_{k-1}\, \lambda_{k+1}}{\eta_{k-1}}\right\} \\
H_{k+1} &= H_k + \eta_{k+1}, \qquad \beta_{k+1} = \frac{\eta_{k+1}}{\alpha_{k+1} H_{k+1}}
\end{aligned}
$$

where $\theta,\gamma,\nu>0$ satisfy $4\nu\theta(1+\gamma)^2=\gamma$, the curvature ratio is $\Lambda(x;z)=2 B_f(x;z)/\lVert\nabla f(x)-\nabla f(z)\rVert^2$ (and $+\infty$ when $\nabla f(x)=\nabla f(z)$) with Bregman divergence $B_f(x;z)=f(x)-f(z)-\langle\nabla f(z),\,x-z\rangle$, $H_k=\sum_{i\le k}\eta_i$ is the accumulated step size, $\bar{x}_k$ is the returned averaged iterate, and the recursion is initialized by $\alpha_0=\beta_0=1$, $H_0=H_{-1}=\eta_{-1}=\eta_0$, $\tilde{x}_0=\bar{x}_0=x_0$.

Reference: Ekaterina Borodich, Dmitry Kovalev, "Nesterov Finds GRAAL: Optimal and Adaptive Gradient Method for Convex Optimization", arXiv 2025. https://arxiv.org/abs/2507.09823

---
[Back to the Canon](../index.md)
