# Focal-SAM

Implements Focal-SAM, a sharpness-aware optimizer that reweights class-wise sharpness with focal weights to favor tail classes in long-tailed recognition.

Standard SAM penalizes the worst-case loss in a neighborhood of radius $\rho$, treating every class equally. Focal-SAM instead applies a focal weight $(1-\pi_i)^\gamma$ to each class-wise sharpness term, so under-represented (tail) classes receive a stronger flatness penalty. The objective adds this focal sharpness term to the empirical loss, and the perturbation is computed from the focally weighted loss $L^\gamma(\theta)=\sum_{i=1}^{C}(1-\pi_i)^\gamma L^i(\theta)$. A first-order approximation yields an update needing only three backpropagations per step, independent of the number of classes.

$$
\begin{aligned}
\hat{\epsilon}(\theta_t) &= \rho \, \frac{\nabla_\theta L^\gamma(\theta_t)}{\lVert \nabla_\theta L^\gamma(\theta_t) \rVert_2}, \\
g_1 &= \nabla_\theta L^\gamma(\theta)\big|_{\theta_t + \hat{\epsilon}(\theta_t)}, \\
g_2 &= \nabla_\theta \big[\, L(\theta) - \lambda \, L^\gamma(\theta) \,\big]\big|_{\theta_t}, \\
\theta_{t+1} &= \theta_t - \eta \,(\lambda \, g_1 + g_2).
\end{aligned}
$$

where $L(\theta)$ is the empirical loss, $L^\gamma(\theta)=\sum_{i=1}^{C}(1-\pi_i)^\gamma L^i(\theta)$ is the focally weighted loss with class-wise loss $L^i$, $\pi_i$ the proportion of class $i$, $\gamma$ the focal exponent, $\rho$ the perturbation radius, $\lambda$ the sharpness weight, and $\eta$ the learning rate.

Reference: Sicong Li, Qianqian Xu, Zhiyong Yang, Zitai Wang, Linchao Zhang, Xiaochun Cao, Qingming Huang, "Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification", ICML 2025. https://arxiv.org/abs/2505.01660

---
[Back to the Canon](../README.md)
