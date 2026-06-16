# SGLBO

Implements SGLBO (Stochastic Gradient Line Bayesian Optimization), a shot-efficient optimizer for parameterized quantum circuits.

SGLBO decouples the choice of update direction from the choice of step size. At each iteration it estimates the gradient of the cost $f$ with respect to the circuit parameters via the parameter-shift rule, fixing the descent direction $-g_t$. It then restricts the cost to the one-dimensional line through $\theta_t$ along that direction and uses Bayesian optimization (a Gaussian-process surrogate with Thompson sampling) to locate the step size $\eta$ that minimizes the cost on the line, which is far cheaper in measurement shots than a high-dimensional search. The optimizer pairs this with $\alpha$-suffix averaging of the final iterates to suppress statistical and hardware noise.

$$
\begin{aligned}
g_t &= \nabla_\theta f(\theta_t), \qquad
\frac{\partial f(\theta)}{\partial \theta_i} = \tfrac{1}{2}\left[\, f\!\left(\theta + \tfrac{\pi}{2}e_i\right) - f\!\left(\theta - \tfrac{\pi}{2}e_i\right) \right], \\
\eta_t^{*} &= \arg\min_{\eta \in [-\eta_{\max},\,\eta_{\max}]} f(\theta_t - \eta\, g_t), \\
\theta_{t+1} &= \theta_t - \eta_t^{*}\, g_t, \\
\bar{\theta}_{\alpha,T} &= \frac{1}{\alpha T} \sum_{t=(1-\alpha)T-1}^{T-1} \theta_t .
\end{aligned}
$$

where $\theta$ are the circuit parameters, $g_t$ is the parameter-shift gradient estimate, $e_i$ is the $i$-th unit vector, $\eta_t^{*}$ is the step size chosen by Bayesian optimization over the line $\{\theta_t - \eta\, g_t : \eta \in [-\eta_{\max}, \eta_{\max}]\}$, $\eta_{\max}$ bounds the line search, $T$ is the total number of iterations, and $\alpha \in (0,1]$ sets the fraction of final iterates averaged in the suffix-averaged output $\bar{\theta}_{\alpha,T}$.

Reference: Shiro Tamiya, Hayata Yamasaki, "Stochastic Gradient Line Bayesian Optimization for Efficient Noise-Robust Optimization of Parameterized Quantum Circuits", npj Quantum Information 2022. https://www.nature.com/articles/s41534-022-00592-6

---
[Back to the Canon](../index.md)
