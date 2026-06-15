# SAM-Polyak (Adaptive SAM with Polyak step size)

Implements SAM-Polyak, Sharpness-Aware Minimization with an adaptive Polyak-type step size.

SAM perturbs the weights toward the worst-case nearby point and then descends on the gradient evaluated there. The contribution of this method is to replace SAM's hand-tuned learning rate with a closed-form Polyak step size: by upper-bounding $\|\theta_{t+1}-\theta^\star\|^2$ and minimizing that bound, the optimal step size is expressed purely from quantities already available at the perturbed point, the loss and the gradient. When the sharpness radius $\rho=0$ the rule collapses to the classical (stochastic) Polyak step size.

For the normalized SAM update, the deterministic step size is $\gamma_t=[f(e_t)-f^\star-\langle g_t,\,e_t-\theta_t\rangle]_+/\|g_t\|^2$; the stochastic (mini-batch) form caps it at $\gamma_b$ and uses a lower bound $\ell^\star$ on the loss (typically $0$).

$$
\begin{aligned}
e_t &= \theta_t + \rho\,\frac{\nabla f(\theta_t)}{\|\nabla f(\theta_t)\|}, \qquad g_t = \nabla f(e_t),\\
\gamma_t &= \min\!\left(\frac{\left[\,f(e_t) - \ell^\star - \langle g_t,\, e_t - \theta_t\rangle\,\right]_+}{\|g_t\|^2},\; \gamma_b\right),\\
\theta_{t+1} &= \theta_t - \gamma_t\, g_t.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\rho$ is the sharpness radius, $e_t$ the perturbed (ascent) point, $g_t=\nabla f(e_t)$ the gradient at the perturbed point, $\ell^\star$ a lower bound on the loss (take $\ell^\star=0$ for non-negative losses; $f^\star=\inf f$ in the full-batch case), $\gamma_b>0$ a cap on the step size, and $[z]_+=\max(z,0)$. The unnormalized variant (USAM-SPS) uses $e_t=\theta_t+\rho\,\nabla f(\theta_t)$, for which the numerator becomes $f(e_t)-\ell^\star-\rho\langle g_t,\nabla f(\theta_t)\rangle$ and the $[\cdot]_+$ safeguard is provably redundant when $\rho\le 1/L$.

Reference: Dimitris Oikonomou, Nicolas Loizou, "Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler", ICML 2026. https://arxiv.org/abs/2606.01827

---
[Back to the Canon](../README.md)
