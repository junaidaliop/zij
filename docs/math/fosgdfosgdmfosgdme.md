# FOSGD / FOSGDM / FOSGDME

Implements FOSGD / FOSGDM / FOSGDME, a fractional-order SGD built on a modified Caputo derivative, with momentum and energy variants.

Classical fractional-order gradient descent replaces the integer first derivative with a Caputo fractional derivative, but the standard definition can converge to a fixed point that is not the true extremum. FOSGD fixes this by modifying the Caputo derivative so the fractional gradient is the ordinary gradient $g_t$ multiplied by a memory factor built from the most recent parameter displacement, $\left(|\theta_t-\theta_{t-1}|+\delta\right)^{1-\alpha}$, normalized by $1/\Gamma(2-\alpha)$. As $\alpha\to 1$ the factor collapses and the rule reduces to ordinary SGD; for $\alpha\in(0,1)$ it injects long-memory weighting, and the offset $\delta>0$ keeps the step from stalling when consecutive iterates coincide.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \frac{\eta_t}{\Gamma(2-\alpha)}\left(\left|\theta_t-\theta_{t-1}\right|+\delta\right)^{1-\alpha} g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ is the (typically diminishing) step size, $g_t=\nabla f(\theta_t)$ is the stochastic gradient, $\Gamma(\cdot)$ is the gamma function, $\alpha\in(0,1)$ is the fractional order, and $\delta>0$ is the displacement offset; the operations are taken element-wise (Hadamard product).

FOSGDM extends this by accumulating the fractional gradient into a momentum buffer and driving the parameter step with that buffer, accelerating convergence in the same way ordinary heavy-ball momentum does. FOSGDME further introduces an auxiliary energy variable, in the spirit of the invariant energy quadratization (IEQ) approach for gradient flows: the energy is updated jointly with the parameters and rescales the effective learning rate, yielding unconditional energy stability and improved robustness to the choice of base step size and initialization.

Reference: Xingwen Zhou, Zhenghao You, Weiguo Sun, Dongdong Zhao, Shi Yan, "Fractional-order stochastic gradient descent method with momentum and energy for deep neural networks", Neural Networks 2025. https://doi.org/10.1016/j.neunet.2024.106810

---
[Back to the Canon](../index.md)
