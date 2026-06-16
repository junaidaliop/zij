# adaNAPG

Implements adaNAPG, an accelerated proximal gradient method with adaptive sampling for stochastic composite minimization of $f(\theta) + h(\theta)$.

The method targets composite objectives where $f$ is smooth (accessed only through a stochastic gradient estimate) and $h$ is a possibly nonsmooth regularizer reached via its proximal operator. It combines a Nesterov extrapolation with adaptive sampling: at each step the gradient estimate $\hat{g}_t = \hat{\nabla} f(y_t)$ is built from a mini-batch whose size grows until two inner-product/variance tests are met, so the sampling effort tracks the gradient mapping. The step size $\eta$ is fixed from the smoothness constant and the test tolerances, and the momentum weights $\pi_t$ follow a recursion that recovers the strongly convex acceleration rate.

$$
\begin{aligned}
\eta &= \frac{1}{L(\theta^2 + \nu^2 + 1)}, \qquad q = \mu\eta, \\
x_{t+1} &= \mathrm{prox}_{\eta h}\!\left(y_t - \eta\, \hat{g}_t\right), \\
\pi_{t+1}^2 &- (q - \pi_t^2)\,\pi_{t+1} - \pi_t^2 = 0, \\
y_{t+1} &= x_{t+1} + \frac{\pi_t(1 - \pi_t)}{\pi_t^2 + \pi_{t+1}}\,(x_{t+1} - x_t).
\end{aligned}
$$

where $x_t$ are the iterates, $y_t$ the extrapolation points, $\hat{g}_t = \hat{\nabla} f(y_t)$ the adaptively sampled stochastic gradient at $y_t$, $\mathrm{prox}_{\eta h}$ the proximal operator of $h$, $L$ the smoothness constant of $f$, $\mu$ the strong-convexity modulus, $\theta,\nu > 0$ the sample-test parameters, $\eta$ the resulting fixed step size, $q$ the strong-convexity factor, $\pi_t \in (0,1)$ the momentum sequence (with $\pi_{t+1}$ the positive root of the quadratic), and $h$ the nonsmooth regularizer.

Reference: Dongxuan Zhu, Weihuan Huang, Caihua Chen, "Boosting Accelerated Proximal Gradient Method with Adaptive Sampling for Stochastic Composite Optimization", arXiv 2025. https://arxiv.org/abs/2507.18277

---
[Back to the Canon](../index.md)
