# Caputo-based SGD (L1 scheme)

Implements Caputo-based SGD (L1 scheme), a fractional gradient descent method that replaces the ordinary gradient with a Caputo fractional gradient discretized by the L1 numerical scheme.

The Caputo derivative of order $\alpha \in (0,1)$ carries long-term memory: instead of the local slope, it weights the running history of parameter increments. Applying the L1 linear-interpolation discretization to the Caputo derivative yields a fractional gradient that is a weighted sum of past parameter differences scaled by the ordinary backpropagation factor. This fractional gradient is then used in place of $g_t$ inside an otherwise standard SGD step (which may add weight decay and momentum), so that gradient information from the trajectory's history steers each update.

$$
\begin{aligned}
\delta_r &= \frac{(\Delta\theta)^{-\alpha}}{\Gamma(2-\alpha)}\left[(r+1)^{1-\alpha} - r^{1-\alpha}\right], \\
g_t^{(\alpha)} &= \frac{\partial E}{\partial \theta_t}\sum_{s=0}^{n-1} \delta_{n-s-1}\left(\theta^{\,s+1} - \theta^{\,s}\right), \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{(\alpha)}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\alpha \in (0,1)$ the fractional order, $\Delta\theta$ the L1 step size, $\Gamma(\cdot)$ the gamma function, $E$ the loss, $\frac{\partial E}{\partial \theta_t}$ the ordinary (integer-order) backpropagated gradient factor, $\delta_r$ the L1 memory weights, and $\theta^{\,s}$ the parameter history with $n$ retained past states. Optionally a weight decay $\lambda$ ($g_t^{(\alpha)} \leftarrow g_t^{(\alpha)} + \lambda\theta_t$) and momentum $\mu$ are applied to $g_t^{(\alpha)}$ before the step, exactly as in standard SGD.

Reference: Anonymous authors, "Stochastic Fractional Gradient Descent with Caputo $L_1$ Scheme for Deep Neural Networks", TMLR submission 2024. https://openreview.net/forum?id=hCGaySEW9q

---
[Back to the Canon](../README.md)
