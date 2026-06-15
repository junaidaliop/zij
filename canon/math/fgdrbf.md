# FGD-RBF

Implements FGD-RBF, a fractional gradient descent learning rule for radial basis function neural networks.

The method trains an RBF network with a fractional gradient descent (FGD) step formed as a convex combination of the conventional integer-order gradient and a modified Riemann-Liouville derivative-based fractional gradient. The fractional term replaces the ordinary gradient $g_t$ by a fractional power of the loss derivative: following the modified Riemann-Liouville construction the reference point is reset to the current iterate at every step (a variable initial value), which removes the lower-terminal memory of the classical Riemann-Liouville derivative and lets the iterates converge to the true extremum rather than a fractional-order one.

Writing $g_t = \nabla_\theta E_t$ for the gradient of the squared-error loss, the modified Riemann-Liouville fractional gradient and the convex-combination update are

$$
\begin{aligned}
g_t^{(\alpha)} &= \frac{1}{\Gamma(2-\alpha)}\, g_t\, \lvert \theta_t - \theta_{t-1} \rvert^{\,1-\alpha} \\
\theta_{t+1} &= \theta_t - \eta\,\big[\,(1-\lambda)\, g_t + \lambda\, g_t^{(\alpha)}\,\big]
\end{aligned}
$$

where $\theta$ are the RBF weights (and, applied componentwise, the centers and widths), $\eta$ is the learning rate, $g_t$ the integer-order gradient, $g_t^{(\alpha)}$ the modified Riemann-Liouville fractional gradient of order $\alpha \in (0,1)$, $\Gamma(\cdot)$ the gamma function, $\theta_{t-1}$ the previous iterate used as the variable initial value, and $\lambda \in [0,1]$ the mixing weight of the convex combination (with $\lambda=0$ recovering plain gradient descent).

Reference: Shujaat Khan, Imran Naseem, Muhammad Ammar Malik, Roberto Togneri, Mohammed Bennamoun, "A Fractional Gradient Descent-Based RBF Neural Network", Circuits, Systems, and Signal Processing 2018. https://doi.org/10.1007/s00034-018-0835-3

---
[Back to the Canon](../README.md)
