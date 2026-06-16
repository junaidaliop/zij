# Caputo BP-NN FOGD (ISNN)

Implements Caputo BP-NN FOGD, a fractional-order gradient descent for backpropagation neural networks built on the Caputo fractional derivative.

Standard backpropagation moves each weight along the integer-order gradient $\partial E / \partial \theta$ of the quadratic error $E$. Here the integer derivative is replaced by a Caputo fractional derivative of order $\mu \in (0,1)$, so each step depends on the weight's value and not only on the local slope. For the quadratic energy and the usual chain rule, the Caputo derivative of $E$ with respect to a weight $\theta$ admits the closed form below, in which the integer gradient is rescaled by a power of the weight and a Gamma-function factor; this injects a memory/non-locality effect that smooths the trajectory and damps oscillations. The order $\mu$ interpolates between fractional ($\mu \to 0$) and ordinary gradient descent ($\mu \to 1$).

$$
\begin{aligned}
D^{\mu}_{\theta} E &= \frac{\partial E}{\partial \theta}\cdot\frac{\theta^{\,1-\mu}}{\Gamma(2-\mu)}, \\
\theta_{t+1} &= \theta_t - \eta\, D^{\mu}_{\theta} E .
\end{aligned}
$$

where $\theta$ is a network weight, $\eta$ the learning rate, $E$ the quadratic error, $\mu \in (0,1)$ the fractional order, $\partial E/\partial \theta$ the ordinary backpropagated gradient, $\Gamma(\cdot)$ the Gamma function, and $D^{\mu}_{\theta}E$ the Caputo fractional-order gradient.

Reference: Guoling Yang, Bingjie Zhang, Zhaoyang Sang, Jian Wang, Hua Chen, "A Caputo-Type Fractional-Order Gradient Descent Learning of BP Neural Networks", International Symposium on Neural Networks (ISNN) 2017. https://doi.org/10.1007/978-3-319-59072-1_64

---
[Back to the Canon](../index.md)
