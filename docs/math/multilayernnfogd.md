# Multi-layer NN FOGD

Implements the Multi-layer NN FOGD, a multi-layer feedforward network trained by fractional-order gradient descent with the Caputo derivative.

The method replaces the integer-order gradient in backpropagation with a Caputo fractional derivative of order $\nu$. The loss $\mathcal{L}$ is the traditional quadratic energy (error) function augmented with an $L_{1/2}$ smooth regularization term, and each weight is moved against the fractional-order gradient of that loss. Exploiting the power-law memory of the Caputo derivative, the update carries information from past states, and the paper proves monotonicity of the error and weak/strong convergence of the resulting multi-layer algorithm.

For the weight matrix $W$ of a layer, the per-iteration update is:

$$
\begin{aligned}
\Delta W &= W^{+} - W = -\eta\, \mathcal{D}^{\nu}_{W}\,\mathcal{L} \\
W^{+} &= W - \eta\, \mathcal{D}^{\nu}_{W}\,\mathcal{L}
\end{aligned}
$$

where $W$ are the layer weights, $W^{+}$ the updated weights, $\eta > 0$ the learning rate, $\nu$ the fractional order, $\mathcal{L}$ the quadratic energy error with $L_{1/2}$ smooth regularization, and $\mathcal{D}^{\nu}_{W}$ the Caputo fractional derivative with respect to $W$, defined as $\,_a^{C}\mathcal{D}_x^{\nu} f(x) = \frac{1}{\Gamma(n-\nu)} \int_a^x (x-y)^{n-\nu-1} f^{(n)}(y)\, dy$ with $n = \lceil \nu \rceil$ and $\Gamma(\cdot)$ the Gamma function; applying it to the power-form error and regularization terms yields Gamma-normalized fractional powers of the weights.

Reference: Yu Zhao et al., "Convergence Analysis and Application for Multi-Layer Neural Network Based on Fractional-Order Gradient Descent Learning", Advanced Theory and Simulations 2024. https://doi.org/10.1002/adts.202300662

---
[Back to the Canon](../index.md)
