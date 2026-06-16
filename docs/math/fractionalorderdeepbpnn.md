# Fractional-Order Deep BP NN

Implements the Fractional-Order Deep Backpropagation Neural Network, a deep BP network trained by fractional gradient descent with the Caputo derivative and L2 regularization.

The optimizer replaces the integer-order gradient in backpropagation with a Caputo fractional-order derivative of order $v$. The loss is the standard error $E$ augmented with an L2 penalty, $E_{L2} = E + \tfrac{\lambda}{2}\lVert W \rVert^2$, and each weight is moved against its fractional derivative. Applying the Caputo derivative through the chain rule introduces fractional powers of the weight together with Gamma-function normalizers: the error term picks up a $\Gamma(2-v)$ factor and the regularization term a $\Gamma(3-v)$ factor.

For a weight $w_{jil}$ (connecting unit $i$ in layer $l$ to unit $j$ in layer $l+1$), with local gradient signal $\delta_j^{l+1}$ and activation $a_i^l$, the per-iteration update is:

$$
\begin{aligned}
D^{v}_{w_{jil}} E &= \delta_j^{l+1}\, a_i^{l}\, \frac{w_{jil}^{\,1-v}}{\Gamma(2-v)} \\
D^{v}_{w_{jil}} E_{L2} &= D^{v}_{w_{jil}} E + \lambda\, \frac{w_{jil}^{\,2-v}}{\Gamma(3-v)} \\
w_{jil}^{\,t+1} &= w_{jil}^{\,t} - \eta\, D^{v}_{w_{jil}} E_{L2}
\end{aligned}
$$

where $\eta > 0$ is the learning rate, $v$ is the fractional order, $\lambda \ge 0$ is the L2 regularization parameter, $\Gamma(\cdot)$ is the Gamma function, $\delta_j^{l+1}$ is the backpropagated error signal, $a_i^l$ is the incoming activation, and $D^{v}_{w}$ denotes the Caputo fractional derivative $\,_a^C D_x^{v} f(x) = \tfrac{1}{\Gamma(n-v)} \int_a^x (x-y)^{n-v-1} f^{(n)}(y)\, dy$ with $n = \lceil v+1 \rceil$.

Reference: Chunhui Bao, Yifei Pu, Yi Zhang, "Fractional-Order Deep Backpropagation Neural Network", Computational Intelligence and Neuroscience 2018. https://doi.org/10.1155/2018/7361628

---
[Back to the Canon](../index.md)
