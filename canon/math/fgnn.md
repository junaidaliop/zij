# FGNN

Implements FGNN, a regularized graph neural network trained by approximate fractional-order gradient descent.

Classic gradient descent follows the steepest first-order direction and can settle into local optima. FGNN replaces the integer-order weight gradient with a Grünwald-Letnikov fractional-order gradient $D^{\nu}$ of order $\nu$, whose extrema differ from the first-order ones, helping the optimizer escape integer-order local minima. To avoid the high cost of the exact fractional chain rule (Di Bruno's formula), the method uses an approximate fractional chain rule, so the fractional gradient is the ordinary backpropagation gradient scaled by a power of the weight and a gamma-function factor. An $L_2$ term with coefficient $\lambda$ regularizes the weights, contributing a second fractional term.

For the weights $W_m$ of layer $m$, with $\delta_{m+1}$ the backpropagated error and $\tilde A$ the normalized adjacency matrix, the update is

$$
\begin{aligned}
D^{\nu}_{W_m} L_{L2} &\approx \delta_{m+1}\,\tilde A\,(H^m)^{\top}\,\frac{(W_m)^{1-\nu}}{\Gamma(2-\nu)} + \lambda\,\frac{(W_m)^{2-\nu}}{\Gamma(3-\nu)}, \\
W_m &\leftarrow W_m - \eta\, D^{\nu}_{W_m} L_{L2}.
\end{aligned}
$$

where $\eta>0$ is the learning rate, $\nu$ is the fractional order, $\Gamma(\cdot)$ is the gamma function, $H^m$ is the layer-$m$ activation, $\delta_{m+1}=\partial L/\partial Z^{m+1}$ is the layer error, $\tilde A = D^{-1/2}(I+A)D^{-1/2}$ is the symmetrically normalized adjacency matrix, and $\lambda$ is the $L_2$ regularization coefficient; setting $\nu=1$ recovers ordinary gradient descent.

Reference: Zijian Liu, Yaning Wang, Yang Luo, Chunbo Luo, "A Regularized Graph Neural Network Based on Approximate Fractional Order Gradients", Mathematics 2022. https://doi.org/10.3390/math10081320

---
[Back to the Canon](../README.md)
