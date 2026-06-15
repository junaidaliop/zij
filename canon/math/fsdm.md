# FSDM

Implements FSDM, an improved fractional-order steepest descent method for training fractional-order backpropagation neural networks.

The method trains a fractional-order backpropagation neural network (FBPNN) by reverse incremental search in the negative directions of the fractional-order partial derivatives of the square error, generalizing the classical first-order steepest descent rule. Because the exact Caputo fractional-order partial derivatives of the loss are not directly computable through the layers, the improved scheme replaces them with tractable approximate fractional-order partial derivatives $\widetilde{D}^{\nu}$, and each weight and bias is moved against its approximate fractional gradient with a single step size. The classical first-order steepest descent method is recovered as $\nu \to 1$.

$$
\begin{aligned}
w_{i,j}^{m}(k+1) &= w_{i,j}^{m}(k) - \mu\, \widetilde{D}^{\nu}_{w_{i,j}^{m}} \hat{F}(k), \\
b_{i}^{m}(k+1) &= b_{i}^{m}(k) - \mu\, \widetilde{D}^{\nu}_{b_{i}^{m}} \hat{F}(k).
\end{aligned}
$$

where $w_{i,j}^{m}$ and $b_{i}^{m}$ are the weights and biases of layer $m$, $k$ is the iteration index, $\mu > 0$ is the learning rate, $\nu$ is the fractional order, $\hat{F}$ is the square-error objective, and $\widetilde{D}^{\nu}_{\theta}\hat{F}$ denotes the approximate fractional-order (Caputo-type) partial derivative of $\hat{F}$ with respect to parameter $\theta$, replacing the integer-order gradient used in standard steepest descent.

Reference: Yi-fei Pu, Jian Wang, "Fractional-order global optimal backpropagation machine trained by an improved fractional-order steepest descent method", Frontiers of Information Technology & Electronic Engineering 2020, 21(6): 809-833. https://doi.org/10.1631/FITEE.1900593

---
[Back to the Canon](../README.md)
