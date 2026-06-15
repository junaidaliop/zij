# MAC

Implements MAC, a Kronecker-factored second-order method that approximates the curvature with the mean activation of each layer.

MAC builds on K-FAC, which approximates the Fisher information of layer $l$ as a Kronecker product $F^{(l)} \approx A^{(l)} \otimes P^{(l)}$, where $A^{(l)} = \mathbb{E}[a\,a^\top]$ is the uncentered second moment of the layer inputs and $P^{(l)}$ is the corresponding factor from the pre-activation gradients. Forming and inverting $A^{(l)}$ is expensive, so MAC replaces it with a damped rank-one factor built only from the mean activation $\bar a = \mathbb{E}[a]$, and approximates the output-side factor by the identity.

Because the input factor is rank-one plus a multiple of the identity, the Sherman-Morrison identity gives its inverse in closed form, so the natural-gradient step reduces to a cheap right-multiplication of the reshaped gradient matrix $g_t$:

$$
\begin{aligned}
F^{(l)} &\approx \left(\bar a\,\bar a^\top + \rho I\right) \otimes I, \\
\theta_{t+1} &= \theta_t - \eta\, g_t \left( I - \frac{\bar a\,\bar a^\top}{\rho + \lVert \bar a \rVert^2} \right).
\end{aligned}
$$

where $\theta$ are the layer weights (reshaped as a matrix), $\eta$ is the learning rate, $g_t$ is the gradient matrix at step $t$, $\bar a = \mathbb{E}[a]$ is the mean of the layer's input activations, $\rho$ is an adaptive damping coefficient, and $I$ is the identity of the appropriate dimension.

Reference: Hyunseok Seung, Jaewoo Lee, Hyunsuk Ko, "MAC: An Efficient Gradient Preconditioning using Mean Activation Approximated Curvature", arXiv 2025. https://arxiv.org/abs/2506.08464

---
[Back to the Canon](../README.md)
