# FO-STDGD

Implements FO-STDGD, fractional-order spike-timing-dependent gradient descent for spiking neural networks.

FO-STDGD replaces the first-order derivative in backpropagation with the $\alpha$-th order Caputo fractional derivative, taken with respect to the synaptic weight about its previous value. Because the Caputo derivative is nonlocal, the update mixes the current gradient with the size of the last weight step, and the fractional order $\alpha \in (0, 2)$ tunes how strongly that memory is weighted; $\alpha = 1$ recovers ordinary gradient descent.

In practice the Caputo derivative is expanded as a power series and only the leading term is retained, which turns the abstract operator into a closed-form weight update. Applied to the spike-timing-dependent loss gradient $\partial L / \partial w_{ij}$ of a multi-layer SNN, the rule is

$$
\begin{aligned}
w_{ij,(k+1)} &= w_{ij,(k)} - \mu \, \frac{\partial^{\alpha} L}{\partial w_{ij}^{\,\alpha}}, \\
\frac{\partial^{\alpha} L}{\partial w_{ij}^{\,\alpha}} &= \frac{1}{\Gamma(2-\alpha)} \, \frac{\partial L}{\partial w_{ij,(k-1)}} \, \left| w_{ij,(k)} - w_{ij,(k-1)} + \epsilon \right|^{1-\alpha}.
\end{aligned}
$$

where $w_{ij}$ is the weight from neuron $j$ to neuron $i$, $k$ the iteration index, $\mu$ the learning rate, $\alpha \in (0,2)$ the fractional order, $\Gamma$ the gamma function, $\partial L / \partial w_{ij,(k-1)}$ the loss gradient evaluated at the previous weight, and $\epsilon$ a small constant guarding against the singularity when consecutive weights coincide.

Reference: Yi Yang, Richard M. Voyles, Haiyan H. Zhang, Robert A. Nawrocki, "Fractional-order spike-timing-dependent gradient descent for multi-layer spiking neural networks", Neurocomputing 2025. https://doi.org/10.1016/j.neucom.2024.128662

---
[Back to the Canon](../README.md)
