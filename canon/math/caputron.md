# Caputron

Implements Caputron, a Caputo fractional-order optimizer for Tempotron-like spiking neural networks.

Caputron replaces the first-order Tempotron gradient with a Caputo fractional derivative of order $\alpha \in (0,1)$. The standard Tempotron update scales each weight by the kernel contribution $K$ and the loss sign; Caputron multiplies this gradient by a power-law factor $(\theta - c)^{1-\alpha}$ with a Gamma-function normalization, where $c$ is the per-neuron minimum weight that serves as the lower terminal of the fractional integral. As $\alpha \to 1$ the factor reduces to the ordinary derivative, while smaller orders inject a memory-dependent, weight-magnitude-aware rescaling that the authors interpret as an adaptive normalization.

$$
\begin{aligned}
c_i &= \min_j \theta_{ij}, \\
\theta_{t+1} &= \theta_t - \eta \, s_t \odot K \odot \frac{(\theta_t - c)^{1-\alpha}}{\Gamma(\alpha)\,(1-\alpha)},
\end{aligned}
$$

where $\theta$ are the synaptic weights, $\eta$ the learning rate, $\alpha \in (0,1)$ the Caputo derivative order, $K$ the postsynaptic kernel contribution, $s_t$ the Tempotron loss sign ($+1$ for a missing required spike, $-1$ for a spurious spike, $0$ otherwise), $c_i$ the minimum weight over the inputs of neuron $i$, $\Gamma$ the Gamma function, and $\odot$ elementwise multiplication.

Reference: Natabara Máté Gyöngyössy, Gábor Erős, János Botzheim, "Exploring the Effects of Caputo Fractional Derivative in Spiking Neural Network Training", Electronics 2022. https://doi.org/10.3390/electronics11142114

---
[Back to the Canon](../README.md)
