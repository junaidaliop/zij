# SGDSaI

Implements SGD-SaI, SGD with momentum and learning rate Scaling at Initialization.

SGD-SaI replaces Adam's second-order momentum with a per-block scaling
factor, the gradient signal-to-noise ratio (g-SNR), computed once from the
gradients of the first batch and then held constant for the rest of
training. For a parameter block $i$ with gradient $g$, the g-SNR
is the ratio of the gradient norm to its standard deviation:


$$
G^{(i)}_{\mathrm{snr}} = \frac{\lVert g \rVert_2}{\sigma(g) + \epsilon}
$$

where $\sigma(g)$ is the standard deviation of the gradient entries.
The block is then updated with momentum and decoupled weight decay, scaling
the learning rate by the constant g-SNR:


$$
\begin{aligned}
m_t &= \mu m_{t-1} + (1 - \mu)\, g_t \\
\theta_t &= (1 - \gamma \lambda)\, \theta_{t-1}
            - \gamma\, G^{(i)}_{\mathrm{snr}}\, m_t
\end{aligned}
$$

where $\gamma$ is the learning rate, $\mu$ the momentum
coefficient, and $\lambda$ the weight decay. With
`weight_decouple=False` the weight decay is instead added to the gradient
as an L2 penalty.

Reference: Minghao Xu et al., "No More Adam: Learning Rate Scaling at
Initialization is All You Need", 2024.
https://arxiv.org/abs/2412.11768

---
[Back to the Canon](../index.md)
