# ElasticZO

Implements ElasticZO, a hybrid on-device trainer that updates early layers with zeroth-order estimates and later layers with backpropagation.

The network of $L$ layers is split at a cutoff $C$. The first $C$ layers are trained without storing activations or backward gradients: a single shared random direction $z\sim\mathcal{N}(0,I)$ is used to perturb the parameters in both directions, and the resulting loss difference yields a scalar projected gradient $g$ that, multiplied by each layer's slice of $z$, gives a memory-free SPSA gradient estimate. The remaining $L-C$ layers are trained normally with a first-order optimizer (e.g. SGD). Increasing $C$ trades accuracy for a smaller memory footprint.

$$
\begin{aligned}
g &= \frac{\mathcal{L}(\theta+\epsilon z;\mathcal{B}) - \mathcal{L}(\theta-\epsilon z;\mathcal{B})}{2\epsilon}, \\
\theta_l &\leftarrow \theta_l - \eta\, g\, z_l, \quad &&l \le C, \\
\theta_l &\leftarrow \theta_l - \eta\, g_t, \quad &&l > C.
\end{aligned}
$$

where $\theta_l$ are the parameters of layer $l$, $z_l$ is the slice of the shared perturbation $z\sim\mathcal{N}(0,I)$ for that layer, $\epsilon$ is the perturbation scale, $g$ is the projected (scalar) zeroth-order gradient over minibatch $\mathcal{B}$, $\eta$ is the learning rate, and $g_t$ is the backpropagated gradient for the first-order layers.

Reference: Keisuke Sugiura, Hiroki Matsutani, "ElasticZO: A Memory-Efficient On-Device Learning with Combined Zeroth- and First-Order Optimization", arXiv 2025. https://arxiv.org/abs/2501.04287

---
[Back to the Canon](../index.md)
