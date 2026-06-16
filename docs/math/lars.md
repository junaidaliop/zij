# LARS

Implements LARS, layer-wise adaptive rate scaling for large-batch SGD.

Each multidimensional parameter (treated as a layer) is given a local
learning rate proportional to the ratio between the norm of its weights and
the norm of its gradient, so the magnitude of every layer's update no longer
depends on the scale of its gradient:


$$
\begin{aligned}
\lambda^l &= \eta \, \frac{\lVert \theta^l \rVert}
                           {\lVert g^l \rVert}                          \\
g^l       &\leftarrow \lambda^l \left( g^l + \beta\, \theta^l \right)   \\
v_t^l     &= \mu\, v_{t-1}^l + g^l                                      \\
\theta_t^l &= \theta_{t-1}^l - \gamma\, v_t^l
\end{aligned}
$$

where $\eta$ is the trust coefficient, $\beta$ the weight decay,
$\mu$ the momentum, and $\gamma$ the global learning rate. The
trust ratio $\lambda^l$ falls back to one when either norm is zero.
Parameters with one dimension or fewer (biases and scalars) skip the rate
scaling and weight decay and are updated by plain momentum SGD.

Reference: Yang You, Igor Gitman, Boris Ginsburg, "Large Batch Training of
Convolutional Networks", arXiv 2017.
https://arxiv.org/abs/1708.03888


**Note:** `foreach` selects a multi-tensor implementation that yields the same result as the per-parameter path; it is disabled for groups using Nesterov momentum.


---
[Back to the Canon](../index.md)
