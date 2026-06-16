# tpSGD

Implements tpSGD (Target Projection SGD), a backprop-free, layer-local training rule that learns from forward passes alone.

Each layer is trained independently against a local target obtained by projecting the one-hot label $y^*$ through a fixed random matrix $P_i$, removing the need for a backward pass and the storage of intermediate activations across layers. The layer minimizes the discrepancy between its output $y_i$ and the projected target $P_i y^*$; with an $\ell_2$ layer loss this yields a delta-rule update gated by the local activation derivative, and an $\ell_1$ loss replaces the residual with its sign for a robust, low-precision-friendly variant.

$$
\begin{aligned}
J_i &= \lVert P_i y^* - y_i \rVert_2^2, \\
w_i^{(t+1)} &= w_i^{(t)} + \eta\,(P_i y^* - y_i) \odot \sigma_i'(z_i) \quad (\ell_2), \\
w_i^{(t+1)} &= w_i^{(t)} + \eta\,\mathrm{sign}(P_i y^* - y_i) \odot \sigma_i'(z_i) \quad (\ell_1),
\end{aligned}
$$

where $w_i$ are layer $i$ weights, $\eta$ is the learning rate, $P_i$ is the fixed random projection matrix for layer $i$, $y^*$ is the one-hot label, $y_i = \sigma_i(z_i)$ is the layer output with pre-activation $z_i$ and activation $\sigma_i$, $\sigma_i'$ its derivative, and $\odot$ is element-wise multiplication.

Reference: Michael Lomnitz, Zachary Daniels, David Zhang, Michael Piacentino, "Learning with Local Gradients at the Edge", arXiv 2022. https://arxiv.org/abs/2208.08503

---
[Back to the Canon](../index.md)
