# DP-KFC

Implements DP-KFC, differentially private training with a data-free Kronecker-factored preconditioner.

DP-SGD injects isotropic Gaussian noise into an anisotropic loss landscape, wasting privacy budget on poorly-conditioned directions. DP-KFC corrects the geometry with a K-FAC preconditioner whose factors are built by probing the network with synthetic noise rather than data, so it consumes no privacy budget and incurs no distribution shift. Each layer's preconditioner is refreshed every $T_\mathrm{freq}$ steps from activation and pre-activation-gradient covariances $\hat{A}_{l-1}$ and $\hat{G}_l$, decomposed into normalized inverse-square-root factors $U_{A,l}$ and $U_{G,l}$.

At each step, per-sample gradients $g_l^{(i)}$ are preconditioned layer-wise, clipped to a global Frobenius norm $C$, summed, perturbed with Gaussian noise calibrated to the clipping bound, and used to update the parameters.

$$
\begin{aligned}
U_{A,l} &= Q_A(\Lambda_A + \gamma I)^{-1/2} Q_A^\top, \quad U_{G,l} = Q_G(\Lambda_G + \gamma I)^{-1/2} Q_G^\top \\
\tilde{g}_l^{(i)} &= U_{G,l}\, g_l^{(i)}\, U_{A,l} \\
\nu_i &= \sqrt{\textstyle\sum_l \lVert \tilde{g}_l^{(i)} \rVert_F^2}, \quad \bar{g}^{(i)} = \tilde{g}^{(i)} / \max\!\left(1, \tfrac{\nu_i}{C}\right) \\
\tilde{G} &= \frac{1}{|\mathcal{B}|}\left(\sum_{i \in \mathcal{B}} \bar{g}^{(i)} + \mathcal{N}(0, \sigma^2 C^2 I)\right) \\
\theta_{t+1} &= \theta_t - \eta\, \tilde{G}
\end{aligned}
$$

where $g_l^{(i)}$ is the per-sample gradient of layer $l$, $(Q_A, \Lambda_A)$ and $(Q_G, \Lambda_G)$ are eigendecompositions of the synthetic-probe covariances $\hat{A}_{l-1}$ and $\hat{G}_l$, $\gamma$ is an eigenvalue stability constant, $C$ is the clipping threshold, $\sigma$ the noise multiplier, $\eta$ the learning rate, and $\mathcal{B}$ the private minibatch.

Reference: Marc Molina Van den Bosch, Riccardo Taiello, Albert Sund Aillet, Andrea Protani, Miguel Angel Gonzalez Ballester, Luigi Serio, "DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning", ICML 2026. https://arxiv.org/abs/2605.13418

---
[Back to the Canon](../README.md)
