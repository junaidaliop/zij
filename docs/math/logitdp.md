# Logit-DP

Implements Logit-DP, DP-SGD for similarity-based, non-decomposable objectives via clipping pairwise similarity gradients.

Standard DP-SGD clips per-example gradients, but contrastive and spreadout losses are non-decomposable: the loss for example $i$ depends on the whole batch, so its gradient does not split into independent per-example terms. Logit-DP instead clips the pairwise similarity gradients $\nabla_w S(\Phi_w(x_i), \Phi_w(x_j'))$ to norm $B$ and recombines them with the loss derivatives with respect to the logits $Z_{ij}$. This bounds the $L_2$ sensitivity of the full-batch gradient by a closed form, after which a single Gaussian noise draw is added before the descent step.

$$
\begin{aligned}
g_{ij} &= \nabla_{w_t} S(\Phi_{w_t}(x_i), \Phi_{w_t}(x_j')) \\
\bar{g}_{ij} &= \min\!\left\{ \frac{B}{\lVert g_{ij} \rVert}, 1 \right\} g_{ij} \\
\bar{g} &= \sum_{i=1}^{n} \sum_{j=1}^{n} \frac{\partial \ell^{(i,n)}}{\partial Z_{ij}} \, \bar{g}_{ij} \\
\tilde{g} &= \bar{g} + Y, \qquad Y \sim \mathcal{N}(0, \sigma C I_p) \\
C &= (G_1 + G_2 + nL)\,B \\
w_{t+1} &= w_t - \eta \, \tilde{g}
\end{aligned}
$$

where $S$ is the similarity function on embeddings produced by the model $\Phi_w$, $g_{ij}$ is the gradient of the pairwise similarity, $B$ is the clipping norm for similarity gradients, $Z_{ij}$ are the logits and $\ell^{(i,n)}$ the per-row loss, $G_1, G_2, L$ are Lipschitz/derivative bounds on the loss giving the sensitivity $C$, $\sigma$ is the noise multiplier, $I_p$ the $p \times p$ identity, $\eta$ the learning rate, and $w_t$ the model parameters at step $t$.

Reference: William Kong, Andrés Muñoz Medina, Mónica Ribero, "DP-SGD for non-decomposable objective functions", arXiv 2023. https://arxiv.org/abs/2310.03104

---
[Back to the Canon](../index.md)
