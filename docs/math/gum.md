# GUM

Implements GUM, an unbiased low-rank gradient projection built on Muon.

Low-rank projection methods such as GaLore compress the optimizer state by projecting gradients onto the top-$r$ left singular subspace $P_t$ of the gradient, but discarding the orthogonal complement makes the projected update a biased estimate of the full gradient. GUM removes this bias by stochastically choosing, per layer and per period, between a low-rank update on the captured subspace and a full-rank update on its complement, then importance-reweighting each branch so the expected update equals the full-gradient Muon step. Each branch orthogonalizes its accumulated momentum with the Newton-Schulz iteration, inheriting Muon's matrix-sign update while keeping low-rank memory cost on the dominant branch.

For each period $t$ a projector $P_t = U_t[:,{:}r]$ is taken from the SVD of the period's first gradient, and each layer is sampled for a full-rank update with probability $q$. With $\mathrm{NS}(\cdot)$ the Newton-Schulz orthogonalization:

$$
\begin{aligned}
P_t &= U_t[:,{:}r], \qquad U_t S_t V_t^\top = \mathrm{SVD}(G_{t,0}), \qquad q = \gamma / N_L, \\
\text{low-rank: }\quad R_{t,k} &= \beta\, R_{t,k-1} + \tfrac{1}{1-q}\, P_t^\top G_{t,k}, & W_{t,k+1} &= W_{t,k} + \eta\, P_t\, \mathrm{NS}(R_{t,k}), \\
\text{full-rank: }\quad R_{t,k} &= \beta\, R_{t,k-1} + \tfrac{1}{q}\,\big(G_{t,k} - P_t P_t^\top G_{t,k}\big), & W_{t,k+1} &= W_{t,k} + \eta\, \mathrm{NS}(R_{t,k}),
\end{aligned}
$$

where $W$ is the weight matrix, $G_{t,k}$ the gradient at iteration $k$ of period $t$, $P_t$ the rank-$r$ projector, $\beta$ the momentum coefficient, $\eta$ the learning rate, $q=\gamma/N_L$ the per-layer full-rank sampling probability ($\gamma$ full-rank layers out of $N_L$ total), and the factors $\tfrac{1}{1-q}$ and $\tfrac{1}{q}$ importance-reweight the two branches so the expected update is unbiased.

Reference: Rui Pan, Yang Luo, Yuxing Liu, Yang You, Tong Zhang, "Unbiased Gradient Low-Rank Projection", arXiv 2025. https://arxiv.org/abs/2510.17802

---
[Back to the Canon](../index.md)
