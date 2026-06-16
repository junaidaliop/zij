# BADM

Implements BADM (Batch ADMM), a data-driven ADMM optimizer that splits each mini-batch into sub-batches and updates primal, global, and dual variables per block.

BADM recasts training as a consensus problem: the loss is partitioned over $B$ batches, each further split into $S$ sub-batches, and a local parameter $w_{bs}$ is forced to agree with a global parameter $w$ through the constraint $w = w_{bs}$. Each epoch sweeps the batches; within a batch the global parameter is aggregated from the previous sub-batch solutions and their scaled multipliers, then every sub-batch performs an inexact (single-gradient) local solve and a dual ascent step. The sub-batches inside a batch are independent, so the $S$ local updates run in parallel.

Carrying state forward across batches via $w_{0s}^{\ell+1} = w_{Bs}^{\ell},\ \pi_{0s}^{\ell+1} = \pi_{Bs}^{\ell}$, for each batch $b = 1,\dots,B$:

$$
\begin{aligned}
w_b^{\ell+1} &= \sum_{s\in S} \alpha_s\left(w_{(b-1)s}^{\ell+1} + \frac{\pi_{(b-1)s}^{\ell+1}}{\sigma}\right) \\
w_{bs}^{\ell+1} &= w_b^{\ell+1} - \frac{\nabla F_{bs}(w_b^{\ell+1}) + \pi_{(b-1)s}^{\ell+1}}{\rho + \sigma}, \quad s \in S \\
\pi_{bs}^{\ell+1} &= \pi_{(b-1)s}^{\ell+1} + \sigma\left(w_{bs}^{\ell+1} - w_b^{\ell+1}\right), \quad s \in S
\end{aligned}
$$

where $w_b$ is the global parameter, $w_{bs}$ the local parameter for sub-batch $\mathcal{N}_{bs}$, $\pi_{bs}$ its Lagrange multiplier, $\nabla F_{bs}$ the sub-batch gradient, $\alpha_s$ the sub-batch sampling weight, $\sigma$ the augmented-Lagrangian penalty, $\rho$ the proximal coefficient of the inexact local solve, and $\ell$ the epoch index. The returned parameter is $w_B^{\ell+1}$.

Reference: Ouya Wang, Shenglong Zhou, Geoffrey Ye Li, "BADM: Batch ADMM for Deep Learning", arXiv 2024. https://arxiv.org/abs/2407.01640

---
[Back to the Canon](../index.md)
