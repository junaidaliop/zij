# SLTrain

Implements SLTrain, a sparse-plus-low-rank weight reparameterization for parameter- and memory-efficient pretraining.

SLTrain replaces each dense weight matrix with the sum of a low-rank factorization $BA$ and a sparse matrix $S$. The sparse support is drawn once by uniform random sampling and held fixed throughout training, so only the non-zero values are learned alongside the low-rank factors. The three components $B$, $A$, and the sparse values are optimized jointly with Adam; storing only the factors and the sparse entries cuts both parameter count and optimizer-state memory relative to full-rank training.

$$
\begin{aligned}
W &= \frac{\alpha}{r}\, B A + S, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times p},\ \mathrm{nnz}(S) = \delta\, d p \\
A_0 &\sim \mathrm{Kaiming}, \quad B_0 = 0, \quad \mathcal{V}_0 \sim \mathcal{U}\!\left[-\tfrac{1}{\sqrt{d_{\mathrm{in}}}},\ \tfrac{1}{\sqrt{d_{\mathrm{in}}}}\right], \quad \mathcal{I}\ \text{fixed, uniform random} \\
\{B, A, \mathcal{V}\} &\leftarrow \mathrm{Adam}\big(\nabla_{\{B, A, \mathcal{V}\}} \mathcal{L}\big)
\end{aligned}
$$

where $W$ is the effective weight, $B,A$ are the rank-$r$ low-rank factors scaled by $\alpha/r$ ($\alpha$ a balancing hyperparameter), $S$ is the sparse matrix with index set $\mathcal{I}$ and learnable values $\mathcal{V}$, $\delta$ is the sparsity density, and the trainable parameters $\{B, A, \mathcal{V}\}$ are updated by Adam.

Reference: Andi Han, Jiaxiang Li, Wei Huang, Mingyi Hong, Akiko Takeda, Pratik Jawanpuria, Bamdev Mishra, "SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining", NeurIPS 2024. https://arxiv.org/abs/2406.02214

---
[Back to the Canon](../index.md)
