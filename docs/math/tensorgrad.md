# TensorGRaD

Implements TensorGRaD, a memory-efficient optimizer that splits each gradient tensor into a low-rank part and a sparse part and keeps Adam state for each in compressed form.

TensorGRaD targets the tensor-parameterized layers of neural operators, where the optimizer state dominates memory. It applies robust tensor decomposition to the gradient: a Tucker low-rank component models the smooth, global structure while an unstructured top-$k$ sparse component preserves the sharp, localized outliers a pure low-rank fit would discard. The two components are maintained independently, each with its own Adam moments living in the compressed space (the low-rank core for $\hat{\mathcal{G}}_L$, the index–value set for $\hat{\mathcal{G}}_S$), so the state is far smaller than a full Adam buffer. The sparse support $\Omega$ and the Tucker factor matrices $U^{(n)}$ are recomputed only every $T$ steps and reused in between.

$$
\begin{aligned}
\hat{\mathcal{G}}_S &= \mathrm{Sparse}(\mathcal{G}_t, \Omega), \qquad |\Omega| = \lceil \rho I \rceil \\
\hat{\mathcal{G}}_L &= (\mathcal{G}_t - \hat{\mathcal{G}}_S) \times_1 U^{(1)\top} \cdots \times_N U^{(N)\top} \\
\mathcal{N}_L &= \mathrm{AdamUpdate}(\hat{\mathcal{G}}_L, \mathcal{M}_L, \mathcal{V}_L, \beta_1, \beta_2, t) \\
\mathcal{N}_S &= \mathrm{AdamUpdate}(\hat{\mathcal{G}}_S, \mathcal{M}_S, \mathcal{V}_S, \beta_1, \beta_2, t) \\
\tilde{\mathcal{G}}_t &= \alpha \,\big(\mathcal{N}_L \times_1 U^{(1)} \cdots \times_N U^{(N)}\big) + \lambda \, \mathcal{N}_S \\
\mathcal{W}_{t+1} &= \mathcal{W}_t - \eta \, \tilde{\mathcal{G}}_t
\end{aligned}
$$

where $\mathcal{W}$ are the parameters, $\mathcal{G}_t$ the gradient tensor, $\eta$ the learning rate, $\hat{\mathcal{G}}_S$ the sparse part formed by keeping the $\lceil \rho I \rceil$ largest-magnitude entries on support $\Omega$ ($\rho$ the density, $I$ the tensor size), $\hat{\mathcal{G}}_L$ the Tucker-projected low-rank part of the residual through factor matrices $U^{(n)}$, $\times_n$ the mode-$n$ product, $\mathrm{AdamUpdate}$ the standard Adam step (moments $\mathcal{M},\mathcal{V}$ with decays $\beta_1,\beta_2$, bias correction, and $\mathcal{M}/(\sqrt{\mathcal{V}}+\epsilon)$ normalization) applied independently in compressed space, $\mathcal{N}_L,\mathcal{N}_S$ the resulting normalized updates, and $\alpha,\lambda$ the scaling factors for the low-rank and sparse contributions.

Reference: Sebastian Loeschcke, David Pitt, Robert Joseph George, Jiawei Zhao, Cheng Luo, Yuandong Tian, Jean Kossaifi, Anima Anandkumar, "TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training", arXiv 2025. https://arxiv.org/abs/2501.02379

---
[Back to the Canon](../index.md)
