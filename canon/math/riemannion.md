# Riemannion

Implements Riemannion (RiemannLoRA), Riemannian momentum optimization of LoRA adapters on the fixed-rank manifold.

Standard LoRA parametrizes a low-rank update as $\Delta W = A B^\top$, which is ambiguous: any invertible $C$ gives the same product via $A C, B C^{-\top}$. Riemannion removes this ambiguity by treating $\Delta W$ as a point on the manifold $\mathcal{M}_r$ of fixed-rank matrices, factored as $A_L \Sigma B_R^\top$ with orthonormal $A_L, B_R$. Each step projects the Euclidean gradient onto the tangent space at the current point, blends it with a momentum buffer carried over by vector transport, and retracts the result back onto $\mathcal{M}_r$ through a truncated SVD.

$$
\begin{aligned}
\mathrm{grad}\,\mathcal{L} &= (I - A_L A_L^\top)\,\nabla_Y \mathcal{L}(Y)\,B_R B_R^\top + A_L A_L^\top\,\nabla_Y \mathcal{L}(Y), \\
(\dot A_\mathrm{prev}, \dot B_\mathrm{prev}) &= \mathcal{T}\big((\dot A_\mathrm{old}, \dot B_\mathrm{old}),\, A_L, B_R\big), \\
\dot A &= \beta\,\dot A_\mathrm{prev} + (1-\beta)(I - A_L A_L^\top)\,\dot A, \\
\dot B &= \beta\,\dot B_\mathrm{prev} + (1-\beta)\,\dot B, \\
U, \Sigma, V^\top &= \mathrm{truncSVD}\big(\,[\,\eta\,\dot A,\; A_L\,],\; [\,B_R,\; \eta\,\dot B + B\,]\,\big), \\
A_L, B &\leftarrow U,\; \Sigma V^\top.
\end{aligned}
$$

where $A_L, B_R$ are the orthonormal factors of the current point, $\nabla_Y \mathcal{L}$ is the Euclidean gradient of the loss, $\mathrm{grad}\,\mathcal{L}$ its projection onto the tangent space, $\mathcal{T}$ the vector transport of the previous momentum to the current tangent space, $\dot A, \dot B$ the tangent-space momentum components, $\beta$ the momentum coefficient, $\eta$ the learning rate, and $\mathrm{truncSVD}$ a rank-$r$ truncated SVD that retracts the candidate factorization back onto $\mathcal{M}_r$.

Reference: Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, Denis Bobkov, Vera Soboleva, Aibek Alanov, Maxim Rakhuba, "RiemannLoRA: A Unified Riemannian Framework for Ambiguity-Free LoRA Optimization", arXiv 2025. https://arxiv.org/abs/2507.12142

---
[Back to the Canon](../README.md)
