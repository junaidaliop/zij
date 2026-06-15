# PolarGrad

Implements PolarGrad, a polar-decomposition preconditioned optimizer.

For a matrix parameter, PolarGrad orthogonalizes the gradient (or its
momentum average) through the polar decomposition and rescales the
orthogonal factor by the nuclear norm of the same matrix. Writing
$U_t H_t = \mathrm{polar}(M_t)$ for the polar decomposition, the
nuclear norm equals $\mathrm{tr}(H_t) = \langle M_t, U_t
\rangle_F$, which avoids a full singular value decomposition. With
exponential-moving-average momentum and decoupled weight decay, the
momentum-first update (the default) is


$$
\begin{aligned}
M_t &= \beta M_{t-1} + (1 - \beta) G_t \\
U_t H_t &= \mathrm{polar}(M_t) \\
\theta_t &= (1 - \lambda \gamma)\, \theta_{t-1}
    - \gamma\, \mathrm{tr}(H_t)\, U_t
\end{aligned}
$$

where $\gamma$ is the learning rate, $\beta$ is `momentum`,
and $\lambda$ is `weight_decay`. Setting `polar_first=True`
selects the polar-first variant, which decomposes the gradient before the
momentum average,


$$
U_t H_t = \mathrm{polar}(G_t), \quad
M_t = \beta M_{t-1} + (1 - \beta) U_t, \quad
\theta_t = (1 - \lambda \gamma)\, \theta_{t-1}
    - \gamma\, \mathrm{tr}(H_t)\, M_t.
$$

The nuclear-norm scaling subsumes Muon, whose orthogonalized update PolarGrad
recovers when the scaling is dropped. The orthogonal polar factor is computed
by the backend named in `method` (`'qdwh'`, `'ns'`, or
`'polar_express'`).


**Note:** only matrix (two-dimensional) parameters are supported; pair PolarGrad

with another optimizer for embeddings, biases, and scalar parameters.

Reference: Tim Tsz-Kit Lau, Qi Long, Weijie Su, "PolarGrad: A Class of
Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective",
2025. https://arxiv.org/abs/2505.21799

---
[Back to the Canon](../README.md)
