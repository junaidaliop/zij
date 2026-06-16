# MuonBP

Implements MuonBP, a distributed Muon variant that interleaves cheap blockwise orthogonalization with periodic full orthogonalization.

Muon orthogonalizes the momentum matrix every step, which forces an all-gather of the sharded gradient on every iteration. MuonBP (Block-Periodic) instead orthogonalizes each device's local shard $M_t^{(m)}$ independently most of the time, and gathers the full momentum to orthogonalize globally only once every $P$ steps. Block and full steps use separate learning rates $\eta_{\mathrm{block}}$ and $\eta_{\mathrm{full}}$, with RMS-norm matching scaling each update by the square root of the relevant matrix dimensions. The period $P$ interpolates between fully blockwise updates ($P\to\infty$) and standard Muon ($P=1$).

$$
\begin{aligned}
M_t^{(m)} &= \mu\, M_{t-1}^{(m)} + G_t^{(m)} \\
\text{if } t \bmod P = 0:\quad
M_t &= \mathrm{gather}\big(\{M_t^{(m)}\}_m\big), \quad
U_t = \mathrm{Orth}(M_t), \quad
\theta_{t+1} = \theta_t - \eta_{\mathrm{full}}\, U_t \\
\text{else}:\quad
U_t^{(m)} &= \mathrm{Orth}\big(M_t^{(m)}\big), \quad
\theta_{t+1}^{(m)} = \theta_t^{(m)} - \eta_{\mathrm{block}}\, U_t^{(m)} \\
\mathrm{Orth}(M) &= (MM^\top)^{-\dagger/2} M = UV^\top, \quad M = U\Sigma V^\top
\end{aligned}
$$

where $\theta$ are the matrix parameters (sharded across devices $m$), $G_t^{(m)}$ is the local gradient shard, $M_t^{(m)}$ the local momentum buffer, $\mu \in [0,1)$ the momentum coefficient, $P$ the orthogonalization period, and $\mathrm{Orth}(\cdot)$ the orthogonalization (computed via Newton-Schulz iterations) equal to $UV^\top$ from the SVD; the dagger denotes the Moore-Penrose pseudoinverse.

Reference: Ahmed Khaled, Kaan Ozkara, Tao Yu, Mingyi Hong, Youngsuk Park, "MuonBP: Faster Muon via Block-Periodic Orthogonalization", arXiv 2025. https://arxiv.org/abs/2510.16981

---
[Back to the Canon](../index.md)
