# Orth-Dion

Implements Orth-Dion, a distributed low-rank spectral optimizer that orthogonalizes the right factor to remove Dion's geometric mismatch.

Orth-Dion follows Dion's communication-efficient recipe: each parameter matrix keeps an error-feedback buffer that combines the gradient with the residual carried over from the previous step, and one warm-started power-iteration step extracts a rank-$r$ approximation $U_t \bar V_t^\top$ of the buffer. Dion column-normalizes the right factor, which fails to reproduce the rank-$r$ polar factor that Muon implicitly targets and costs an extra $\sqrt{r}$ in the rate. Orth-Dion changes a single line: it QR-orthonormalizes the right factor $W_t$ instead of column-normalizing it, forcing the dual-norm factor to $1$ and recovering the exact spectral convergence rate at Dion's per-step communication cost. The part of the buffer not captured by the rank-$r$ direction is fed back through the residual, scaled by $\beta$.

$$
\begin{aligned}
M_t &= G_t + R_t \\
U_t &= \mathrm{orth}(M_t V_{t-1}) \\
W_t &= M_t^\top U_t \\
\bar V_t &= \mathrm{orth}(W_t) \\
X_{t+1} &= X_t - \eta\, U_t \bar V_t^\top \\
R_{t+1} &= \beta\big(M_t - U_t(U_t^\top M_t)\big) \\
V_t &= \bar V_t
\end{aligned}
$$

where $X_t$ is the parameter matrix, $G_t$ its gradient, $R_t$ the error-feedback residual, $M_t$ the buffer, $U_t \in \mathbb{R}^{m\times r}$ and $\bar V_t \in \mathbb{R}^{n\times r}$ the orthonormal low-rank factors from one power-iteration step, $V_t$ the warm start carried to the next step, $r$ the rank, $\eta$ the learning rate, $\beta$ the error-feedback coefficient, and $\mathrm{orth}(\cdot)$ the QR orthonormalization of the column space. Replacing $\mathrm{orth}(W_t)$ with column normalization recovers Dion.

Reference: Tatsuhiro Nakamori, Laura Gomezjurado Gonzalez, Ganesh Talluri, Ansh Tiwari, Hideyuki Kawashima, Ioannis Mitliagkas, Guillaume Rabusseau, Hiroki Naganuma, "Orth-Dion: Eliminating Geometric Mismatch in Distributed Low-Rank Spectral Optimization", arXiv 2026. https://arxiv.org/abs/2605.16341

---
[Back to the Canon](../index.md)
