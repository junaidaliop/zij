# NuMuon

Implements NuMuon, a nuclear-norm-constrained variant of Muon that drives weights toward low-rank, compressible structure.

Muon orthogonalizes the momentum to take a full-rank step, yet its trained weights still exhibit strong low-rank structure. NuMuon makes this explicit: instead of orthogonalizing the full matrix, it extracts only the top-$k$ singular vector pairs of the momentum and steps along their sum of rank-one outer products. Varying $k$ interpolates between a rank-one nuclear-norm update ($k=1$) and Muon's full-rank orthogonalized update; a rank scheduler anneals $k$ over training to control compressibility.

$$
\begin{aligned}
M_t &= \beta\, M_{t-1} + (1-\beta)\, G_t \\
U_{t,k},\, V_{t,k} &\leftarrow \text{top-}k \text{ left/right singular vectors of } M_t \\
k_t &= \lceil r(t)\,\min(d_\mathrm{in}, d_\mathrm{out}) \rceil \\
\theta_t &= \theta_{t-1} - \gamma\, U_{t,k} V_{t,k}^{\top}
\end{aligned}
$$

where $\theta$ are the matrix-shaped parameters, $G_t$ the gradient, $M_t$ the momentum buffer with decay $\beta$, $\gamma$ the learning rate, and $U_{t,k}, V_{t,k}$ the leading $k$ singular vectors of $M_t$ (computed by a randomized block Krylov method). The relative rank $r(t) \in (0,1]$ is set by a rank scheduler (fixed, piecewise, or cosine) so that $k_t$ is annealed during training, with $d_\mathrm{in}, d_\mathrm{out}$ the layer dimensions.

Reference: Hadi Mohaghegh Dolatabadi, Thalaiyasingam Ajanthan, Sameera Ramasinghe, Chamin P. Hewa Koneputugodage, Shamane Siriwardhana, Violetta Shevchenko, Karol Pajak, James Snewin, Gil Avraham, Alexander Long, "NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training", arXiv 2026. https://arxiv.org/abs/2603.03597

---
[Back to the Canon](../README.md)
