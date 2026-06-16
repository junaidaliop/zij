# SpecMuon

Implements SpecMuon, a spectrally guided Muon variant that regulates per-mode step sizes through a relaxed scalar auxiliary variable (RSAV).

Muon orthogonalizes the gradient in its singular-vector basis and takes unit-singular-value steps, which can be too aggressive for the ill-conditioned, multi-scale gradients of physics-informed learning. SpecMuon decomposes the matrix gradient $G_t$ by SVD and, along the top-$k$ dominant singular directions, replaces the unit weight with a per-mode auxiliary scalar $r_{t,j}$ that is shrunk and relaxed according to the global loss energy $\sqrt{\mathcal{L}_t}$. The remaining modes keep the standard Muon orthogonalized contribution. The resulting search direction $O_t$ is then fed through Nesterov-style momentum.

$$
\begin{aligned}
\hat{G}_t &= G_t / (\lVert G_t \rVert_F + \epsilon), \qquad U \Sigma V^\top = \mathrm{SVD}(\hat{G}_t) \\
r_j^{\mathrm{new}} &= r_{t-1,j} \big/ \Big( 1 + \tfrac{1}{2}\,\tfrac{\gamma}{s_j + \epsilon}\, \lVert d_j \rVert_F \Big), \quad d_j = \tfrac{s_j\, u_j v_j^\top}{\sqrt{\mathcal{L}_t} + \epsilon} \\
\chi_j &= \frac{\sqrt{\mathcal{L}_t} - \sqrt{T_j}}{\sqrt{\mathcal{L}_t} - r_j^{\mathrm{new}} + \epsilon}, \quad T_j = (1-\xi)(r_j^{\mathrm{new}})^2 + \xi\, r_{t-1,j}^2 + (1-\xi)(r_j^{\mathrm{new}} - r_{t-1,j})^2 \\
r_{t,j} &= \mathrm{clamp}(\chi_j, 0, 1)\, r_j^{\mathrm{new}} + \big(1 - \mathrm{clamp}(\chi_j, 0, 1)\big)\sqrt{\mathcal{L}_t} \\
O_t &= \sum_{j=1}^{k} \frac{r_j^{\mathrm{new}}}{\sqrt{\mathcal{L}_t} + \epsilon}\, u_j v_j^\top \;+\; U_{>k}\, \mathrm{diag}(\Sigma_{>k})\, V_{>k}^\top \\
B_t &= \mu B_{t-1} + O_t, \qquad \theta_t = \theta_{t-1} - \gamma B_t
\end{aligned}
$$

where $G_t$ is the matrix-valued gradient, $\hat{G}_t$ its Frobenius-normalized form, $u_j, s_j, v_j$ the $j$-th singular triple, $r_{t,j}$ the per-mode auxiliary variable initialized at $\sqrt{\mathcal{L}_0}$, $\mathcal{L}_t$ the loss, $\gamma$ the learning rate, $\mu$ the momentum, $\xi \in [0,1]$ the SAV smoothing factor, $k$ the number of guided modes, and $\epsilon$ a stability constant; $U_{>k}, \Sigma_{>k}, V_{>k}$ are the remaining (non-guided) singular components.

Reference: Binghang Lu, Jiahao Zhang, Guang Lin, "Muon with Spectral Guidance: Efficient Optimization for Scientific Machine Learning", arXiv 2025. https://arxiv.org/abs/2602.16167

---
[Back to the Canon](../index.md)
