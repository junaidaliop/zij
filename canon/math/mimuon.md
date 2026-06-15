# MiMuon

Implements MiMuon, a mixed Muon optimizer that orthogonalizes the momentum only when its singular values are well separated.

MiMuon keeps Muon's momentum buffer but makes the orthogonalization cautious. At each step it inspects the singular values of the momentum: when the nonzero singular values are sufficiently distinct (their minimum pairwise gap exceeds a threshold $\tau$), it takes the matrix-normalized direction $U_t V_t^\top$; otherwise the spectral factorization is ill-conditioned, so it falls back to a plain momentum step. Both branches share decoupled weight decay and collapse into one expression through the map $\mathcal{G}(M_t,\alpha)=U_t\,\Sigma_t^{\alpha}\,V_t^\top$, with $\alpha=0$ for the orthogonalized branch and $\alpha=1$ for the standard branch. In practice the SVD is approximated by Newton-Schulz iteration.

$$
\begin{aligned}
M_t &= \beta\, g_t + (1-\beta)\,M_{t-1}, \qquad (U_t,\Sigma_t,V_t)=\mathrm{SVD}(M_t),\\
S_t &= \{\, i : \Sigma_{t,ii}\neq 0 \,\},\\
\theta_t &=
\begin{cases}
(1-\eta\lambda)\,\theta_{t-1} - \eta\, U_t V_t^\top, & \min_{i\neq j\in S_t}\lvert \Sigma_{t,ii}-\Sigma_{t,jj}\rvert \ge \tau,\\
(1-\eta\lambda)\,\theta_{t-1} - \eta\, M_t, & \text{otherwise},
\end{cases}\\
&= (1-\eta\lambda)\,\theta_{t-1} - \eta\,\mathcal{G}(M_t,\alpha), \qquad \mathcal{G}(M_t,\alpha)=U_t\,\Sigma_t^{\alpha}\,V_t^\top.
\end{aligned}
$$

where $\theta$ are the matrix parameters, $g_t=\nabla f(\theta_{t-1};\xi_t)$ the stochastic gradient, $M_t$ the momentum, $\beta$ the momentum coefficient, $\eta$ the learning rate, $\lambda$ the decoupled weight decay, $U_t\Sigma_t V_t^\top$ the SVD of $M_t$, $\tau$ the singular-value gap threshold, and $\alpha\in\{0,1\}$ selects the orthogonalized ($\alpha=0$) or standard ($\alpha=1$) branch.

Reference: Feihu Huang, Yuning Luo, Songcan Chen, "MiMuon: Mixed Muon Optimizer with Improved Generalization for Large Models", arXiv 2025. https://arxiv.org/abs/2605.19619

---
[Back to the Canon](../README.md)
