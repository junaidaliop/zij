# HTMuon

Implements HTMuon, Muon with a heavy-tailed spectral correction that raises the momentum matrix's singular values to a power $p \in (0,1)$.

Muon orthogonalizes the momentum buffer, which is equivalent to setting all of its singular values to one and assigns equal weight to every singular-vector direction. This produces light-tailed update and weight spectra, which Heavy-Tailed Self-Regularization theory associates with weaker generalization. HTMuon keeps Muon's momentum and matrix structure but replaces the orthogonalization with a power transform of the singular values: from the SVD $m_t = U_t \Sigma_t V_t^\top$ it forms $U_t \Sigma_t^{p} V_t^\top$. Smaller $p$ moves toward Muon's all-ones spectrum, while $p \to 1$ recovers SGDM; the intermediate regime makes updates more heavy-tailed while retaining the matrix-based coupling between directions.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
U_t,\ \Sigma_t,\ V_t^\top &= \mathrm{SVD}(m_t) \\
O_t &= U_t\, \Sigma_t^{\,p}\, V_t^\top \\
s &= \max\!\left(1,\ \tfrac{m}{n}\right) \\
\theta_{t+1} &= \theta_t - \eta\lambda\,\theta_t - \eta\, s\, O_t
\end{aligned}
$$

where $\theta$ are the matrix-shaped parameters of shape $m \times n$, $g_t$ is the gradient, $m_t$ the momentum buffer ($m_0 = 0$), $\beta$ the momentum coefficient, $U_t \Sigma_t V_t^\top$ the SVD of $m_t$, $p \in (0,1)$ the spectral power (default $p = 0.125$; $p = 0$ recovers Muon, $p = 1$ gives SGDM), $s$ the shape-dependent scaling factor, $\eta$ the learning rate, and $\lambda$ the decoupled weight decay.

Reference: Tianyu Pang, Yujie Fang, Zihang Liu, Shenyang Deng, Lei Hsiung, Shuhua Yu, Yaoqing Yang, "HTMuon: Improving Muon via Heavy-Tailed Spectral Correction", arXiv 2026. https://arxiv.org/abs/2603.10067

---
[Back to the Canon](../README.md)
