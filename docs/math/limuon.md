# LiMuon

Implements LiMuon, a light and fast Muon-style optimizer with variance-reduced momentum and low-rank state.

LiMuon orthogonalizes its momentum matrix before each step, like Muon, but obtains the orthogonal factor directly from the SVD of $M_t$ and replaces the heavy-ball momentum with a STORM-style variance-reduced estimator. To cut memory it keeps $M_t$ in compressed form: a randomized low-rank SVD $\hat{M}_t = \hat{U}_t \hat{S}_t \hat{V}_t^\top$ of target rank $\hat{r} \ll \min(m,n)$ is fed back into the recursion, so only the low-rank factors are stored. This yields an $O(\varepsilon^{-3})$ sample complexity under both standard and generalized smoothness.

$$
\begin{aligned}
(U_t, \Sigma_t, V_t) &= \mathrm{SVD}(M_t) \\
\theta_{t+1} &= \theta_t - \eta_t\, U_t V_t^\top \\
M_{t+1} &= g_{t+1} + (1 - \beta_{t+1})\,(M_t - \tilde{g}_t)
\end{aligned}
$$

where $\theta_t$ are the (matrix-shaped) parameters $W_t$, $\eta_t$ is the learning rate, $M_0 = g_0$, $g_{t+1} = \nabla f(\theta_{t+1}; \xi_{t+1})$ is the stochastic gradient at the new point, $\tilde{g}_t = \nabla f(\theta_t; \xi_{t+1})$ is the gradient at the old point under the same sample $\xi_{t+1}$, and $\beta_{t+1} \in [0,1)$ is the variance-reduction coefficient. The memory-efficient variant substitutes the randomized low-rank approximation $\hat{M}_t$ for $M_t$ in the recursion.

Reference: Feihu Huang, Yuning Luo, Songcan Chen, "LiMuon: Light and Fast Muon Optimizer for Large Models", arXiv 2025. https://arxiv.org/abs/2509.14562

---
[Back to the Canon](../index.md)
