# GRASS

Implements GRASS, low-memory LLM training via structured sparse gradient projection.

GRASS optimizes full parameters through a low-dimensional subspace, like GaLore, but replaces the dense projection with a sparse one $P^\top = \rho B$, where $B$ is a row-selection matrix and $\rho$ is a diagonal scaling. Rows are chosen by gradient row norm (top-$r$ deterministically, or sampled with $\rho_{jj} = 1/\sqrt{r\,q_{\sigma_j}}$ for an unbiased estimator). Because $P$ is sparse, the gradient never has to be materialized in full: projection, optimizer storage, and the weight update all cost $O(rn)$ instead of $O(mn)$.

The compressed gradient $g_t = P^\top \nabla L(\theta_t)$ is fed to a standard Adam step in the $r \times n$ subspace, and the resulting update is projected back up through the same sparse $P$. The projection is refreshed every $K$ steps, at which point the moments are reset.

$$
\begin{aligned}
g_t &= P^\top \nabla L(\theta_t), \qquad P^\top = \rho B \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{2} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
\Delta_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
\theta_{t+1} &= \theta_t - \alpha\, \eta\, P\, \Delta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\alpha$ a tuned scale factor, $P = (\rho B)^\top$ the sparse up-projection, $g_t$ the projected gradient, $m_t, v_t$ the first and second moments in the subspace, $\beta_1, \beta_2$ the Adam decay rates, and $\epsilon$ the stability constant.

Reference: Aashiq Muhamed, Oscar Li, David Woodruff, Mona Diab, Virginia Smith, "Grass: Compute Efficient Low-Memory LLM Training with Structured Sparse Gradients", arXiv 2024. https://arxiv.org/abs/2406.17660

---
[Back to the Canon](../index.md)
