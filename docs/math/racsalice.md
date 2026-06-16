# RACS / Alice

Implements RACS and Alice, structured Fisher-approximation optimizers for memory-efficient LLM training.

Both stem from a structured approximation of the Fisher information. RACS (Row And Column Scaled SGD) preconditions the gradient $G_t$ by two diagonal scalings, one acting on rows and one on columns, each maintained by an exponential moving average and estimated through a fixed-point iteration. Alice extends this with a low-rank subspace: it projects the gradient onto a leading eigenbasis $U_t$ (refreshed every $K$ steps by subspace switching), runs Adam-style moments inside that subspace, and adds a compensation term that recovers signal from the discarded complement directions.

For RACS, with diagonal scalings $s_t,q_t$ tracking $\mathrm{Diag}(S_t),\mathrm{Diag}(Q_t)$:

$$
\begin{aligned}
s_t &= \beta\, s_{t-1} + (1-\beta)\,\mathrm{Diag}(S_t), \\
q_t &= \beta\, q_{t-1} + (1-\beta)\,\mathrm{Diag}(Q_t), \\
\tilde{G}_t &= \mathrm{Diag}(q_t)^{-1/2}\, G_t\, \mathrm{Diag}(s_t)^{-1/2}, \\
\eta_t &= \gamma / \max\{\,\|\tilde{G}_t\|\,\phi_{t-1},\, \gamma\,\}, \\
W_{t+1} &= W_t - \lambda\,\eta_t\,\alpha\,\tilde{G}_t.
\end{aligned}
$$

For Alice, with projected gradient $\sigma_t = U_t^{\top} G_t$:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\,\sigma_t, \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\,\sigma_t^{\odot 2}, \\
\omega_t &= m_t / \sqrt{v_t}, \\
C_t &= \sqrt{m-r}\,(G_t - U_t U_t^{\top} G_t)\,\mathrm{Diag}(p_t)^{-1/2}, \\
W_{t+1} &= W_t - \lambda\,\alpha\,(U_t\,\omega_t + \alpha_c\, C_t).
\end{aligned}
$$

where $W$ are the weights, $G_t$ the gradient, $\lambda$ the learning rate, $\alpha$ a scaling factor, $\beta,\beta_1,\beta_2$ EMA decay rates, $S_t,Q_t$ the column/row preconditioner estimates, $\gamma$ a norm-growth limit with running norm $\phi_t$, $U_t$ the rank-$r$ subspace basis (for an $m\times n$ weight), $\sigma_t^{\odot 2}$ the elementwise square, $p_t$ the complement scaling, $C_t$ the compensation for discarded directions weighted by $\alpha_c$, and $\epsilon$ a small constant inside the square roots for stability.

Reference: Wenbo Gong, Meyer Scetbon, Chao Ma, Edward Meeds, "Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension", arXiv 2025. https://arxiv.org/abs/2502.07752

---
[Back to the Canon](../index.md)
