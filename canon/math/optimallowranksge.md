# Optimal Low-Rank SGE

Implements Optimal Low-Rank SGE, a memory-efficient optimizer that descends in randomly projected low-rank subspaces with the projector chosen to minimize gradient-estimator variance.

At each outer step the parameters $\theta$ are confined to an $r$-dimensional subspace spanned by a sampled projection matrix $V_t$. A small low-rank coordinate $B$ is updated by $K$ inner gradient steps inside that subspace and then lifted back to the full space, so only the $r$-column factor is kept in memory rather than the full gradient. The projector is sampled so that $\mathbb{E}[V_t V_t^\top] = c\,I_n$ (weak unbiasedness); the contribution is an instance-dependent sampling distribution that allocates the rank budget across spectral directions to attain the minimum mean-squared error of the estimated gradient.

$$
\begin{aligned}
B_{t,0} &= 0, \\
B_{t,k+1} &= B_{t,k} - \eta_t \, \nabla_B F\!\left(\xi_{t,k},\, \theta_t + B V_t^\top\right)\big|_{B = B_{t,k}}, \quad k = 0,\dots,K-1, \\
\theta_{t+1} &= \theta_t + B_{t,K} V_t^\top, \\
\pi_i^\star &= \min\!\left\{1,\ \frac{(r-\tau)\sqrt{\sigma_i}}{\sum_{j:\pi_j^\star<1}\sqrt{\sigma_j}}\right\}, \qquad
V_t = Q_J \, \mathrm{diag}\!\left(\sqrt{c/\pi_i^\star}\right)_{i\in J}.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta_t$ the step size, $F$ the per-sample loss on data $\xi_{t,k}$, $B$ the low-rank inner coordinate of width $r$, and $V_t$ the sampled projection matrix; $\Sigma = Q\,\mathrm{diag}(\sigma_1,\dots,\sigma_n)\,Q^\top$ is the spectral decomposition of the gradient covariance, $\pi_i^\star$ the optimal inclusion probabilities, $\tau = \#\{i:\pi_i^\star=1\}$, $J\subset\{1,\dots,n\}$ a sampled index set of size $r$ with $\Pr(i\in J)=\pi_i^\star$, and $c$ the isotropy constant.

Reference: Zehao Li, Tao Ren, Zishi Zhang, Xi Chen, Yijie Peng, "Optimal Low-Rank Stochastic Gradient Estimation for LLM Training", arXiv 2026. https://arxiv.org/abs/2603.20632

---
[Back to the Canon](../README.md)
