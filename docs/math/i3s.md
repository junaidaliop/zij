# I3S

Implements I3S, importance-sampling subspace selection for memory-efficient low-rank optimization.

Low-rank optimizers such as GaLore project gradients onto the top singular subspace of the gradient, but periodically refreshing that subspace via plain SVD keeps re-selecting the same dominant directions, leaving the trajectory trapped in a frozen subspace. I3S instead samples $r$ singular vectors without replacement, with probability proportional to their singular values, so the projection explores more diverse directions while still favoring the dominant ones. The sampled columns of $U_t$ form the projection $P_t$, which is reused for $\tau$ steps and then resampled; the projected low-rank state is updated with Adam.

$$
\begin{aligned}
G_t &= \nabla_\theta f(\theta_{t-1}) \\
U_t, \Sigma_t, V_t &= \mathrm{SVD}(G_t), \qquad \omega_i = \frac{\sigma_i}{\sum_{j=1}^{m}\sigma_j} \\
\mathcal{I} &= \mathrm{Sample}\big([m],\, r,\, \omega\big), \qquad P_t = U_t[:,\mathcal{I}] \quad (t \bmod \tau = 0,\ \text{else } P_t = P_{t-1}) \\
R_t &= P_t^{\top} G_t \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) R_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) R_t \odot R_t \\
N_t &= P_t \frac{m_t}{\sqrt{v_t} + \epsilon} \\
\theta_t &= \theta_{t-1} - \eta\, N_t
\end{aligned}
$$

where $\sigma_i$ are the singular values of the gradient, $\mathrm{Sample}([m], r, \omega)$ draws $r$ indices from $\{1,\dots,m\}$ without replacement with weights $\omega$, $P_t$ is the low-rank projection refreshed every $\tau$ steps, $R_t$ the projected gradient, $m_t, v_t$ the projected Adam moments, $\beta_1, \beta_2$ their decay rates, $\eta$ the learning rate, and $\epsilon$ a stability constant.

Reference: Haochen Zhang, Junze Yin, Guanchu Wang, Zirui Liu, Tianyi Zhang, Anshumali Shrivastava, Lin Yang, Vladimir Braverman, "I3S: Importance Sampling Subspace Selection for Low-Rank Optimization in LLM Pretraining", ICML 2025. https://arxiv.org/abs/2502.05790

---
[Back to the Canon](../index.md)
