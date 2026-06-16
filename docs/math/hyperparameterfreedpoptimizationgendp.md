# Hyperparameter-free DP optimization (GeN-DP)

Implements HyFreeDP, a hyperparameter-free differentially private optimizer that sets its own learning rate via a privatized generalized-Newton step.

Standard DP training fixes a learning rate schedule by grid search, which is costly and itself leaks information through the data-dependent choice. HyFreeDP removes this by combining automatic per-sample gradient clipping with the generalized Newton (GeN) learning rate, adapted to DP. Each step privatizes both the gradient direction $m_t$ and a few loss probes along that direction; fitting a quadratic to the probed losses yields a privatized slope and curvature whose ratio is the learning rate. The direction can be passed through any base optimizer (momentum, Adam-style preconditioning, weight decay) as DP post-processing.

$$
\begin{aligned}
m_t &= \frac{1}{B}\Big(\sum_{i} \min\!\big(\tfrac{R_g}{\lVert g_i\rVert},1\big)\,g_i + \sigma_g R_g\, z_g\Big),\qquad z_g\sim\mathcal{N}(0,I)\\
\tilde{L}(\theta) &= \frac{1}{B}\Big(\sum_{i} \min\!\big(\tfrac{R_l}{L_i},1\big)\,L_i + \sigma_l R_l\, z_l\Big),\qquad z_l\sim\mathcal{N}(0,1)\\
(b,a) &= \arg\min_{a,b}\ \sum_{j}\Big|\,\tilde{L}(\theta-\eta_j m_t) - \big(\tilde{L}(\theta) - b\,\eta_j + \tfrac{a}{2}\,\eta_j^{2}\big)\Big|^{2}\\
\eta_t &= \frac{b}{a} = \frac{(G^{\top}m)_{\mathrm{DP}}}{(m^{\top}Hm)_{\mathrm{DP}}}\\
\theta_{t+1} &= \theta_t - \eta_t\, m_t
\end{aligned}
$$

where $g_i,L_i$ are the per-sample gradient and loss, $R_g,R_l$ the gradient and loss clipping norms, $\sigma_g,\sigma_l$ the noise multipliers, $B$ the batch size, and $z_g,z_l$ Gaussian DP noise; $\tilde{L}$ is the privatized loss probed at offsets $\eta_j$ along $-m_t$, and the least-squares fit returns the privatized directional slope $b=(G^{\top}m)_{\mathrm{DP}}$ and curvature $a=(m^{\top}Hm)_{\mathrm{DP}}$, giving the generalized-Newton step $\eta_t=b/a$.

Reference: Zhiqi Bu, Ruixuan Liu, "Towards hyperparameter-free optimization with differential privacy", ICLR 2025. https://arxiv.org/abs/2503.00703

---
[Back to the Canon](../index.md)
