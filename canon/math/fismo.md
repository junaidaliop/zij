# FISMO

Implements FISMO (Fisher-Structured Momentum-Orthogonalized optimizer), a Kronecker-factored second-order method that orthogonalizes momentum in a whitened geometry.

FISMO models the per-layer Fisher information as a Kronecker product of a left factor $P_t \in \mathbb{R}^{m\times m}$ and a right factor $Q_t \in \mathbb{R}^{n\times n}$, each maintained by an exponential moving average and trace-normalized to control scale. The raw gradient is whitened by these factors before momentum is accumulated, and the momentum is then orthogonalized with the matrix polar factor (computed via Newton-Schulz iterations, $\mathrm{Polar}(M)=UV^\top$ for an SVD $M=U\Sigma V^\top$). The orthogonalized step is mapped back through the preconditioners, combining the curvature awareness of Fisher methods with the spectral conditioning of Muon-style updates.

$$
\begin{aligned}
L_t &\leftarrow \tfrac{1}{n} G_t Q_{t-1}^{-1} G_t^\top + \mu\,\tfrac{\mathrm{tr}(P_{t-1})}{m} I_m, &
P_t &\leftarrow \mathrm{sym}\!\Big(\tfrac{m}{\mathrm{tr}(\tilde P_t)}\tilde P_t\Big),\quad \tilde P_t = \gamma P_{t-1} + (1-\gamma)L_t \\
R_t &\leftarrow \tfrac{1}{m} G_t^\top P_t^{-1} G_t + \mu\,\tfrac{\mathrm{tr}(Q_{t-1})}{n} I_n, &
Q_t &\leftarrow \mathrm{sym}\!\Big(\tfrac{n}{\mathrm{tr}(\tilde Q_t)}\tilde Q_t\Big),\quad \tilde Q_t = \gamma Q_{t-1} + (1-\gamma)R_t \\
\tilde G_t &\leftarrow P_t^{-1/2} G_t Q_t^{-1/2}, &
M_t &\leftarrow \beta M_{t-1} + (1-\beta)\tilde G_t \\
\Delta W_t &\leftarrow P_t^{-1/2}\,\mathrm{Polar}(M_t)\,Q_t^{-1/2}, &
W_t &\leftarrow W_{t-1} - \eta\,\Delta W_t
\end{aligned}
$$

where $W_t\in\mathbb{R}^{m\times n}$ are the matrix-shaped parameters, $G_t$ the minibatch gradient, $\eta$ the learning rate, $\beta$ the momentum coefficient, $\gamma$ the EMA decay of the Fisher factors, $\mu$ the damping factor, $\mathrm{sym}(A)=\tfrac12(A+A^\top)$ symmetrizes, and $\mathrm{Polar}(M)=UV^\top$ is the orthogonal polar factor of $M$.

Reference: Chenrui Xu, Wenjing Yan, Ying-Jun Angela Zhang, "FISMO: Fisher-Structured Momentum-Orthogonalized Optimizer", ICML 2026. https://arxiv.org/abs/2601.21750

---
[Back to the Canon](../README.md)
