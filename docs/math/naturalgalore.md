# Natural GaLore

Implements Natural GaLore, a low-rank gradient projection method that applies an inverse-Fisher (natural gradient) correction within the projected subspace.

Like GaLore, the gradient is projected onto a low-rank subspace spanned by the top singular vectors $P_t$ of the gradient, and the optimizer state lives in that compact space. Natural GaLore adds a second-order step: an empirical Fisher matrix is built from a sliding window of recent projected gradients, and its inverse is applied to the current projected gradient via the Woodbury identity, so the natural-gradient direction is computed cheaply by solving a small $s \times s$ system. Adam-style moments are then run on the corrected gradient, and the result is projected back to full dimension.

$$
\begin{aligned}
g_t &= P_t^\top \nabla_\theta \Phi(\theta_t) \\
G_t &= [\,g_t,\, g_{t-1},\, \dots,\, g_{t-s}\,] \\
\tilde{g}_t &= \tfrac{1}{\lambda} g_t - \tfrac{1}{\lambda} G_t (\lambda I + G_t^\top G_t)^{-1} G_t^\top g_t \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{g}_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \tilde{g}_t^2 \\
u_t &= \frac{m_t}{\sqrt{v_t + \epsilon}} \\
\theta_{t+1} &= \theta_t - \eta\, P_t u_t
\end{aligned}
$$

where $P_t$ are the top-$r$ left singular vectors of the gradient (the projection onto the low-rank subspace), $g_t$ is the projected gradient, $G_t$ is the window of the last $s$ projected gradients forming the empirical Fisher $\lambda I + G_t G_t^\top$, $\tilde{g}_t$ is the natural (inverse-Fisher-preconditioned) gradient obtained via the Woodbury identity, $\lambda$ is the Tikhonov regularization constant, $m_t,v_t$ are the first and second moments with decay rates $\beta_1,\beta_2$, $\eta$ is the learning rate, and $\epsilon$ ensures numerical stability.

Reference: Arijit Das, "Natural GaLore: Accelerating GaLore for Memory-Efficient LLM Training and Fine-tuning", arXiv 2024. https://arxiv.org/abs/2410.16029

---
[Back to the Canon](../index.md)
