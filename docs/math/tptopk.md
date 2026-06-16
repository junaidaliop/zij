# TP-TopK

Implements TP-TopK, a two-phase coordinate-sparse DP-SGD that restricts both optimization and noise injection to a learned coordinate support.

Standard DP-SGD injects isotropic Gaussian noise into every coordinate, so the optimizer-facing noise energy grows with the ambient dimension $d$. TP-TopK first runs a short full-parameter DP warm-up that scores each coordinate by its noise-corrected squared gradient signal, selects the top $k = \lfloor \rho d \rfloor$ coordinates as the active support $A$, and then runs masked DP-SGD restricted to $A$. Because masking precedes clipping, per-example sensitivity stays at $C_2$ regardless of $k$, and the effective noise dimension drops from $d$ to $k \ll d$.

In Phase 1, per-example gradients are clipped to $C_1$, averaged with Gaussian noise, used to update $\theta$, and accumulated into a coordinate score that is denoised by subtracting the per-coordinate noise floor; the support is then chosen by Top-$K$. In Phase 2, each per-example gradient is masked to $A$, clipped to $C_2$, averaged with noise injected only on the active coordinates, and applied to $\theta$.

$$
\begin{aligned}
\bar g_{t,i} &= g_{t,i}\cdot\min\!\left(1,\frac{C_1}{\lVert g_{t,i}\rVert_2}\right), &
\tilde g_t &= \frac{1}{B}\Big(\sum_{i\in B_t}\bar g_{t,i} + z_t\Big),\quad z_t\sim\mathcal{N}(0,\sigma_1^2 C_1^2 I_d), \\
\theta_{t+1} &= \theta_t - \eta_t\,\tilde g_t, &
a_p &= \frac{1}{T_1}\sum_{t=1}^{T_1}\tilde g_{t,p}^{\,2} - \left(\frac{\sigma_1 C_1}{B}\right)^2, \\
A &= \mathrm{TopK}(a,k),\quad k=\lfloor\rho d\rfloor, & m_p &= \mathbf{1}[p\in A], \\
\bar g_{t,i}^{A} &= (m\odot g_{t,i})\cdot\min\!\left(1,\frac{C_2}{\lVert m\odot g_{t,i}\rVert_2}\right), &
\tilde g_t^{A} &= \frac{1}{B}\Big(\sum_{i\in B_t}\bar g_{t,i}^{A} + m\odot z_t\Big),\quad z_t\sim\mathcal{N}(0,\sigma_2^2 C_2^2 I_d), \\
\theta_{t+1} &= \theta_t - \eta_t\,\tilde g_t^{A}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_{t,i}$ the per-example gradient, $C_1,C_2$ the $\ell_2$ clip norms, $\sigma_1,\sigma_2$ the noise multipliers, $B$ the batch size, $\rho$ the active ratio, $a$ the noise-corrected coordinate score, $A$ the active support of size $k$, $m\in\{0,1\}^d$ the support mask, and $\odot$ elementwise product. Phase 1 runs for $T_1$ steps to build the score; Phase 2 runs for $T_2$ steps with the support held fixed.

Reference: Huiqi Zhang, Fang Xie, "When Do Fewer Coordinates Suffice in DP-SGD?", arXiv 2026. https://arxiv.org/abs/2606.04375

---
[Back to the Canon](../index.md)
