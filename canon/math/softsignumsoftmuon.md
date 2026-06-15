# SoftSignum / SoftMuon

Implements SoftSignum / SoftMuon, sign-based updates softened by a temperature-scaled $\tanh$ (and its spectral analogue for matrices).

The sign optimizer replaces the gradient with its sign, which discards magnitude and treats every coordinate identically. SoftSignum interpolates between signed and raw momentum by passing the momentum through $\tanh(\tau_t m_t)$: at high temperature it recovers $\mathrm{sign}(m_t)$, and for small arguments it behaves linearly, so coordinates with small momentum keep their scale. The temperature $\tau_t$ is set adaptively from a quantile of the momentum magnitudes, which adapts the saturation point to the parameter group's heterogeneity.

SoftMuon lifts the same idea to matrix-valued parameters by applying the algebraic soft sign $\phi(x)=x/\sqrt{1+x^2}$ to the singular values of the momentum matrix. Writing the momentum SVD as $M_t = U\Sigma V^\top$, the spectral soft sign $\Phi_\tau(M_t)=M_t(M_t^\top M_t + \tau_t^{-2} I)^{-1/2}$ equals $U\,\mathrm{diag}\big(\tau_t\sigma_i/\sqrt{1+(\tau_t\sigma_i)^2}\big)V^\top$, smoothly transitioning from a Muon-style orthogonalized step at high temperature toward the raw momentum at low temperature.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t,\\
\theta_t &= (1-\gamma\lambda)\,\theta_{t-1} - \gamma\, \tanh(\tau_t\, m_t) && \text{(SoftSignum)},\\
\Theta_t &= (1-\gamma\lambda)\,\Theta_{t-1} - \gamma\, M_t\big(M_t^\top M_t + \tau_t^{-2} I\big)^{-1/2} && \text{(SoftMuon)},\\
\tau_t &= \max\Big\{1,\; \frac{\mathrm{arctanh}(1-\epsilon)}{q_{p}(|m_t|)}\Big\}.
\end{aligned}
$$

where $\gamma$ is the learning rate, $\beta$ the momentum decay, $\lambda$ the (decoupled) weight decay, $\tau_t$ the adaptive temperature, $\epsilon$ the saturation tolerance, and $q_{p}(|m_t|)$ a quantile of the momentum magnitudes; $M_t$ is the matrix momentum with SVD $U\Sigma V^\top$ used in the spectral update.

Reference: Dmitrii Feoktistov, Timofey Belinsky, Andrey Veprikov, Amir Zainullin, Aleksandr Beznosikov, "Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling", arXiv 2026. https://arxiv.org/abs/2605.31371

---
[Back to the Canon](../README.md)
