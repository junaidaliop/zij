# Atlas

Implements Atlas, a curvature-aware optimizer that combines a Hutchinson trace estimate, a trust-radius clamp, and a fixed three-phase descent cascade.

Atlas centralizes and median-clips the gradient, then periodically probes the Hessian trace with a single Hutchinson product $\widehat{\mathrm{tr}}(H_t)=|v^\top \nabla^2 L(\theta_t)\,v|$ (Rademacher $v$) to set a trust radius $r_t=\sqrt{\widehat{\mathrm{tr}}(H_t)}$. The candidate step is produced by one of three phase-specific rules selected by training progress $\phi_t=t/T$ (AdaGrad-momentum, then a rectified AdamW step with a trust-ratio scaling, then SGD-Nesterov), rescaled so its norm never exceeds $r_t$, and applied with decoupled weight decay. A Safe-Step monitor re-evaluates the loss and rolls the step back when it rises by more than the tolerance factor; optional LookAhead averaging and a CheapSAM perturbation wrap the core step.

$$
\begin{aligned}
g_t^\star &= \mathrm{AGC2}\!\left(g_t - \mathrm{mean}(g_t)\right), \qquad r_t = \sqrt{\widehat{\mathrm{tr}}(H_t)} \\
\Delta_t^{\mathrm{adg}} &= \eta\,\frac{m_{t+1}}{\sqrt{v_{t+1}}+\epsilon}, &\quad& v_{t+1}=v_t+g_t^{\star 2},\;\; m_{t+1}=0.9\,m_t+g_t^\star \\
\Delta_t^{\mathrm{rad}} &= \eta\,\tau_{t+1}\,u_t, &\quad& u_t=\frac{0.9\,m_t+0.1\,g_t^\star}{\sqrt{r_{t+1}/(1-0.999^{\,t})}+\widehat{\mathrm{tr}}(H_t)},\;\; \tau_{t+1}=0.9\,\tau_t+0.1\,\frac{\|\theta_t\|}{\|u_t\|+\epsilon} \\
\Delta_t^{\mathrm{sgd}} &= \eta\left(0.9\,m_{t+1}+g_t^\star\right), &\quad& m_{t+1}=0.9\,m_t+0.1\,g_t^\star \\
\Delta_t^{\mathrm{raw}} &= \mathbb{1}[\phi_t<0.2]\,\Delta_t^{\mathrm{adg}} + \mathbb{1}[0.2\le\phi_t<0.8]\,\Delta_t^{\mathrm{rad}} + \mathbb{1}[\phi_t\ge 0.8]\,\Delta_t^{\mathrm{sgd}} \\
\Delta_t &= \min\!\left(1,\;\frac{r_t}{\|\Delta_t^{\mathrm{raw}}\|+\epsilon}\right)\Delta_t^{\mathrm{raw}} \\
\theta_{t+1} &= \theta_t - \Delta_t - \eta\,\lambda\,\theta_t
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta$ the cosine-scheduled learning rate, $g_t^\star$ the centralized and adaptively clipped gradient, $m_t/v_t/r_t$ the first- and second-moment buffers, $\tau_t$ the running trust ratio, $\widehat{\mathrm{tr}}(H_t)$ the Hutchinson Hessian-trace estimate (refreshed every $h$ steps), $\phi_t=t/T$ the fractional training progress selecting the active phase, $\lambda$ the decoupled weight decay, and $\epsilon$ a small stabilizer; $\mathrm{AGC2}$ rescales the gradient when its norm exceeds a median-of-weights threshold.

Reference: János Horváth, "Atlas – Rethinking Optimizer Design for Stability and Speed", OPT 2025: 17th Annual Workshop on Optimization for Machine Learning (NeurIPS workshop). https://opt-ml.org/papers/2025/paper6.pdf

---
[Back to the Canon](../index.md)
