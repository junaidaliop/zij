# FiBeR

Implements FiBeR, a differentially private adaptive optimizer that denoises temporally filtered privatized gradients before feeding them to AdamW.

When privatized gradients are temporally filtered to reduce variance, the filtering also reshapes the DP noise statistics seen by the second-moment accumulator, so bias corrections calibrated for unfiltered noise become miscalibrated. FiBeR forms each step's gradient from a two-point observation, privatizes it with clipping and Gaussian noise, then performs denoising in innovation (residual) space rather than smoothing the gradient directly. The integrated filtered estimate drives standard AdamW moments, and a filter-aware second-moment correction subtracts the attenuated DP noise contribution using a closed-form factor before the parameter update.

$$
\begin{aligned}
u_t(\xi) &= a\,\nabla f(\theta_t + \gamma d_{t-1};\xi) + (1-a)\,\nabla f(\theta_t;\xi), \quad a = \frac{1-\kappa}{\kappa\gamma},\; d_{t-1} = \theta_t - \theta_{t-1} \\
g_t &= \frac{1}{B}\sum_{\xi \in B_t} \mathrm{clip}(u_t(\xi), C) + w_t, \quad w_t \sim \mathcal{N}(0,\sigma_w^2 I),\; \sigma_w^2 = (\sigma_{DP} C / B)^2 \\
\nu_t &= g_t - \tilde{g}_{t-1}, \quad r_t = (1-\omega)\,r_{t-1} + \omega\,\nu_t, \quad \tilde{g}_t = \tilde{g}_{t-1} + r_t \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde{g}_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,\tilde{g}_t \odot \tilde{g}_t \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t+1}}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^{t+1}} \\
\bar{v}_t &= \max\!\big(\hat{v}_t - A(\omega)\,\sigma_w^2,\; \epsilon_v\big), \quad A(\omega) = \frac{2-\omega}{4-3\omega} \\
\theta_{t+1} &= (1-\eta\lambda)\,\theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\bar{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\lambda$ the decoupled weight decay, $g_t$ the privatized minibatch gradient over batch $B_t$ of size $B$, $\tilde{g}_t$ the filtered gradient estimate with residual stream $r_t$ and innovation $\nu_t$, $\kappa$ and $\gamma$ the two-point observation geometry, $\omega$ the innovation gain, $C$ the clipping norm, $\sigma_{DP}$ the DP noise multiplier, $\beta_1,\beta_2$ the moment decays, $A(\omega)$ the closed-form noise-attenuation factor for the filter-aware second-moment correction, and $\epsilon,\epsilon_v$ small stability constants.

Reference: Duc Dm, Thao Do, Minh Son Hoang, Tran Le Duc Anh, Daeyoung Kim, Huy Nguyen, "FiBeR: A Differentially Private Optimizer with Filter-Aware Innovation Bias Correction", ICML 2025. https://arxiv.org/abs/2605.03425

---
[Back to the Canon](../README.md)
