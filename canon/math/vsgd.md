# VSGD

Implements VSGD (Variational Stochastic Gradient Descent), a Bayesian optimizer that treats the true gradient as a latent variable inferred from noisy stochastic gradients.

VSGD places a hierarchical Gaussian–Gamma model on the observed gradient $\hat{g}_t$ and the latent true gradient, then performs coordinate-wise variational inference. The posterior mean $\mu_{t,g}$ and variance $\sigma_{t,g}^2$ of the gradient are obtained by a precision-weighted fusion of the running estimate and the new observation, where the precisions are themselves estimated online through Gamma rate parameters $b_{t,g}$ (state precision) and $b_{t,\hat{g}}$ (observation precision). The parameter step rescales the inferred mean by the inverse root second moment, recovering an Adam-like denominator from first principles.

$$
\begin{aligned}
\mu_{t,g} &= \frac{b_{t-1,\hat{g}}}{b_{t-1,\hat{g}} + b_{t-1,g}}\,\mu_{t-1,g} + \frac{b_{t-1,g}}{b_{t-1,\hat{g}} + b_{t-1,g}}\,\hat{g}_t \\
\sigma_{t,g}^2 &= \left(\frac{a_{t-1,g}}{b_{t-1,g}} + \frac{a_{t-1,\hat{g}}}{b_{t-1,\hat{g}}}\right)^{-1} \\
b'_{t,g} &= \gamma + \tfrac{1}{2}\big(\sigma_{t,g}^2 + (\mu_{t,g} - \mu_{t-1,g})^2\big) \\
b'_{t,\hat{g}} &= K_g\,\gamma + \tfrac{1}{2}\big(\sigma_{t,g}^2 + (\mu_{t,g} - \hat{g}_t)^2\big) \\
b_{t,g} &= (1 - \rho_{t,1})\,b_{t-1,g} + \rho_{t,1}\,b'_{t,g} \\
b_{t,\hat{g}} &= (1 - \rho_{t,2})\,b_{t-1,\hat{g}} + \rho_{t,2}\,b'_{t,\hat{g}} \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{\mu_{t,g}^2 + \sigma_{t,g}^2}}\,\mu_{t,g}
\end{aligned}
$$

where $\hat{g}_t$ is the stochastic gradient, $\mu_{t,g}$ and $\sigma_{t,g}^2$ are the variational posterior mean and variance of the true gradient, $a_{t,g} = a_{t,\hat{g}} = \gamma + \tfrac{1}{2}$ are the (fixed) Gamma shape parameters, $b_{t,g}$ and $b_{t,\hat{g}}$ are the Gamma rate parameters for the state and observation precisions, $\gamma$ is the prior rate and $K_g$ a prior scaling constant, $\rho_{t,1} = t^{-\kappa_1}$ and $\rho_{t,2} = t^{-\kappa_2}$ are decaying step sizes (with $\kappa_1, \kappa_2 \in (0.5, 1]$ satisfying the Robbins–Monro conditions), and $\eta$ is the learning rate.

Reference: Haotian Chen, Anna Kuzina, Babak Esmaeili, Jakub M. Tomczak, "Variational Stochastic Gradient Descent for Deep Neural Networks", ICML 2024. https://arxiv.org/abs/2404.06549

---
[Back to the Canon](../README.md)
