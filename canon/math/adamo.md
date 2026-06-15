# AdamO

Implements AdamO, an Adam-style optimizer that decouples isometry regularization from the gradient update, analogous to AdamW.

To preserve plasticity in continual learning, the authors promote dynamical isometry by keeping each layer's Jacobian singular values near one, enforced through a Gram-deviation penalty $\mathcal{R}_{\mathrm{iso}}$. Rather than folding this penalty into the loss (where its gradient would be rescaled by Adam's adaptive denominator), AdamO maintains the adaptive moments using only the task gradient $g_t$, then applies the isometry step separately, mirroring decoupled weight decay.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta_{\mathrm{iso}} \, \lambda \, r_t
\end{aligned}
$$

where $g_t = \nabla_\theta \ell_t(\theta)$ is the stochastic task gradient, $r_t = \nabla_\theta \sum_\ell \mathcal{R}_{\mathrm{iso}}(W_\ell)$ is the isometry-regularizer gradient, $\eta$ is the base learning rate, $\eta_{\mathrm{iso}}$ is the isometry step size (set to $\eta$ by default), $\beta_1, \beta_2$ are the moment decay rates, $\epsilon$ is the stability constant, and $\lambda$ is the isometry-regularization strength. The penalty is $\mathcal{R}_{\mathrm{iso}}(W_\ell) = \lVert W_\ell^\top W_\ell - I \rVert_F^2$ when $d_\ell \ge d_{\ell-1}$ and $\lVert W_\ell W_\ell^\top - I \rVert_F^2$ otherwise.

Reference: Andries Rosseau, Robert Müller, Ann Nowé, "Preserving Plasticity in Continual Learning via Dynamical Isometry", ICML 2026. https://arxiv.org/abs/2606.09762

---
[Back to the Canon](../README.md)
