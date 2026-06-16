# AdamHD (AdamHuberDecay)

Implements AdamHD (AdamHuberDecay), Adam with a decoupled Huber-decay proximal regularizer in place of L2 weight decay.

AdamHD keeps Adam's momentum and second-moment estimates unchanged, then replaces AdamW's uniform L2 shrinkage with a Huber-shaped regularizer $R_\delta(\theta) = \sum_i H_\delta(\theta_i)$, where $H_\delta$ is quadratic for small magnitudes and linear beyond a threshold $\delta$. Applied as a proximal step, this decays small weights quadratically (stabilizing) while capping the shrinkage force on large weights at a constant $\ell_1$-like magnitude, preventing over-decay of grown parameters.

After the standard Adam preconditioned step, the next iterate is the proximal map of $\alpha_t \lambda R_\delta$, which admits a closed form per coordinate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t \odot g_t \\
\tilde{\theta}_t &= \theta_t - \alpha_t \frac{m_t}{\sqrt{v_t}+\epsilon} \\
\theta_{t+1} &= \mathrm{prox}_{\tau R_\delta}(\tilde{\theta}_t) =
\begin{cases}
\dfrac{\tilde{\theta}_t}{1+\tau}, & |\tilde{\theta}_t| \le (1+\tau)\,\delta_t \\
\tilde{\theta}_t - \tau\,\delta_t\,\mathrm{sign}(\tilde{\theta}_t), & |\tilde{\theta}_t| > (1+\tau)\,\delta_t
\end{cases}
\end{aligned}
$$

where $\tau = \alpha_t \lambda$, $\alpha_t$ is the learning rate, $\lambda$ the decay strength, $\delta_t$ a per-layer threshold set from an exponential moving average of parameter magnitudes, $\beta_1,\beta_2$ the moment decay rates, and $\epsilon$ a stability constant. The prox is applied elementwise.

Reference: Fu-Ming Guo and Yingfang Fan, "AdamHD: Decoupled Huber Decay Regularization for Language Model Pre-Training", ScaleOpt Workshop 2025. https://arxiv.org/abs/2511.14721

---
[Back to the Canon](../index.md)
