# DP-AdamW

Implements DP-AdamW, AdamW with decoupled weight decay and bias correction under differential privacy.

DP-AdamW privatizes AdamW by clipping each per-sample gradient to a fixed $\ell_2$ norm $C$ and adding Gaussian noise before forming the moment estimates, so the gradient release satisfies the standard Gaussian DP mechanism. The adaptive moments, bias correction, and decoupled weight decay then follow AdamW exactly, applied to the privatized gradient. The paper studies whether decoupling the weight decay (rather than coupling it through the adaptive denominator, as in DP-Adam) and retaining bias correction improve private training.

$$
\begin{aligned}
\tilde{g}_t &= \frac{1}{B}\sum_i \frac{g_{t,i}}{\max\!\left(1,\ \|g_{t,i}\|_2 / C\right)} + \frac{1}{B}\,\mathcal{N}\!\left(0, \sigma^2 C^2 I\right) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde{g}_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde{g}_t^{2} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}} + \lambda\,\theta_{t-1}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_{t,i}$ the per-sample gradient over a batch of size $B$, $C$ the clipping norm, $\sigma$ the noise multiplier, $\tilde{g}_t$ the clipped-and-noised batch gradient, $m_t,v_t$ the first and second moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected forms, $\lambda$ the decoupled weight decay, and $\epsilon$ the stability constant.

Reference: Jay Chooi, Kevin Cong, Russell Li, Lillian Sun, "DP-AdamW: Investigating Decoupled Weight Decay and Bias Correction in Private Deep Learning", arXiv 2025. https://arxiv.org/abs/2511.07843

---
[Back to the Canon](../index.md)
