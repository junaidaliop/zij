# DP-FedAdamW

Implements DP-FedAdamW, a differentially private AdamW variant for federated training of large models.

Each client clips per-sample gradients to norm $C$, averages them over the local batch, and injects Gaussian noise to enforce $(\epsilon,\delta)$-differential privacy. The noise inflates the second-moment estimate, so DP-FedAdamW debiases $\hat v_t$ by subtracting the known noise variance before forming the adaptive denominator. A local-global alignment term $\gamma\,\Delta G_t$ nudges each client toward the aggregated global descent direction, and weight decay is decoupled in the AdamW style.

$$
\begin{aligned}
\bar g_{ij} &= g_{ij} \big/ \max\!\left(1, \tfrac{\lVert g_{ij}\rVert_2}{C}\right) \\
\tilde g_t &= \frac{1}{sR}\sum_j \bar g_{ij} + \frac{C}{sR}\,\mathcal{N}(0,\sigma^2 C^2 I) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde g_t \odot \tilde g_t \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\vartheta_t &= \frac{1}{\sqrt{\hat v_t - \left(\tfrac{\sigma C}{sR}\right)^2} + \epsilon} \\
\theta_t &= \theta_{t-1} - \eta\left(\hat m_t \odot \vartheta_t + \gamma\,\Delta G_t - \lambda\,\theta_{t-1}\right)
\end{aligned}
$$

where $g_{ij}$ is the per-sample gradient, $C$ the clipping norm, $\sigma$ the noise multiplier, $sR$ the local batch size, $\mathcal{N}$ Gaussian noise, $\beta_1,\beta_2$ the moment decays, $\epsilon$ the stability constant, $\lambda$ the decoupled weight decay, $\eta$ the learning rate, $\gamma$ the alignment weight, and $\Delta G_t = -\tfrac{1}{SK\eta}\sum_i(\theta_i^{t,k}-\theta_i^{t,0})$ the empirical global descent estimate.

Reference: Jin Liu, Yinbin Miao, Ning Xi, Junkang Liu, "DP-FedAdamW: An Efficient Optimizer for Differentially Private Federated Large Models", arXiv 2026. https://arxiv.org/abs/2602.19945

---
[Back to the Canon](../index.md)
