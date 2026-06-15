# SF-NorMuon

Implements SF-NorMuon, a schedule-free variant of NorMuon that needs no decay schedule and yields a well-trained model at every step.

SF-NorMuon fuses three ingredients. From Muon it keeps the *polar transform*: the momentum buffer is orthogonalized via Newton-Schulz so the update is the steepest-descent direction under the spectral norm. From NorMuon it adds *row-wise normalization*: a running second moment of the squared polar update is tracked per row and used to rescale it, much like Adam's RMS normalization. From schedule-free optimization it borrows two coupled sequences — a fast iterate $z_t$ that moves at a constant learning rate, and an averaged iterate $x_t$ (the output, where loss is evaluated) formed by an online weighted average. Gradients are queried at an interpolation $y_t$ of the two.

Because the averaging weight $c_{t+1}=\eta_t^2/s_t$ is driven by the cumulative sum of squared learning rates, no learning-rate decay schedule is required; the method is "anytime," producing a deployable model at any stopping point.

$$
\begin{aligned}
y_t &= (1-\beta_1)\, z_t + \beta_1\, x_t \\
g_t &= \nabla_\theta \mathcal{L}(y_t, \zeta_t) \\
m_t &= \mu\, m_{t-1} + (1-\mu)\, g_t \\
P_t &= \mathrm{polar}(m_t) \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, \mathrm{mean_{cols}}(P_t \odot P_t) \\
\hat{P}_t &= P_t \oslash \left(\sqrt{\mathrm{ExpandRows}(v_t)} + \epsilon\right) \\
\eta_t &= \eta \cdot \min\!\left(1,\ t/T_{\mathrm{warmup}}\right), \qquad
\hat{\eta}_t = 0.2\, \eta_t\, \frac{\sqrt{mn}}{\lVert \hat{P}_t \rVert_F} \\
z_{t+1} &= z_t - \eta\lambda\, z_t - \hat{\eta}_t\, \hat{P}_t \\
s_t &= s_{t-1} + \eta_t^2, \qquad c_{t+1} = \eta_t^2 / s_t \\
x_{t+1} &= (1-c_{t+1})\, x_t + c_{t+1}\, z_{t+1}
\end{aligned}
$$

where $z_t$ is the fast iterate, $x_t$ the averaged (output) iterate, $y_t$ the interpolation point at which the gradient $g_t$ is taken, $m_t$ the momentum buffer with decay $\mu$, $\mathrm{polar}(\cdot)$ the orthogonal polar factor $UV^\top$ of the SVD (computed by Newton-Schulz), $\mathrm{mean_{cols}}$ the per-row mean across columns and $\mathrm{ExpandRows}$ its broadcast back to the $m\times n$ matrix shape, $v_t$ the row-wise second moment with decay $\beta_2$, $\hat{P}_t$ the row-normalized update, $\beta_1$ the interpolation weight, $\eta$ the base learning rate with warmup over $T_{\mathrm{warmup}}$ steps, $\hat{\eta}_t$ the RMS-matched effective step, $\lambda$ the weight decay, $s_t$ the cumulative sum of squared learning rates, $c_{t+1}$ the schedule-free averaging coefficient, $\odot$ and $\oslash$ element-wise product and division, and $\epsilon$ a stability constant.

Reference: Anuj Apte, Pranav Deshpande, Niraj Kumar, Shouvanik Chakrabarti, Junhyung Lyle Kim, "Anytime Training with Schedule-Free Spectral Optimization", arXiv 2026. https://arxiv.org/abs/2605.23061

---
[Back to the Canon](../README.md)
