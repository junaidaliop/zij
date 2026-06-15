# DP-MacAdam

Implements DP-MacAdam, a differentially private optimizer that reuses one set of mean and variance estimates for both adaptive per-example clipping and adaptive momentum.

Standard DP-SGD clips each per-example gradient to a fixed norm before adding Gaussian noise, which discards scale information and couples the clipping bias to a hand-tuned threshold. DP-MacAdam instead centers each per-example gradient by the bias-corrected first moment $\hat{m}_{t-1}$ and rescales it by a per-coordinate bound $b_{t-1}$, normalizes the result to unit norm, averages over the batch, and adds noise. The same Adam-style moment estimates that drive the momentum update are also used to refresh the clipping bound, so clipping adapts automatically to the running gradient statistics.

$$
\begin{aligned}
w_t^{(i)} &= \frac{g_t^{(i)} - \hat{m}_{t-1}}{b_{t-1}}, \qquad
\bar{w}_t^{(i)} = \frac{w_t^{(i)}}{\max\!\left(1,\, \lVert w_t^{(i)} \rVert_2\right)} \\
z_t &\sim \mathcal{N}\!\left(0,\, \tfrac{\sigma^2}{B^2} I\right), \qquad
\tilde{w}_t = \frac{1}{B}\sum_{i=1}^{B} \bar{w}_t^{(i)} + z_t, \qquad
\tilde{g}_t = b_{t-1} \odot \tilde{w}_t + \hat{m}_{t-1} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, \tilde{g}_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, \tilde{g}_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
u_t &= (\tilde{g}_t - \hat{m}_t)^2, \qquad
s_t = \beta_1 s_{t-1} + (1-\beta_1)\, u_t, \qquad
\kappa_t = \frac{2(\beta_1 - \beta_1^{\,t})}{1+\beta_1} \\
\hat{s}_t &= \mathrm{clamp}\!\left(\frac{s_t}{\kappa_t} - \frac{b_{t-1}^2 \sigma^2}{B^2},\; h_1,\, h_2\right), \qquad
b_t = \hat{s}_t^{\,1/4} \left(\sum_{j=1}^{d} \hat{s}_{t,j}^{\,1/2}\right)^{1/2} \\
\theta_t &= \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \gamma}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t^{(i)}$ the per-example gradient, $B$ the batch size, $\sigma$ the noise multiplier, $\beta_1,\beta_2$ the moment decay rates, $\gamma$ a stability constant, $b_t$ the per-coordinate adaptive clipping bound, $s_t/\hat{s}_t$ the (debiased, noise-corrected) variance estimate, $\kappa_t$ the bias-correction factor for the variance EMA, and $h_1,h_2$ lower and upper clamps; all squarings, divisions, and roots are coordinate-wise.

Reference: Naima Tasnim, Lalitha Sankar, Oliver Kosut, "DP-MacAdam: Differentially Private Mechanism with Adaptive Clipping and Adaptive Momentum", arXiv 2026. https://arxiv.org/abs/2606.05435

---
[Back to the Canon](../README.md)
