# DP-AdamBC

Implements DP-AdamBC, differentially private Adam with bias correction of the noised second moment.

Under differential privacy the gradient is privatized by per-example clipping followed by Gaussian noise. That noise inflates the second-moment estimate $v_t$: it accumulates an extra variance term that does not vanish, so the denominator $\sqrt{v_t}$ is dominated by noise and the update collapses toward plain DP-SGD. DP-AdamBC subtracts this known noise variance $\Phi = (\sigma C / B)^2$ from $\hat v_t$ before taking the square root, restoring Adam's adaptive behavior, with a floor $\gamma'$ for numerical stability.

$$
\begin{aligned}
\tilde g_t &= \frac{1}{B}\left(\sum_i \frac{g_{t,i}}{\max\!\left(1, \tfrac{\lVert g_{t,i}\rVert_2}{C}\right)} + z_t\right), \quad z_t \sim \mathcal{N}(0, \sigma^2 C^2 \mathbb{I}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde g_t, \qquad \hat m_t = \frac{m_t}{1-\beta_1^t} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde g_t^2, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\max\!\left(\hat v_t - \left(\tfrac{\sigma C}{B}\right)^2,\ \gamma'\right)}}
\end{aligned}
$$

where $\theta$ are parameters, $\eta$ the learning rate, $g_{t,i}$ the per-example gradients, $C$ the clipping norm, $\sigma$ the noise multiplier, $B$ the batch size, $\beta_1,\beta_2$ the moment decay rates, $\Phi=(\sigma C/B)^2$ the subtracted DP-noise variance, and $\gamma'$ a small stability floor inside the max.

Reference: Qiaoyue Tang, Frederick Shpilevskiy, Mathias Lécuyer, "DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)", AAAI 2024. https://arxiv.org/abs/2312.14334

---
[Back to the Canon](../index.md)
