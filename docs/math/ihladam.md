# IHL-Adam

Implements IHL-Adam, an Adam variant whose gradient is taken with an improved Hausdorff-like (fractal-order) derivative.

The Hausdorff derivative of fractal order $\alpha$ measures change against the fractal scale $\theta^{\alpha}$ rather than $\theta$, and for a smooth loss reduces to a power-law rescaling of the ordinary gradient, $\partial J / \partial \theta^{\alpha} = \tfrac{1}{\alpha}\,\theta^{1-\alpha}\,\partial J/\partial \theta$. Unlike the Caputo or Riemann-Liouville fractional derivatives it carries no integral memory term, so it stays local and cheap. The improved Hausdorff-like (IHL) derivative folds the cost function and a per-stage order into this rescaling, and the resulting gradient $g_t^{(\alpha)}$ is fed into the standard adaptive moment estimation machinery in place of the integer-order gradient.

The order $\alpha$ is tuned by judging the size of the cost during training: a larger order early on (when the error is large) speeds convergence, and it is reduced as the loss falls. With $g_t = \nabla_\theta J(\theta_t)$ the per-step update is

$$
\begin{aligned}
g_t^{(\alpha)} &= \frac{1}{\alpha}\,\theta_t^{\,1-\alpha}\, g_t \\
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t^{(\alpha)} \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, \big(g_t^{(\alpha)}\big)^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t - \gamma\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the network parameters, $\gamma$ the learning rate, $g_t$ the integer-order gradient of the cost $J$, $g_t^{(\alpha)}$ the improved Hausdorff-like gradient of fractal order $\alpha \in (0,1]$ (adapted from the cost magnitude, recovering ordinary Adam as $\alpha \to 1$), $m_t$/$v_t$ the first and second moments with bias corrections $\hat{m}_t$/$\hat{v}_t$, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ the stability constant.

Reference: Kai Jia, Zhe Gao, Shasha Xiao, "Parameter training method for convolutional neural networks based on improved Hausdorff-like derivative", Expert Systems with Applications 2024. https://doi.org/10.1016/j.eswa.2023.121659

---
[Back to the Canon](../index.md)
