# CfGD / CfAdam

Implements CfGD / CfAdam, gradient descent and Adam driven by a Caputo fractional-based gradient.

The ordinary gradient is replaced by a Caputo fractional gradient that, coordinate-wise, mixes the order-$\alpha$ and order-$(1+\alpha)$ Caputo derivatives taken from a lower/upper integral terminal $c$. By Theorem 2.3 this direction is the steepest-descent direction of a smoothing $c F_{\alpha,\beta}$ of the objective, so the parameters $\alpha,\beta$ act as an implicit regularizer that can mitigate the dependence on the condition number; $\alpha=1,\beta=0$ recovers the ordinary gradient. CfGD plugs this direction into gradient descent; CfAdam is standard Adam with its gradient replaced by the same Caputo fractional-based gradient.

$$
\begin{aligned}
d_t &= \mathrm{diag}\!\left(\,{}^{C}_{c}D_x I(\theta_{t,j})\right)^{-1}\, {}^{C}_{c}\nabla_x f(\theta_t) + \beta \cdot \mathrm{diag}\!\left(|\theta_{t,j}-c_j|\right)\, {}^{C}_{c}\nabla_x^{\,1+\alpha} f(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta\, d_t && \text{(CfGD)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, d_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, d_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} && \text{(CfAdam)}
\end{aligned}
$$

where ${}^{C}_{c}\nabla_x^{\,\alpha} f$ is the Caputo fractional gradient of order $\alpha\in(0,1)$ with per-coordinate integral terminal $c=(c_j)$, ${}^{C}_{c}\nabla_x^{\,1+\alpha} f$ its order-$(1+\alpha)$ counterpart, $d_t$ the resulting Caputo fractional-based gradient (replacing the ordinary $g_t$), $\eta$ the learning rate, $\beta\in\mathbb{R}$ the smoothing weight, $m_t,v_t$ the first/second moments with decays $\beta_1,\beta_2$, and $\epsilon$ a stability constant.

Reference: Yeonjong Shin, Jérôme Darbon, George Em Karniadakis, "Accelerating gradient descent and Adam via fractional gradients", Neural Networks 2023. https://doi.org/10.1016/j.neunet.2023.01.002

---
[Back to the Canon](../README.md)
