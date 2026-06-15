# IAdaPID-ADG

Implements IAdaPID-ADG, an improved adaptive PID optimizer combining AMSGrad and DiffGrad.

IAdaPID-ADG casts optimization as PID control: the integral term $I_t$ accumulates past gradients while the derivative term $D_t$ tracks gradient changes, with no explicit proportional term. The "ADG" component (AMSDiffGrad) fixes Adam-style non-convergence by tracking running maxima of the second moments (AMSGrad), and stabilizes steps through a DiffGrad sigmoid factor $\mu_t$ that shrinks the effective step when consecutive gradients differ sharply.

$$
\begin{aligned}
\Delta g_t &= g_t - g_{t-1}, \qquad \mu_t = \frac{1}{1 + e^{-|\Delta g_t|}} \\
I_t &= \gamma I_{t-1} + g_t, \qquad D_t = \gamma D_{t-1} + (1-\gamma)\,\Delta g_t \\
v_t &= \beta v_{t-1} + (1-\beta) g_t^2, \qquad d_t = \beta d_{t-1} + (1-\beta)(\Delta g_t)^2 \\
v_t^{\max} &= \max(v_{t-1}^{\max}, v_t), \qquad d_t^{\max} = \max(d_{t-1}^{\max}, d_t) \\
\hat{v}_t^{\max} &= \frac{v_t^{\max}}{1-\beta^t}, \qquad \hat{d}_t^{\max} = \frac{d_t^{\max}}{1-\beta^t} \\
\theta_t &= \theta_{t-1} - \eta\,\mu_t \left( \frac{K_i\, I_t}{\sqrt{\hat{v}_t^{\max}} + \epsilon} + \frac{K_d\, D_t}{\sqrt{\hat{d}_t^{\max}} + \epsilon} \right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $\Delta g_t$ the gradient difference, $\mu_t$ the DiffGrad modulation factor, $I_t$ and $D_t$ the integral and derivative terms, $v_t$ and $d_t$ second-moment estimates of $g_t$ and $\Delta g_t$, $v_t^{\max}$ and $d_t^{\max}$ their running maxima, $\gamma$ and $\beta$ decay rates, $K_i$ and $K_d$ the integral and derivative gains, and $\epsilon$ a stability constant.

Reference: Saurabh Saini, Kapil Ahuja, Thomas Wick, Saurav Kumar, "An Improved Adaptive PID Optimizer with Enhanced Convergence and Stability for Deep Learning", arXiv 2026. https://arxiv.org/abs/2605.21968

---
[Back to the Canon](../README.md)
