# EmoNavi

Implements EmoNavi, an Adam-style optimizer whose step size is modulated by an "emotion" signal read from the training loss.

EmoNavi tracks the loss with three exponential moving averages at short, medium, and long timescales. The normalized gap between the long and short averages is squashed through $\tanh$ into a bounded scalar $s_t \in (-1,1)$ that measures how fast the loss is moving relative to its slow trend. From this scalar it forms a signed trust value and an emoDrive multiplier that amplifies the learning rate when the loss is improving steadily and clamps it during sharp swings, leaving the rest of the update as plain Adam with decoupled weight decay.

$$
\begin{aligned}
e^{(s)}_t &= 0.3\,\ell_t + 0.7\,e^{(s)}_{t-1}, \quad
e^{(m)}_t = 0.05\,\ell_t + 0.95\,e^{(m)}_{t-1}, \quad
e^{(l)}_t = 0.01\,\ell_t + 0.99\,e^{(l)}_{t-1} \\
s_t &= \tanh\!\left(\frac{e^{(l)}_t - e^{(s)}_t}{\max(e^{(l)}_t,\,10^{-5})}\right), \qquad
\tau_t = \mathrm{sign}(s_t)\,(1-|s_t|) \\
d_t &= \begin{cases}
8\,|\tau_t|\,(1 + 0.1\,\tau_t) & 0.25 < |s_t| < 0.5 \\
1 - |s_t| & |s_t| > 0.75 \\
1 & \text{otherwise}
\end{cases} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \\
\theta_t &= (1 - \gamma\lambda)\,\theta_{t-1} - \gamma\,d_t\,\frac{m_t}{\sqrt{v_t}+\epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $g_t$ the gradient, $\ell_t$ the current loss value, $e^{(s)},e^{(m)},e^{(l)}$ the short/medium/long loss EMAs, $s_t$ the emotion scalar, $\tau_t$ the trust value, $d_t$ the emoDrive multiplier, $m_t,v_t$ the first and second gradient moments with decays $\beta_1,\beta_2$, $\lambda$ the weight decay, and $\epsilon$ the stability constant.

Reference: muooon, "EmoNavi: An emotion-driven optimizer that feels loss and navigates accordingly", 2025. https://github.com/muooon/EmoNavi

---
[Back to the Canon](../index.md)
