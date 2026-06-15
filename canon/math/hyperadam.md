# HyperAdam

Implements HyperAdam, a learned task-adaptive optimizer that combines a field of Adam-style candidate updates with data-driven decay rates and weights.

HyperAdam casts each step as a weighted combination of $J$ Adam-like candidate directions. Two recurrent cells generate, per coordinate, adaptive decay-rate vectors $\beta_t$ and $\gamma_t$ (one entry per candidate), and a separate cell produces combination weights $\rho_t$. Each candidate forms its own bias-corrected moment estimate, and the final search direction is the weighted sum of these candidates. The weights use an ELU activation rather than a softmax, so they need not be positive and the candidates do not form a convex combination: this lets the optimizer down-weight or reverse candidate directions that point poorly.

$$
\begin{aligned}
\beta_t &= \sigma\!\big([m'_{t-1}, s_t]\,\theta_u + b_u\big), \qquad
\gamma_t = \sigma\!\big([m'_{t-1}, s_t]\,\theta_r + b_r\big), \\
m_t^{j} &= \beta_t^{j}\, m_{t-1}^{j} + (1-\beta_t^{j})\, g_t, \qquad
v_t^{j} = \gamma_t^{j}\, v_{t-1}^{j} + (1-\gamma_t^{j})\, g_t^{2}, \\
\hat m_t^{j} &= \frac{m_t^{j}/\hat\beta_t^{j}}{\sqrt{v_t^{j}/\hat\gamma_t^{j}} + \epsilon}, \\
\rho_t &= \mathrm{ELU}\!\big(s_t\,\theta_q + b_q\big), \\
d_t &= \sum_{j=1}^{J} \rho_t^{j} \odot \hat m_t^{j}, \\
\theta_t &= \theta_{t-1} - \alpha\, d_t .
\end{aligned}
$$

where $\theta$ are the optimizee parameters, $\alpha$ the step size, $g_t$ the gradient, $m_t^{j}/v_t^{j}$ the first/second moment of candidate $j$ with adaptive decays $\beta_t^{j},\gamma_t^{j}$, $\hat\beta_t^{j},\hat\gamma_t^{j}$ the running bias-correction factors, $\hat m_t^{j}$ the bias-corrected candidate update, $\rho_t^{j}$ its combination weight, $s_t$ the LSTM task state, $m'_{t-1}$ the $\ell_2$-normalized candidate moments, $\theta_u,\theta_r,\theta_q,b_u,b_r,b_q$ the learned meta-parameters, $\odot$ elementwise product, and $\epsilon$ a stability constant.

Reference: Shipeng Wang, Jian Sun, Zongben Xu, "HyperAdam: A Learnable Task-Adaptive Adam for Network Training", AAAI 2019. https://arxiv.org/abs/1811.08996

---
[Back to the Canon](../README.md)
