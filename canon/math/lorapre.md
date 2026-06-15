# LoRA-Pre

Implements LoRA-Pre, a memory-efficient Adam variant that stores momentum as a low-rank product.

The key observation is that an exponential moving average of the gradient is equivalent to fitting an online linear regressor, so the momentum matrix can be carried in a compact factorized form $m_B m_A$ instead of storing the dense moment. Each step updates the low-rank factors with one regression step, reconstructs the moment, and then applies a standard Adam update with bias correction and decoupled weight decay. The same factorization is applied to the (elementwise-squared) second moment $v_B v_A$.

$$
\begin{aligned}
m_{B,t} &\leftarrow (1-\gamma_1)\, m_{B,t-1} + \gamma_1\, g_t\, m_{A,t-1}^{\top}\big(m_{A,t-1} m_{A,t-1}^{\top}\big)^{-1}, \\
m_{A,t} &\leftarrow (1-\gamma_1)\, m_{A,t-1} + \gamma_1\big(m_{B,t-1}^{\top} m_{B,t-1}\big)^{-1} m_{B,t-1}^{\top}\, g_t, \\
v_{B,t} &\leftarrow (1-\gamma_2)\, v_{B,t-1} + \gamma_2\, |g_t|\, v_{A,t-1}^{\top}\big(v_{A,t-1} v_{A,t-1}^{\top}\big)^{-1}, \\
v_{A,t} &\leftarrow (1-\gamma_2)\, v_{A,t-1} + \gamma_2\big(v_{B,t-1}^{\top} v_{B,t-1}\big)^{-1} v_{B,t-1}^{\top}\, |g_t|, \\
m_t &\leftarrow \beta_1\, m_{B,t} m_{A,t} + (1-\beta_1)\, g_t, \\
v_t &\leftarrow \beta_2\, (v_{B,t} v_{A,t})^{\odot 2} + (1-\beta_2)\, g_t^{\odot 2}, \\
\hat{m}_t &\leftarrow m_t / (1-\beta_1^{t}), \qquad \hat{v}_t \leftarrow v_t / (1-\beta_2^{t}), \\
\theta_t &\leftarrow \theta_{t-1} - \gamma\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\, \theta_{t-1}\right).
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $g_t$ the gradient, $m_{B},m_{A}$ and $v_{B},v_{A}$ the rank-$r$ factors of the first and second moments, $\beta_1,\beta_2$ the Adam decay rates, $\gamma_1,\gamma_2$ the regressor (EMA) rates chosen so that $1-\gamma_1=\sqrt{\beta_1}$ and $1-\gamma_2=\beta_2^{1/4}$, $\lambda$ the weight decay, $\epsilon$ a stability constant, and $\odot 2$ elementwise squaring.

Reference: Zhengbo Wang, Jian Liang, Ran He, Zilei Wang, Tieniu Tan, "Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation", arXiv 2025. https://arxiv.org/abs/2602.24283

---
[Back to the Canon](../README.md)
