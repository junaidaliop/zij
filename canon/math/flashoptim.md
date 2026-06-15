# FlashOptim

Implements FlashOptim, a memory-efficient wrapper that runs a standard optimizer step on dequantized state and recompresses it afterward.

FlashOptim keeps the optimizer math unchanged (it ships drop-in variants of SGD, AdamW, and Lion) and instead shrinks what is stored between steps. Each step reconstructs the master weights and moment estimates from compressed form, performs the ordinary update, then re-compresses. Two pieces do the heavy lifting: a master-weight split that stores $\theta$ as a BF16 value $\theta'$ plus an INT8 residual $\rho$ encoding the deviation within the unit-in-the-last-place interval, and companding transforms $\varphi_m,\varphi_v$ that reshape the heavy-tailed moment distributions before 8-bit quantization. Together with 16-bit gradients this cuts AdamW state from 16 to 7 bytes per parameter.

The FlashAdamW step, with $Q_m,Q_v$ the (de)quantizers for the moments and $C$ the master-weight split, is:

$$
\begin{aligned}
m_{t-1} &= Q_m^{-1}(m_{t-1}^q, m_{t-1}^s), \quad v_{t-1} = Q_v^{-1}(v_{t-1}^q, v_{t-1}^s), \quad \theta_{t-1} = C^{-1}(\theta'_{t-1}, \rho_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) \\
(m_t^q, m_t^s) &= Q_m(m_t), \quad (v_t^q, v_t^s) = Q_v(v_t), \quad (\theta'_t, \rho_t) = C(\theta_t)
\end{aligned}
$$

where the master-weight split sets $\theta' = \mathrm{downcast}(\theta)$ (BF16) and $\rho = \mathrm{Int}\big(\mathrm{round}(\mathrm{clip}(e_{\mathrm{norm}}, -1, 1)\cdot 127)\big)$ from the normalized rounding error $e = \theta - \theta'$, reconstructed as $\hat{\theta} = \theta' + \frac{\rho}{127}\cdot\frac{\mathrm{ULP}(\theta')}{2}$; the moment companders are $\varphi_m(x) = \frac{2x}{1+|x|}$ (INT8, inverse $\frac{z}{2-|z|}$) and $\varphi_v(x) = \sqrt{x}$ (UINT8, inverse $z^2$), applied per group with absmax scales. Here $\theta$ are parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t/v_t$ the first/second moments, $\beta_1,\beta_2$ the decay rates, $\lambda$ the decoupled weight decay, $\epsilon$ the stability constant, and $\mathrm{ULP}$ the unit in the last place of $\theta'$.

Reference: Jose Javier Gonzalez Ortiz, Abhay Gupta, Chris Renard, Davis Blalock, "FlashOptim: Optimizers for Memory Efficient Training", arXiv preprint 2026. https://arxiv.org/abs/2602.23349

---
[Back to the Canon](../README.md)
