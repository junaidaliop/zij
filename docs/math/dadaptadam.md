# DAdaptAdam

Implements Adam with D-Adaptation automatic step sizes.


$$
\begin{aligned}
   m_{t+1} &= \beta_1 m_t + (1 - \beta_1)\, d_t \gamma\, g_t \\
   v_{t+1} &= \beta_2 v_t + (1 - \beta_2)\, g_t^2 \\
   A_{t+1} &= \mathrm{diag}\bigl(\sqrt{v_{t+1}} + \epsilon\bigr) \\
   \theta_{t+1} &= \theta_t - A_{t+1}^{-1} m_{t+1} \\
   s_{t+1} &= \sqrt{\beta_2}\, s_t + (1 - \sqrt{\beta_2})\, d_t \gamma\, g_t \\
   r_{t+1} &= \sqrt{\beta_2}\, r_t + (1 - \sqrt{\beta_2})\, d_t \gamma\,
       \langle g_t, s_t \rangle_{A_t^{-1}} \\
   \hat{d}_{t+1} &= \frac{r_{t+1}}{(1 - \sqrt{\beta_2})\,
       \lVert s_{t+1} \rVert_1} \\
   d_{t+1} &= \max(d_t, \hat{d}_{t+1})
\end{aligned}
$$

where $\gamma$ is `lr`. Following the official implementation,
the $r$ recursion weights the inner product with the pre-update
moment matrix $A_t$ rather than the $A_{t+1}$ written in
Algorithm 4 of the paper; the parameter step uses $A_{t+1}$.


**Note:** `lr` rescales the D-adapted step size and should normally stay

at its default of 1.0. To scale the learning rate differently for each
layer, set the `layer_scale` value of the parameter group instead.

Reference: Aaron Defazio and Konstantin Mishchenko,
"Learning-Rate-Free Learning by D-Adaptation", ICML 2023.
https://arxiv.org/abs/2301.07733

---
[Back to the Canon](../index.md)
