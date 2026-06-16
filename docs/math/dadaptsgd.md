# DAdaptSGD

Implements SGD with D-Adaptation automatic step sizes.


$$
\begin{aligned}
   \lambda_t &= \frac{d_t \gamma}{\lVert g_0 \rVert} \\
   s_{t+1} &= s_t + \lambda_t g_t \\
   z_{t+1} &= z_t - \lambda_t g_t \\
   \theta_{t+1} &= \beta \theta_t + (1 - \beta) z_{t+1} \\
   \hat{d}_{t+1} &= \frac{2 \sum_{i=0}^{t} \lambda_i
       \langle g_i, s_i \rangle}{\lVert s_{t+1} \rVert} \\
   d_{t+1} &= \max(d_t, \hat{d}_{t+1})
\end{aligned}
$$

where $\gamma$ is `lr` and $\beta$ is `momentum`.


**Note:** `lr` rescales the D-adapted step size and should normally stay

at its default of 1.0.

Reference: Aaron Defazio and Konstantin Mishchenko,
"Learning-Rate-Free Learning by D-Adaptation", ICML 2023.
https://arxiv.org/abs/2301.07733

---
[Back to the Canon](../index.md)
