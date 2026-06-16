# R-AdaZO

Implements R-AdaZO (Refined Adaptive Zeroth-Order Optimization), an adaptive zeroth-order method that refines Adam-style moment estimates from random-perturbation gradients.

R-AdaZO estimates the gradient with a finite-difference, random-direction scheme that queries only function values, then feeds that estimate into an Adam-like update. Its refinement is to drive the second-moment accumulator with the squared first moment $m_t^2$ rather than the squared raw estimate $g_t^2$. Because the momentum buffer $m_t$ has lower variance than the noisy single-step estimate, this yields a more reliable adaptive preconditioner and sharper coordinate-wise scaling.

$$
\begin{aligned}
g_t &= \frac{d}{K}\sum_{k=1}^{K}\frac{f(\theta_{t-1}+\mu u_k;\xi_t)-f(\theta_{t-1};\xi_t)}{\mu}\,u_k, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,m_t^2, \\
\theta_t &= \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t + \zeta}},
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the zeroth-order gradient estimate built from $K$ directions $u_k$ drawn uniformly from the unit sphere with smoothing radius $\mu>0$, $d$ is the parameter dimension, $\xi_t$ is the sampled mini-batch, $m_t$ and $v_t$ are the first and refined second moments, $\beta_1,\beta_2$ are the decay rates, and $\zeta$ is a small constant for numerical stability.

Reference: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai, "Refining Adaptive Zeroth-Order Optimization at Ease", ICML 2025. https://arxiv.org/abs/2502.01014

---
[Back to the Canon](../index.md)
