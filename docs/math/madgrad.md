# MADGRAD

Implements MADGRAD, a momentumized, adaptive, dual averaged gradient method.


$$
\begin{aligned}
   \lambda_k &= \gamma \sqrt{k + 1} \\
   s_{k+1} &= s_k + \lambda_k g_k \\
   \nu_{k+1} &= \nu_k + \lambda_k\, g_k \odot g_k \\
   z_{k+1} &= \theta_0 - \frac{s_{k+1}}{\sqrt[3]{\nu_{k+1}} + \epsilon} \\
   \theta_{k+1} &= (1 - c)\, \theta_k + c\, z_{k+1}
\end{aligned}
$$

where $\gamma$ is `lr`, $\theta_0$ is the initial point,
$c = 1 - \text{momentum}$, and the denominator uses a cube root of the
accumulated squared gradients. With `momentum` set to zero the iterate
reduces to the dual averaging point $z_{k+1}$.


**Note:** `lr` is not comparable to SGD or Adam and should be set by a sweep.

MADGRAD usually needs less weight decay than other methods, often zero. On
sparse problems both `weight_decay` and `momentum` should be set to zero.

Reference: Aaron Defazio and Samy Jelassi, "Adaptivity without Compromise:
A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
Optimization", Journal of Machine Learning Research, 23(144):1-34, 2022
(preprint arXiv:2101.11075).
https://arxiv.org/abs/2101.11075

---
[Back to the Canon](../index.md)
