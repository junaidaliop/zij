# Prodigy

Implements Prodigy, an Adam variant that estimates its own step size online.


$$
\begin{aligned}
m_{t+1} &= \beta_1 m_t + (1 - \beta_1)\, d_t g_t \\
v_{t+1} &= \beta_2 v_t + (1 - \beta_2)\, d_t^2 g_t^2 \\
r_{t+1} &= \beta_3\, r_t + \gamma_t d_t^2
           \langle g_t, \theta_0 - \theta_t \rangle \\
s_{t+1} &= \beta_3\, s_t + \gamma_t d_t^2 g_t \\
\hat{d}_{t+1} &= \frac{r_{t+1}}{\|s_{t+1}\|_1}, \qquad
d_{t+1} = \max(d_t, \hat{d}_{t+1}) \\
\theta_{t+1} &= \theta_t - \gamma_t d_t\,
                \frac{m_{t+1}}{\sqrt{v_{t+1}} + d_t \epsilon}
\end{aligned}
$$

where $d_t$ estimates the distance from $\theta_0$ to the
solution and $\gamma_t$ is the learning rate, acting only as a
multiplier on the estimated step size. The decay rate $\beta_3$ of
$r_t$ and $s_t$ defaults to $\sqrt{\beta_2}$ and can be
overridden through `beta3`. The newly added terms in $r_{t+1}$ and
$s_{t+1}$ are accumulated without the $(1 - \beta_3)$
normalization because the constant cancels in
$\hat{d}_{t+1} = r_{t+1} / \|s_{t+1}\|_1$.


**Note:** Leave `lr` at its default of 1.0. To tune the method, change `d_coef`, which multiplies the estimate $\hat{d}_{t+1}$.

Reference: Konstantin Mishchenko, Aaron Defazio,
"Prodigy: An Expeditiously Adaptive Parameter-Free Learner", ICML 2024.
https://arxiv.org/abs/2306.06101

---
[Back to the Canon](../README.md)
