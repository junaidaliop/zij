# MomoAdam

Implements MoMo-Adam, Adam with an adaptive Polyak step size.


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t \odot g_t \\
D_t &= \mathrm{Diag}\left(\sqrt{v_t / (1 - \beta_2^t)} + \epsilon\right) \\
\bar{f}_t &= \beta_1 \bar{f}_{t-1} + (1 - \beta_1) f_t \\
\gamma_t &= \beta_1 \gamma_{t-1} + (1 - \beta_1) \langle g_t, \theta_t \rangle \\
\tau_t &= \min\left\{ \frac{\eta}{1 - \beta_1^t},
    \frac{\left((1 + \eta\lambda)\left(\bar{f}_t - \gamma_t
    - (1 - \beta_1^t) f_*\right)
    + \langle m_t, \theta_t \rangle\right)_+}
    {\lVert m_t \rVert^2_{D_t^{-1}}} \right\} \\
\theta_{t+1} &= \frac{1}{1 + \eta\lambda}
    \left(\theta_t - \tau_t D_t^{-1} m_t\right)
\end{aligned}
$$

where $f_t$ is the loss, $f_*$ is the lower bound `lb` on
the loss, $\eta$ is `lr`, and $\lambda$ is `weight_decay`.
`divide=False` replaces the proximal division by the AdamW-style decay
$\theta_t \leftarrow (1 - \eta\lambda)\,\theta_t$ before the step;
`use_fstar=True` estimates the lower bound online.


**Note:** `step` needs the current loss value: pass either a closure or, if the backward pass already ran, the loss tensor through `loss`.

Reference: Fabian Schaipp, Ruben Ohana, Michael Eickenberg,
Aaron Defazio, Robert M. Gower,
"MoMo: Momentum Models for Adaptive Learning Rates", ICML 2024.
https://arxiv.org/abs/2305.07583

---
[Back to the Canon](../README.md)
