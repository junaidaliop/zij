# Momo

Implements MoMo, SGD with momentum and an adaptive Polyak step size.


$$
\begin{aligned}
\bar{f}_t &= \beta \bar{f}_{t-1} + (1 - \beta) f_t \\
\gamma_t &= \beta \gamma_{t-1} + (1 - \beta) \langle g_t, \theta_t \rangle \\
m_t &= \beta m_{t-1} + (1 - \beta) g_t \\
h_t &= \bar{f}_t + \langle m_t, \theta_t \rangle - \gamma_t \\
\theta_{t+1} &= \theta_t - \min\left\{ \eta,
    \frac{(h_t - f_*)_+}{\lVert m_t \rVert^2} \right\} m_t
\end{aligned}
$$

where $f_t$ is the loss, $f_*$ is the lower bound `lb` on
the loss, and `lr` sets the cap $\eta$ on the adaptive step size.
With `bias_correction=True` the averages start at zero and $f_*$
and $\eta$ are rescaled by $\rho_t = 1 - \beta^t$; with
`weight_decay` $\lambda > 0$ the update ends with a proximal
division by $1 + \eta\lambda$. `use_fstar=True` estimates the
lower bound online instead of keeping it fixed.


**Note:** `step` needs the current loss value: pass either a closure or, if the backward pass already ran, the loss tensor through `loss`.

Reference: Fabian Schaipp, Ruben Ohana, Michael Eickenberg,
Aaron Defazio, Robert M. Gower,
"MoMo: Momentum Models for Adaptive Learning Rates", ICML 2024.
https://arxiv.org/abs/2305.07583

---
[Back to the Canon](../index.md)
