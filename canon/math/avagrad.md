# AvaGrad

Implements AvaGrad, an adaptive method that decouples the learning rate
from adaptability.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                           \\
     \eta_t &= \frac{1}{\sqrt{v_{t-1}} + \epsilon}                        \\
     \gamma_t &= \frac{\sqrt{d}}{\lVert \eta_t \rVert_2}                  \\
     \theta_t &= \theta_{t-1} - \alpha \gamma_t \, \eta_t \odot m_t       \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}
$$

The per-parameter adaptive rate $\eta_t$ depends on the second moment
from the previous step, and the global scalar $\gamma_t$ normalizes it
by its root-mean-square over the $d$ parameters in the group. This
normalization cancels the dependence of the update on the scale of the
second moment, so the learning rate $\alpha$ and the adaptability
$\epsilon$ can be tuned independently.


**Note:** following the official implementation, the second moment is Adam-style

bias-corrected before use (absent from the paper's Algorithm 2): the update
uses $\hat{v}_{t-1} = v_{t-1} / (1 - \beta_2^{t-1})$ and the
$\gamma_t$ normalization uses the current-step debias
$1 - \beta_2^{t}$.

Reference: Pedro Savarese, David McAllester, Sudarshan Babu, Michael Maire,
"Domain-Independent Dominance of Adaptive Methods", CVPR 2021.
https://arxiv.org/abs/1912.01823

---
[Back to the Canon](../README.md)
