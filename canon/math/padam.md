# PAdam

Implements PAdam, partially adaptive momentum estimation.

PAdam interpolates between Adam and SGD with momentum by raising the
second-moment denominator to a partial power $p \in (0, 1/2]$. With
$p = 1/2$ the update is Adam; as $p \to 0$ the adaptivity
vanishes and the update approaches plain momentum, which lets PAdam use a
larger base learning rate without the gradient explosion that small
denominators cause.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                   \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                 \\
     \theta_t &= \theta_{t-1} - \eta \, \frac{m_t}{v_t^{\,p}}
\end{aligned}
$$

This implementation applies Adam-style bias correction to the moments and
raises the bias-corrected denominator to the partial power, so the effective
step is $\eta \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)^{2p}$, which
equals $\hat{v}_t^{\,p}$ up to the stabilizing $\epsilon$.

Reference: Jinghui Chen, Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao,
Quanquan Gu, "Closing the Generalization Gap of Adaptive Gradient Methods in
Training Deep Neural Networks", IJCAI 2020.
https://arxiv.org/abs/1806.06763

---
[Back to the Canon](../README.md)
