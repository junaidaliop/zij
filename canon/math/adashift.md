# AdaShift

Implements AdaShift, an adaptive method that decorrelates the gradient
from the second-moment estimate by a temporal shift.

In Adam the gradient $g_t$ enters both the numerator and the
denominator of the update, which biases the effective step size. AdaShift
removes that correlation by accumulating the second moment from a gradient
that is $n$ steps in the past, where $n$ is the window size
$\mathrm{keep\_num}$. A spatial reduction $\phi$ (the maximum
over each parameter block by default) is applied to the shifted squared
gradient so that the denominator is independent of the current gradient.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1}
            + \frac{1}{w} g_t
            - \frac{\beta_1^{n}}{w} g_{t-n}                              \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \phi(g_{t-n}^2)            \\
     \hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t-n}}                        \\
     \theta_t &= \theta_{t-1} - \eta \, \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $w = \sum_{i=0}^{n-1} \beta_1^i$ normalizes the windowed first
moment, $g_{t-n}$ is the oldest (evicted) gradient in the window, $\eta$
is the learning rate, and $\epsilon$ is added for numerical stability.
No update is applied until the window has filled with $n$ gradients.

Reference: Zhiming Zhou, Qingru Zhang, Guansong Lu, Hongwei Wang, Weinan
Zhang, Yong Yu, "AdaShift: Decorrelation and Convergence of Adaptive
Learning Rate Methods", ICLR 2019.
https://arxiv.org/abs/1810.00143

---
[Back to the Canon](../README.md)
