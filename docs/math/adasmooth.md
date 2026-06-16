# AdaSmooth

Implements AdaSmooth, an adaptive learning rate method based on the
effective ratio.

AdaSmooth replaces the fixed decay of the squared-gradient running average
with a per-parameter smoothing constant derived from the effective ratio of
the recent parameter trajectory. The effective ratio measures how directed
the movement has been: it is the magnitude of the accumulated change divided
by the accumulated absolute change. A directed trajectory (ratio near one)
yields a short averaging window, which speeds up the descent, while a
zigzagging trajectory (ratio near zero) yields a long window, which slows
the descent near a minimum.


$$
\begin{aligned}
     s_t &= s_{t-1} + (\theta_t - \theta_{t-1})                          \\
     n_t &= n_{t-1} + |\theta_t - \theta_{t-1}|                          \\
     e_t &= \frac{\left| \sum s_t \right|}{\sum n_t}                     \\
     c_t &= (\rho_2 - \rho_1)\, e_t + (1 - \rho_2)                       \\
     v_t &= c_t^2\, g_t^2 + (1 - c_t^2)\, v_{t-1}                        \\
     \theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient,
$s_t$ and $n_t$ accumulate the signed and absolute parameter
changes, $e_t$ is the effective ratio, $c_t$ is the scaled
smoothing constant built from the fast and slow decay rates
$\rho_1, \rho_2$ (passed as `betas`), $v_t$ is the running
average of the squared gradient, $\eta$ is the learning rate, and
$\epsilon$ guards the denominator.

Reference: Jun Lu, "AdaSmooth: An Adaptive Learning Rate Method based on
Effective Ratio", arXiv 2022.
https://arxiv.org/abs/2204.00825

---
[Back to the Canon](../index.md)
