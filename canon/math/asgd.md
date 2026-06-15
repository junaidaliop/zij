# ASGD

Implements ASGD, stochastic gradient descent with Polyak-Ruppert
averaging of the iterates.

ASGD runs a plain SGD recursion with a decaying step size and, in parallel,
maintains a running average $a_t$ of the parameter iterates. Once the step
count passes the threshold $t_0$, the averaging weight $\mu_t$ begins to
shrink so that $a_t$ converges to the mean of the trajectory; this averaged
estimate, rather than the last iterate $\theta_t$, is the accelerated
solution. The step size $\eta_t$ decays as a power of the step count.

$$
\begin{aligned}
\eta_t &= \frac{\eta}{(1 + \lambda \eta\, t)^{\alpha}} \\
\theta_t &= (1 - \lambda \eta_t)\, \theta_{t-1} - \eta_t \big(g_t + \lambda \theta_{t-1}\big) \\
\mu_t &= \frac{1}{\max(1,\ t - t_0)} \\
a_t &= a_{t-1} + \mu_t \big(\theta_t - a_{t-1}\big)
\end{aligned}
$$

where $\theta$ are the parameters, $a_t$ is the averaged iterate, $\eta$
is the base learning rate, $\eta_t$ is the decayed step size, $g_t$ is the
gradient, $\lambda$ is the decay term, $\alpha$ is the power governing the
step-size decay, $t_0$ is the step at which averaging begins, and $\mu_t$
is the averaging weight.

Reference: B. T. Polyak and A. B. Juditsky, "Acceleration of Stochastic Approximation by Averaging", SIAM Journal on Control and Optimization 1992.
https://doi.org/10.1137/0330046

---
[Back to the Canon](../README.md)
