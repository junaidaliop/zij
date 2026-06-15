# Rprop

Implements Rprop, resilient backpropagation, which adapts a separate step
size per parameter from the sign of consecutive gradients and ignores their
magnitude.

Each parameter carries its own update value $\Delta_t$. When the gradient
keeps the same sign across two steps, the optimizer is moving consistently
downhill, so $\Delta_t$ grows by a factor $\eta^{+} > 1$. When the gradient
flips sign, the last step overshot a minimum, so $\Delta_t$ shrinks by a
factor $\eta^{-} < 1$ and the update is clamped to the bounds
$[\Delta_{\min}, \Delta_{\max}]$. The parameter then moves by $\Delta_t$ in
the direction opposite the gradient's sign, independent of the gradient's
size.

$$
\begin{aligned}
\Delta_t &=
\begin{cases}
\min(\eta^{+}\, \Delta_{t-1},\ \Delta_{\max}) & \text{if } g_{t-1}\, g_t > 0 \\
\max(\eta^{-}\, \Delta_{t-1},\ \Delta_{\min}) & \text{if } g_{t-1}\, g_t < 0 \\
\Delta_{t-1} & \text{if } g_{t-1}\, g_t = 0
\end{cases} \\
g_t &\leftarrow 0 \quad \text{if } g_{t-1}\, g_t < 0 \\
\theta_t &= \theta_{t-1} - \mathrm{sign}(g_t)\, \Delta_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient, $\Delta_t$ is the
per-parameter step size bounded by $\Delta_{\min}$ and $\Delta_{\max}$, and
$\eta^{+}, \eta^{-}$ are the step increase and decrease factors. On a sign
change the gradient is set to zero so that $\Delta_t$ is not adapted again at
the next step.

Reference: Martin Riedmiller and Heinrich Braun, "A direct adaptive method
for faster backpropagation learning: the RPROP algorithm", ICNN 1993.
https://doi.org/10.1109/ICNN.1993.298623

---
[Back to the Canon](../README.md)
