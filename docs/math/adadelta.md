# Adadelta

Implements Adadelta, an adaptive learning rate method that needs no
manually tuned global rate.

Adadelta extends Adagrad by accumulating the squared gradients in an
exponentially decaying running average rather than summing them over all
time, which keeps the denominator from growing without bound. It also
maintains a decaying average of the squared parameter updates, so the
effective step is the ratio of the root mean square of past updates to the
root mean square of recent gradients. This ratio matches the units of the
parameter and removes the need to choose a learning rate.

$$
\begin{aligned}
E[g^2]_t &= \rho\, E[g^2]_{t-1} + (1 - \rho)\, g_t^2 \\
\Delta\theta_t &= -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}}\, g_t \\
E[\Delta\theta^2]_t &= \rho\, E[\Delta\theta^2]_{t-1} + (1 - \rho)\, \Delta\theta_t^2 \\
\theta_t &= \theta_{t-1} + \Delta\theta_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient, $\rho$ is the
decay rate of the running averages, $E[g^2]_t$ and $E[\Delta\theta^2]_t$
are the decaying averages of the squared gradients and squared updates, and
$\epsilon$ is a small constant for numerical stability.

Reference: Matthew D. Zeiler, "ADADELTA: An Adaptive Learning Rate Method", arXiv 2012.
https://arxiv.org/abs/1212.5701

---
[Back to the Canon](../index.md)
