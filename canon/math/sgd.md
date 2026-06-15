# SGD

Implements SGD, stochastic gradient descent with optional momentum,
dampening, weight decay, and Nesterov acceleration.

Plain SGD steps the parameters along the negative gradient. With momentum,
a velocity buffer $b_t$ accumulates an exponentially decayed running sum of
the gradients, smoothing the trajectory and damping oscillations. Dampening
$\tau$ scales the contribution of the current gradient to that buffer.
Nesterov momentum looks ahead by adding the freshly updated buffer back into
the gradient before stepping, so the velocity is evaluated at the anticipated
position rather than the current one. Weight decay $\lambda$ adds an $L_2$
penalty by shifting the gradient toward the origin.

$$
\begin{aligned}
g_t &\leftarrow \nabla_\theta f_t(\theta_{t-1}) + \lambda\, \theta_{t-1} \\
b_t &= \mu\, b_{t-1} + (1 - \tau)\, g_t \\
g_t &\leftarrow g_t + \mu\, b_t \quad (\text{Nesterov}) \qquad
g_t \leftarrow b_t \quad (\text{otherwise}) \\
\theta_t &= \theta_{t-1} - \gamma\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ is the learning rate, $g_t$ is
the gradient, $b_t$ is the momentum buffer, $\mu$ is the momentum factor,
$\tau$ is the dampening, and $\lambda$ is the weight decay. The momentum
buffer is active only when $\mu \neq 0$, and dampening applies from the
second step onward.

Reference: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton, "On the importance of initialization and momentum in deep learning", ICML 2013.
https://proceedings.mlr.press/v28/sutskever13.html

---
[Back to the Canon](../README.md)
