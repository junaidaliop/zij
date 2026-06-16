# RMSprop

Implements RMSprop, which divides the gradient by a running average of
its recent squared magnitude.

RMSprop keeps a per-coordinate running average $v_t$ of the squared
gradient, decayed by $\alpha$, and scales each step by the inverse of its
square root. Coordinates with persistently large gradients are damped while
small, steady gradients are amplified, giving an adaptive per-parameter
learning rate without the monotonically shrinking step of Adagrad.

$$
\begin{aligned}
v_t &= \alpha\, v_{t-1} + (1 - \alpha)\, g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta\, g_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is
the gradient, $v_t$ is the running average of the squared gradient,
$\alpha$ is the decay rate, and $\epsilon$ is a small constant for
numerical stability.

Reference: Tijmen Tieleman and Geoffrey Hinton, "Lecture 6.5-rmsprop: Divide
the gradient by a running average of its recent magnitude", Coursera:
Neural Networks for Machine Learning, 2012.
https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

---
[Back to the Canon](../index.md)
