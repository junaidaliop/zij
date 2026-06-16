# ALI-G

Implements ALI-G, an adaptive learning-rate method that exploits the interpolation property of over-parameterized models.

ALI-G assumes the model can drive the training loss to (near) zero, so each step sets its size from the current loss value and gradient norm rather than a hand-tuned schedule. The step-size is the loss divided by the squared gradient norm (a Polyak-style step), capped by a single maximal learning rate $\eta$. With $\eta=\infty$ the cap is dropped entirely and the method has no learning-rate hyperparameter. An optional Nesterov momentum term may be applied to the resulting step.

$$
\begin{aligned}
\gamma_t &= \min\!\left\{ \frac{\ell_t(\theta_t)}{\lVert g_t \rVert^2 + \epsilon},\; \eta \right\} \\
\theta_{t+1} &= \theta_t - \gamma_t\, g_t
\end{aligned}
$$

where $\ell_t(\theta_t)$ is the (regularized) loss on the current minibatch, $g_t = \nabla \ell_t(\theta_t)$ its gradient, $\gamma_t$ the adaptive step-size, $\eta$ the maximal learning rate, and $\epsilon$ a small constant for numerical stability. Optional Nesterov momentum replaces the update with $v_t = \mu v_{t-1} - \gamma_t g_t$ and $\theta_{t+1} = \theta_t + \mu v_t$.

Reference: Leonard Berrada, Andrew Zisserman, M. Pawan Kumar, "Training Neural Networks for and by Interpolation", ICML 2020. https://arxiv.org/abs/1906.05661

---
[Back to the Canon](../index.md)
