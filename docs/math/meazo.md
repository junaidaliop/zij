# MEAZO

Implements MEAZO, a memory-efficient adaptive zeroth-order optimizer that tracks a single scalar for global step-size adaptation.

MEAZO fine-tunes models without backpropagation by estimating gradients from forward passes alone: along each random direction $u_i$ it forms the projected gradient (a central finite-difference directional derivative) and reconstructs a rank-$q$ gradient estimate $\hat\nabla f_\varepsilon^q$. Unlike per-coordinate adaptive methods, which would double the memory footprint of a zeroth-order method, MEAZO keeps only a single scalar second moment: it tracks an exponential moving average of the squared mean projected gradient and uses its bias-corrected root as a global learning-rate scaling. This adds essentially no memory over plain zeroth-order SGD while recovering Adam-like step-size adaptivity.

$$
\begin{aligned}
\Delta f_\varepsilon(\theta; u_i) &= \frac{f(\theta + \varepsilon u_i) - f(\theta - \varepsilon u_i)}{2\varepsilon}, \qquad u_i \sim P \\
g_t &= \frac{1}{q} \sum_{i=1}^{q} \Delta f_\varepsilon(\theta_t; u_i) \\
v_t &= \beta\, v_{t-1} + (1 - \beta)\, g_t^2 \\
\hat v_t &= \frac{v_t}{1 - \beta^{\,t-1}} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat v_t} + \zeta} \left( \frac{1}{q} \sum_{i=1}^{q} \Delta f_\varepsilon(\theta_t; u_i)\, u_i \right)
\end{aligned}
$$

where $\theta$ are the parameters, $f$ is the (stochastic) objective, $u_i$ are i.i.d. random directions drawn from $P$ (Gaussian or uniform on the sphere), $q$ is the number of perturbation directions per step (default $q=1$), $\varepsilon$ is the finite-difference perturbation scale, $\Delta f_\varepsilon$ is the scalar projected gradient, $g_t$ is its mean over the $q$ directions, $v_t$ is the scalar second moment with decay rate $\beta$, $\hat v_t$ is its bias-corrected value, $\eta$ is the base step size, and $\zeta$ is a small constant for numerical stability.

Reference: Hassan Dbouk, Nidham Gazagnadou, Matthias Reisser, Christos Louizos, "On Adaptivity in Zeroth-Order Optimization", arXiv preprint 2026. https://arxiv.org/abs/2605.03869

---
[Back to the Canon](../index.md)
