# VAMO

Implements VAMO, a variance-reduced optimizer that corrects first-order minibatch gradients with a cheap zeroth-order snapshot term.

VAMO follows the SVRG epoch structure but replaces the expensive full-batch snapshot gradient with a zeroth-order estimate computed once per epoch at the checkpoint $\theta_0$. Within an epoch, each step uses a first-order minibatch gradient at the current iterate together with a control variate built from the difference between the zeroth-order estimate on the same minibatch and the checkpoint estimate, scaled by a mixing coefficient $\alpha$.

The zeroth-order estimate uses two-point (or multi-point) finite differences along random directions $u$ on the unit sphere with smoothing parameter $\mu$. At the start of epoch $s$ the snapshot $\theta_0 = \bar\theta_{s-1}$ is taken from the previous epoch's final iterate, and $\hat g = \hat\nabla f(\theta_0)$ is computed there.

$$
\begin{aligned}
\hat\nabla f_i(\theta) &= \frac{d}{\mu}\,\bigl[f_i(\theta + \mu u) - f_i(\theta)\bigr]\,u \\
\hat g &= \hat\nabla f(\theta_0) \\
v_t &= \nabla f_{I_t}(\theta_t) - \alpha\,\bigl(\hat\nabla f_{I_t}(\theta_0) - \hat g\bigr) \\
\theta_{t+1} &= \theta_t - \gamma\,v_t
\end{aligned}
$$

where $\nabla f_{I_t}(\theta_t)$ is the first-order minibatch gradient at the current iterate, $\hat\nabla f_{I_t}(\theta_0)$ is the zeroth-order estimate on the same minibatch at the checkpoint, $\alpha$ is the mixing coefficient (theory: $\alpha = 1/d$), $\gamma$ is the step size, $d$ is the problem dimension, $\mu$ is the smoothing radius, and $u$ is a random unit-sphere direction.

Reference: Jiahe Chen, Ziye Ma, "VAMO: Efficient Large-Scale Nonconvex Optimization via Adaptive Zeroth Order Variance Reduction", arXiv preprint 2025. https://arxiv.org/abs/2505.13954

---
[Back to the Canon](../index.md)
