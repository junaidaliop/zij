# SGDP

Implements SGDP, SGD with the AdamP scale-invariant projection step.

The SGD-with-momentum update $p_t$ is projected onto the tangent
space of the weight $\theta$ whenever the weight is scale-invariant,
removing the radial component that drives effective-step-size decay:


$$
\begin{aligned}
     b_t &= \mu \, b_{t-1} + (1 - \tau) g_t \\
     p_t &= g_t + \mu \, b_t \quad\text{(Nesterov)} \quad\text{or}\quad b_t \\
     q_t &= \Pi_{\theta_{t-1}}(p_t) \\
     \theta_t &= \theta_{t-1} - \eta \, q_t
\end{aligned}
$$

where $\mu$ is the momentum, $\tau$ the dampening, and
$\Pi_{\theta}(p) = p - (\hat{\theta} \cdot p)\,\hat{\theta}$ projects
out the component of $p$ along the unit weight $\hat{\theta}$.

Reference: Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han,
Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha, "AdamP: Slowing Down
the Slowdown for Momentum Optimizers on Scale-invariant Weights", ICLR 2021.
https://arxiv.org/abs/2006.08217

---
[Back to the Canon](../index.md)
