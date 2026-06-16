# AsyncSAM

Implements AsyncSAM, a sharpness-aware optimizer that computes the ascent perturbation from a stale gradient so it can run in parallel with the descent step.

Sharpness-Aware Minimization spends two sequential forward-backward passes per step: one to find the worst-case perturbation $\epsilon_t$ and one to take the descent step at the perturbed point. AsyncSAM breaks this serial dependency by reusing a slightly out-of-date gradient, from iteration $t-\tau$, to build the perturbation. With $\tau$ fixed to one iteration the ascent computation for step $t$ can be overlapped with the descent computation of step $t-1$, removing the extra pass from the critical path while keeping the flat-minima-seeking behavior of SAM.

$$
\begin{aligned}
\epsilon_t &= \rho \, \frac{g_{t-\tau}}{\lVert g_{t-\tau} \rVert} \\
\theta_{t+1} &= \theta_t - \eta \, \nabla L_t(\theta_t + \epsilon_t)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\rho$ the perturbation radius, $g_{t-\tau} = \nabla L_{t-\tau}(\theta_{t-\tau})$ the gradient at the stale iteration $t-\tau$ (with $\tau$ the degree of asynchrony, typically $1$), and $\nabla L_t(\theta_t + \epsilon_t)$ the descent gradient evaluated at the perturbed parameters; setting $\tau = 0$ recovers standard SAM.

Reference: Junhyuk Jo, Jihyun Lim, Sunwoo Lee, "Asynchronous Sharpness-Aware Minimization for Fast and Accurate Deep Learning", arXiv 2025. https://arxiv.org/abs/2503.11147

---
[Back to the Canon](../index.md)
