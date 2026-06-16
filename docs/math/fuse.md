# FUSE

Implements FUSE, a unified synthesis of a first-order Adam step and a second-order L-BFGS step.

FUSE runs both an Adam update and an L-BFGS quasi-Newton update at each iteration and blends them by a convex weight $\theta \in [0,1]$: $\theta = 1$ recovers pure Adam, $\theta = 0$ pure L-BFGS, and intermediate values mix the cheap first-order direction with the curvature-aware second-order direction. The L-BFGS search direction $p_k$ comes from the standard two-loop recursion over a history of $(s_i, y_i)$ pairs, and its step size $\alpha_k$ is chosen by a Wolfe line search. A practical variant (FUSE-PV) hard-switches between the two ($\theta \in \{0,1\}$) once a switchover criterion on the gradient norm or loss change is met.

$$
\begin{aligned}
m_{k+1} &= \beta_1 m_k + (1 - \beta_1)\, g(x_k) \\
v_{k+1} &= \beta_2 v_k + (1 - \beta_2)\, g(x_k) \odot g(x_k) \\
x^A_{k+1} &= x_k - \alpha\, \frac{\sqrt{1 - \beta_2^k}}{1 - \beta_1^k}\; m_{k+1} \oslash \left(\sqrt{v_{k+1}} + a\right) \\
x^L_{k+1} &= x_k + \alpha_k\, p_k \\
x_{k+1} &= \theta\, x^A_{k+1} + (1 - \theta)\, x^L_{k+1}
\end{aligned}
$$

where $x$ are the parameters, $\alpha$ the Adam learning rate, $g(x_k)$ the gradient, $m_k$ and $v_k$ the first- and second-moment estimates with decay rates $\beta_1,\beta_2$, $a$ a small stability constant, $\odot$ and $\oslash$ elementwise product and division, $p_k$ the L-BFGS direction from two-loop recursion, $\alpha_k$ its Wolfe-line-search step, and $\theta \in [0,1]$ the weight blending the Adam step $x^A_{k+1}$ with the L-BFGS step $x^L_{k+1}$.

Reference: Zhanhong Jiang, Md Zahid Hasan, Aditya Balu, Joshua R. Waite, Genyi Huang, Soumik Sarkar, "FUSE: First-Order and Second-Order Unified SynthEsis in Stochastic Optimization", arXiv 2025. https://arxiv.org/abs/2503.04204

---
[Back to the Canon](../index.md)
