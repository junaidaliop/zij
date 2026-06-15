# CRNAS

Implements CRNAS, cubic-regularized Newton with affine scaling for constrained second-order optimization.

CRNAS targets parameter-estimation problems where the objective is non-convex and the parameters are confined to a constraint set $A\theta = b$, $\theta \in \mathcal{K}$. Rather than following only the gradient, it builds a cubic-regularized second-order Taylor model and minimizes it over an affine-scaled trust region, using curvature to converge to points satisfying second-order optimality conditions rather than mere critical points.

The trust region is measured in the local norm induced by a self-concordant barrier $B$ for the cone $\mathcal{K}$, so the affine scaling automatically respects the constraints and keeps iterates strictly feasible. Each step solves

$$
\begin{aligned}
\theta_{t+1} = \arg\min_{\substack{A\theta = b \\ \|\theta - \theta_t\|_{\theta_t} \le 1-\alpha}} \Big( \langle g_t,\, \theta - \theta_t \rangle + \tfrac{1}{2}\, \nabla^2 L(\theta_t)[\theta - \theta_t]^2 + \tfrac{M}{6}\, \|\theta - \theta_t\|_{\theta_t}^3 \Big),
\end{aligned}
$$

where $g_t = \nabla L(\theta_t)$ is the gradient, $\nabla^2 L(\theta_t)$ is the Hessian, $\|v\|_{\theta} = \big(v^\top \nabla^2 B(\theta)\, v\big)^{1/2}$ is the local norm from the barrier $B$ of the cone $\mathcal{K}$, $M \ge \beta$ is the cubic-regularization weight bounded below by the Hessian Lipschitz constant $\beta$, and $\alpha \in (0,1)$ sets the radius of the affine-scaled ball. The method stops when $\|\theta_{t+1} - \theta_t\|_{\theta_t} < \eta$.

Reference: Chenyu Wu, Nuozhou Wang, Casey Garner, Kevin Leder, Shuzhong Zhang, "Novel Optimization Techniques for Parameter Estimation", arXiv 2024. https://arxiv.org/abs/2407.04235

---
[Back to the Canon](../README.md)
