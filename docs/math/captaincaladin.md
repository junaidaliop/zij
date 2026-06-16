# CAPTAIN (C-ALADIN)

Implements CAPTAIN, a Consensus ALADIN scheme that adaptively refreshes its second-order information.

C-ALADIN (Consensus Augmented Lagrangian Alternating Direction Inexact Newton) solves a distributed consensus problem $\min \sum_i f_i(\theta_i)$ subject to $\theta_i = y$ across $N$ agents. Each agent runs a local proximal Newton step in the $M_i$-weighted norm, a coordinator forms the consensus variable $y$, and the duals $\lambda_i$ are updated in closed form. CAPTAIN augments this with an auxiliary variable $z$ that gates when the curvature matrices $M_i$ are refreshed: the Hessian information is only re-evaluated once the coupling potential $\Phi$ has decreased enough relative to its value at the last refresh point, which keeps the iteration cheap while preserving global convergence.

Per iteration $k$, each agent $i$ performs the local solve, gradient read-off, consensus coordination, and dual update:

$$
\begin{aligned}
\theta_i^{k+1} &= \arg\min_{\theta_i}\; f_i(\theta_i) + (\lambda_i^{k})^{\top}\theta_i + \tfrac{1}{2}\lVert \theta_i - y^{k}\rVert_{M_i}^{2},\\
g_i &= M_i\,(y^{k} - \theta_i^{k+1}) - \lambda_i^{k},\\
y^{k+1} &= \Big(\textstyle\sum_{i=1}^{N} M_i\Big)^{-1}\Big(\textstyle\sum_{i=1}^{N} M_i\,\theta_i^{k+1} - g_i\Big),\\
\lambda_i^{k+1} &= M_i\,(\theta_i^{k+1} - y^{k+1}) - g_i.
\end{aligned}
$$

The auxiliary variable and curvature are refreshed to index $t+1$ only when both $\Phi(y^{k+1}) < \Phi(z^{t}) - \tfrac{\gamma^{t}}{2}\lVert y^{k+1} - z^{t}\rVert^{2}$ and $\lVert y^{k+1} - z^{t}\rVert \ge \epsilon$ hold, in which case $z^{t+1} = y^{k+1}$, $M_i \approx \nabla^2 f_i(z^{t+1}) \succ \mu_i I$, and $\gamma^{t+1} = \rho_{\min}\!\big(\sum_i M_i\big)$.

where $\theta_i$ is agent $i$'s local copy of the decision variable, $y$ is the consensus (coordinator) variable, $\lambda_i$ is the dual associated with the consensus constraint, $M_i$ is the local positive-definite scaling/Hessian-approximation matrix, $g_i$ is the recovered local gradient, $z$ is the auxiliary point at which curvature was last refreshed, $\Phi$ is the coupling potential, $\gamma$ the adaptive decrease parameter, $\rho_{\min}$ the smallest eigenvalue, $\mu_i$ a strong-convexity floor, and $\epsilon$ a stalling threshold.

Reference: Xu Du, Shuting Wu, Karl H. Johansson, Apostolos I. Rikos, "A Global Convergence Analysis of Consensus ALADIN for Convex Optimization", arXiv preprint 2026. https://arxiv.org/abs/2606.08112

---
[Back to the Canon](../index.md)
