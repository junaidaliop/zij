# Adam-SHANG

Implements Adam-SHANG, a convergent Adam-type method derived from a symplectic Hamiltonian accelerated gradient flow.

Adam-SHANG couples the parameter iterate with an auxiliary momentum iterate and a diagonal preconditioner that accumulates squared gradients, mirroring Adam's second-moment scaling. The step size is set adaptively from the trace of the preconditioner rather than fixed, which yields provable convergence for stochastic smooth convex objectives while retaining Adam-like per-coordinate adaptivity.

$$
\begin{aligned}
\alpha_k &= \lambda \sqrt{\frac{\mathrm{Tr}\big((P_k+\epsilon I)^{-1}\big)}{\mathrm{Tr}\big((P_k+\epsilon I)^{-2}\big)}} \\
\theta_{k+1} &= \frac{1}{1+\alpha_k}\,\theta_k + \frac{\alpha_k}{1+\alpha_k}\,y_k - \frac{\alpha_k\beta}{1+\alpha_k}\,(P_{k-1}+\epsilon I)^{-1} g_k \\
y_{k+1} &= y_k - \alpha_k\,(P_k+\epsilon I)^{-1} g_{k+1} \\
P_{k+1} &= \frac{1}{1+\alpha_k}\,P_k + \frac{\alpha_k\gamma}{1+\alpha_k}\,(P_k+\epsilon I)^{-1}\,\mathrm{diag}(g_{k+1}^{\odot 2})
\end{aligned}
$$

where $\theta_k$ is the parameter iterate, $y_k$ the auxiliary momentum iterate, $P_k=\mathrm{diag}(p_1,\dots,p_d)\succ 0$ the diagonal preconditioner, $g_k$ the (stochastic) gradient, $g_{k+1}^{\odot 2}$ the elementwise square, $\alpha_k$ the adaptive step size, $\lambda,\beta,\gamma\in(0,1]$ tuning constants, and $\epsilon>0$ a stability term.

Reference: Yaxin Yu, Long Chen, Minfu Feng, "Adam-SHANG: A Convergent Adam-Type Method for Stochastic Smooth Convex Optimization", arXiv 2025. https://arxiv.org/abs/2605.12878

---
[Back to the Canon](../index.md)
