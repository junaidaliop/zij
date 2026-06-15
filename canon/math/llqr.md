# LLQR

Implements LLQR, a layerwise optimal-control preconditioner that learns a structured inverse curvature matrix by solving a linear-quadratic regulator over the network's layers.

The starting point is an exact equivalence: a steepest-descent step under a broad class of divergence-induced quadratic models (Newton, Gauss-Newton, Fisher/natural gradient, and intermediate-layer metrics) can be written as a finite-horizon linear quadratic regulator (LQR) problem, where the "state" is the activation perturbation $\delta x_i$ propagated through the linearized layer dynamics and the "control" is the parameter perturbation $\delta\theta_i$. Solving this LQR by a backward Riccati recursion yields the optimal feedback control, i.e. the exact preconditioned step, without forming or inverting the global curvature matrix.

To make this scalable, LLQR relaxes the problem: instead of solving the full LQR every step, it learns a block-diagonal inverse preconditioner $U=\mathrm{diag}(U_0,\dots,U_{N-1})$ (diagonal, Kronecker-factored, or other structure) by minimizing the LQR objective over a short inner loop, reuses it across iterations via an EMA, and feeds the resulting preconditioned gradient to an outer optimizer (SGDM or AdamW).

$$
\begin{aligned}
\delta x_{i+1} &= A_i\,\delta x_i - B_i U_i\, g_i^k, \qquad \delta x_0 = 0 \\
K_i &= A_i^{\top} K_{i+1} A_i + Q_i - (A_i^{\top} K_{i+1} B_i + M_i^{\top})(R_i + B_i^{\top} K_{i+1} B_i)^{-1}(M_i + B_i^{\top} K_{i+1} A_i) \\
\lambda_i &= A_i^{\top}\lambda_{i+1} - (A_i^{\top} K_{i+1} B_i + M_i^{\top})(R_i + B_i^{\top} K_{i+1} B_i)^{-1} B_i^{\top}\lambda_{i+1} \\
\delta\theta_i^{\star} &= -(R_i + B_i^{\top} K_{i+1} B_i)^{-1}\big[(M_i + B_i^{\top} K_{i+1} A_i)\,\delta x_i + B_i^{\top}\lambda_{i+1}\big] \\
\Delta\theta_i &= -U_i\, g_i^k \\
U &\leftarrow \beta\, U + (1-\beta)\, U_T \\
\theta^{k+1} &= O_{\mathrm{out}}(\theta^k,\, \Delta\theta^k,\, \eta)
\end{aligned}
$$

where $\theta$ are the parameters, $g_i^k=\nabla_{\theta_i}L(\theta^k)$ the per-layer gradient, $A_i,B_i$ the linearized activation dynamics, $Q_i,R_i,M_i$ the state, control, and cross cost matrices induced by the chosen metric, $K_i$ the LQR value (Riccati) matrix with terminal condition $K_N=Q_N$, $\lambda_i$ the costate with $\lambda_N=g_N$, $\delta\theta_i^{\star}$ the optimal control giving the exact step, $U=\mathrm{diag}(U_0,\dots,U_{N-1})$ the learned structured inverse preconditioner ($U_T$ its value after the inner optimization), $\beta$ the EMA decay (default $0.95$), $O_{\mathrm{out}}$ the outer optimizer (SGDM or AdamW) carrying learning rate $\eta$, and $\Delta\theta^k$ the stacked preconditioned step.

Reference: Simon Dufort-Labbé, Pierre-Luc Bacon, Razvan Pascanu, Simon Lacoste-Julien, Aristide Baratin, "Layerwise LQR for Geometry-Aware Optimization of Deep Networks", arXiv 2026. https://arxiv.org/abs/2605.04230

---
[Back to the Canon](../README.md)
