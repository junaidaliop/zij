# SGN

Implements SGN, a stochastic generalized Gauss-Newton method for training deep networks.

SGN approximates the Hessian by the generalized Gauss-Newton matrix, estimated on a mini-batch, and solves a Levenberg-Marquardt-regularized linear system for the search direction. The system is solved matrix-free with conjugate gradients using only Hessian-vector products, so the curvature matrix is never formed explicitly. The parameter is then moved along the resulting direction (unit step by default, optionally scaled by a line-search step size $\gamma$).

For a mini-batch of $M$ samples, with per-sample network outputs $y_i$, the update is

$$
\begin{aligned}
H^{\mathrm{GN}}(\theta_t) &= \frac{1}{M} \sum_{i=1}^{M} J_{f_i}^{\top}(\theta_t)\, H_{\ell \circ o}(y_i)\, J_{f_i}(\theta_t), \\
g_t &= \frac{1}{M} \sum_{i=1}^{M} J_{f_i}^{\top}(\theta_t)\, J_{\ell \circ o}^{\top}(y_i), \\
\big(H^{\mathrm{GN}}(\theta_t) + \rho I\big)\, \Delta\theta_t &= -\, g_t, \\
\theta_{t+1} &= \theta_t + \gamma\, \Delta\theta_t.
\end{aligned}
$$

where $J_{f_i}$ is the Jacobian of the network output with respect to $\theta$ for sample $i$, $H_{\ell \circ o}$ is the Hessian of the loss composed with the output nonlinearity (positive semidefinite, which keeps $H^{\mathrm{GN}}$ PSD), $g_t$ is the mini-batch gradient, $\rho$ is the Levenberg-Marquardt damping, $\Delta\theta_t$ is the search direction found by conjugate gradients, and $\gamma$ is the step size (default $1$).

Reference: Matilde Gargiani, Andrea Zanelli, Moritz Diehl, Frank Hutter, "On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs", arXiv 2020. https://arxiv.org/abs/2006.02409

---
[Back to the Canon](../README.md)
