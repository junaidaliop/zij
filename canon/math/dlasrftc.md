# DLAS-R-FTC

Implements DLAS-R-FTC, distributed adaptive-stepsize gradient descent with finite-time coordination of a common learning rate.

Each node $i$ in a network runs local gradient descent on its own objective $f_i$, but instead of a fixed learning rate it estimates the local Lipschitz constant of the gradient from successive iterates and sets an adaptive stepsize. Because uncoordinated per-node stepsizes break convergence guarantees, every round the nodes run a finite-time exact ratio consensus that averages their local stepsizes into a single common $\gamma_t$, which all nodes then apply in lockstep.

$$
\begin{aligned}
L_{i,t} &= \frac{\lVert g_{i,t} - g_{i,t-1} \rVert}{\lVert \theta_{i,t} - \theta_{i,t-1} \rVert}, \\
\widetilde{L}_{i,t} &= (1-\kappa)\, L_{i,t} + \kappa\, \widetilde{L}_{i,t-1}, \\
\eta_{i,t} &= \min\!\left( \sqrt{1 + \rho_{i,t-1}}\; \eta_{i,t-1},\; \frac{\alpha}{\widetilde{L}_{i,t}} \right), \\
\rho_{i,t} &= \frac{\eta_{i,t}}{\eta_{i,t-1}}, \\
\gamma_t &= \mathrm{FTC}\big(\{\eta_{i,t}\}_i\big) = \frac{1}{n}\sum_{j=1}^{n} \eta_{j,t}, \\
\theta_{i,t+1} &= \theta_{i,t} - \gamma_t\, g_{i,t},
\end{aligned}
$$

where $\theta_{i,t}$ is node $i$'s parameter vector, $g_{i,t}=\hat{\nabla} f_i(\theta_{i,t})$ its local gradient, $L_{i,t}$ the instantaneous local Lipschitz estimate, $\widetilde{L}_{i,t}$ its filtered version with smoothing $\kappa\in[0,1)$, $\eta_{i,t}$ the local adaptive stepsize with safety factor $\alpha$ and growth controlled by the ratio $\rho_{i,t}$, $\mathrm{FTC}(\cdot)$ the finite-time exact ratio consensus that returns the network average over the $n$ nodes, and $\gamma_t$ the resulting common coordinated learning rate.

Reference: Apostolos I. Rikos, Nicola Bastianello, Themistoklis Charalambous, Karl H. Johansson, "Distributed Optimization and Learning for Automated Stepsize Selection with Finite Time Coordination", arXiv 2025. https://arxiv.org/abs/2508.05887

---
[Back to the Canon](../README.md)
