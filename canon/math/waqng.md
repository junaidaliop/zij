# WA-QNG

Implements WA-QNG, a weighted approximate quantum natural gradient for the variational quantum eigensolver.

Quantum natural gradient preconditions the gradient by the inverse of the quantum Fisher information matrix $F$ of the full state, which is expensive to estimate. WA-QNG replaces $F$ with a cheap surrogate built from subsystem metric tensors, one per term of the decomposed Hamiltonian $H=\sum_m h_m H_m$. Each subsystem metric $T_m$ is weighted by $h_m^2$, so terms with larger Hamiltonian coefficients contribute more to the geometry, and the normalization makes the surrogate reduce to standard QNG when every term acts on the full system.

$$
\begin{aligned}
W &= \frac{2}{\sum_m h_m^2}\sum_m h_m^2\, T_m, \qquad (T_m)_{ij} = \mathrm{tr}\!\left(\partial_i \rho_m\, \partial_j \rho_m\right),\\
\theta_{t+1} &= \theta_t - \eta\, W^{+}\, g_t.
\end{aligned}
$$

where $\theta$ are the circuit parameters, $\eta$ is the learning rate, $g_t=\nabla f(\theta_t)$ is the gradient of the objective, $h_m$ are the Hamiltonian coefficients, $\rho_m$ is the reduced density matrix of subsystem $m$, $T_m$ is its Hilbert-Schmidt metric tensor, and $W^{+}$ is the Moore-Penrose pseudoinverse of the weighted metric $W$.

Reference: Chenyu Shi, Vedran Dunjko, Hao Wang, "Weighted Approximate Quantum Natural Gradient for Variational Quantum Eigensolver", arXiv 2025. https://arxiv.org/abs/2504.04932

---
[Back to the Canon](../README.md)
