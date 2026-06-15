# H-QNG

Implements H-QNG, a Hamiltonian-aware quantum natural gradient descent for variational quantum eigensolvers.

Quantum natural gradient preconditions the energy gradient with the quantum Fisher information (the pullback of the Fubini-Study metric), which requires estimating a matrix indexed over the full Pauli basis. H-QNG replaces that full metric with a cheaper one built only from the $v$ Pauli terms that actually appear in the problem Hamiltonian $H = \sum_{r=1}^{v} a_r P_r$, so the preconditioner reuses the same measurements already needed for the energy.

The metric entries are formed from the projections of the parameter derivatives of the state onto the Hamiltonian's Pauli terms, weighted by the squared coefficients, then rescaled so the method coincides with standard QNG when all $4^n$ Pauli terms are present. The parameters are then updated by the preconditioned gradient step:

$$
\begin{aligned}
G_{ij} &= \sum_{r=1}^{v} a_r^2 \, \mathrm{tr}(\partial_i \rho_\theta \, P_r) \, \mathrm{tr}(\partial_j \rho_\theta \, P_r) \\
T_{ij} &= \frac{1}{2\sqrt{\sum_{r=1}^{v} a_r^2}} \, G_{ij} \\
\theta^{(k+1)} &= \theta^{(k)} - \eta \, T^{-1} \nabla f(\theta^{(k)})
\end{aligned}
$$

where $\theta$ are the variational circuit parameters, $\eta$ the learning rate, $\rho_\theta = U(\theta)\,|0\rangle\langle 0|\,U^\dagger(\theta)$ the parameterized state, $\partial_i \rho_\theta$ its derivative with respect to $\theta_i$, $a_r$ and $P_r$ the coefficients and Pauli operators of $H$, $G$ the Hamiltonian-restricted metric, $T$ its rescaled form, and $\nabla f(\theta)$ the gradient of the energy $f(\theta) = \mathrm{tr}(\rho_\theta H)$.

Reference: Chenyu Shi, Hao Wang, "Efficient Hamiltonian-aware Quantum Natural Gradient Descent for Variational Quantum Eigensolvers", arXiv 2025. https://arxiv.org/abs/2511.14511

---
[Back to the Canon](../README.md)
