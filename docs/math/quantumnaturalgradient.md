# Quantum Natural Gradient

Implements Quantum Natural Gradient, steepest descent on the Fubini-Study geometry of a parametrized quantum state.

Ordinary gradient descent treats parameter space as flat Euclidean. Quantum Natural Gradient instead measures distance by the quantum information metric, so the step follows the steepest-descent direction with respect to the geometry of the underlying Hilbert space rather than the raw coordinates. The metric is the real part of the Quantum Geometric Tensor (the Fubini-Study metric tensor), and the update preconditions the gradient by its (pseudo-)inverse. This makes the trajectory invariant to reparametrization and tends to accelerate convergence of variational quantum circuits.

$$
\begin{aligned}
G_{ij}(\theta) &= \langle \partial_i \psi_\theta \,|\, \partial_j \psi_\theta \rangle - \langle \partial_i \psi_\theta \,|\, \psi_\theta \rangle \langle \psi_\theta \,|\, \partial_j \psi_\theta \rangle \\
g_{ij}(\theta) &= \mathrm{Re}\, G_{ij}(\theta) \\
\theta_{t+1} &= \theta_t - \eta\, g^{+}(\theta_t)\, \nabla \mathcal{L}(\theta_t)
\end{aligned}
$$

where $\theta$ are the circuit parameters, $\eta$ the learning rate, $\nabla \mathcal{L}$ the gradient of the loss, $|\psi_\theta\rangle$ the parametrized quantum state with $\partial_i = \partial/\partial\theta^i$, $G$ the Quantum Geometric Tensor, $g$ its real part (the Fubini-Study metric tensor), and $g^{+}$ the Moore-Penrose pseudo-inverse. In practice the linear system $g(\theta_t)(\theta_{t+1}-\theta_t) = -\eta\, \nabla \mathcal{L}(\theta_t)$ is solved rather than inverting $g$ explicitly, and a block-diagonal approximation to $g$ is used for efficiency.

Reference: James Stokes, Josh Izaac, Nathan Killoran, Giuseppe Carleo, "Quantum Natural Gradient", Quantum 2020. https://quantum-journal.org/papers/q-2020-05-25-269/

---
[Back to the Canon](../index.md)
