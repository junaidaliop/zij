# Quantum Hamiltonian Descent (QHD)

Implements Quantum Hamiltonian Descent (QHD), a quantum dynamical optimizer that quantizes the continuous-time limit of accelerated gradient descent.

QHD takes the classical Bregman-Lagrangian view of accelerated first-order methods, whose continuous-time limit is a damped second-order ODE, and replaces the classical trajectory $X(t)$ with a quantum wave function $\Psi(t)$ over the search space. The objective $f(x)$ becomes a potential term and the momentum becomes the kinetic (Laplacian) term of a time-dependent Hamiltonian; the state then evolves under the Schrödinger equation. Because the evolution is quantum, the wave function tunnels through and delocalizes across barriers rather than following a single descent path, which lets QHD escape spurious local minima on nonconvex landscapes. A candidate minimizer is read out by measuring the position observable $\hat{x}$ at the final time $T$, the measured $x$ being the returned parameters $\theta$.

The classical second-order ODE that QHD quantizes, and the resulting quantum dynamics, are

$$
\begin{aligned}
\ddot{X} + (\dot{\gamma}_t - \dot{\alpha}_t)\,\dot{X} + e^{2\alpha_t + \beta_t}\,\nabla f(X) &= 0 \\
i\,\frac{d}{dt}\Psi(t) &= \hat{H}(t)\,\Psi(t) \\
\hat{H}(t) &= e^{\phi_t}\!\left(-\tfrac{1}{2}\,\Delta\right) + e^{\chi_t} f(x) \\
\theta &= \text{measure } \hat{x} \text{ in } \Psi(T)
\end{aligned}
$$

where $\Psi(t)$ is the wave function over the parameter space, $\hat{H}(t)$ the time-dependent Hamiltonian, $\Delta$ the Laplacian (the kinetic/momentum term), $f(x)$ the objective acting as the potential, $\theta = x$ the parameters, and $e^{\phi_t}, e^{\chi_t}$ time-dependent damping coefficients inherited from the Bregman-Lagrangian schedule $(\alpha_t, \beta_t, \gamma_t)$. The ratio $e^{\phi_t}/e^{\chi_t}$ is required to vanish for large $t$ so that kinetic energy is gradually drained and the state concentrates near the global minimizer.

Reference: Jiaqi Leng, Ethan Hickman, Joseph Li, Xiaodi Wu, "Quantum Hamiltonian Descent", 2023. https://arxiv.org/abs/2303.01471

---
[Back to the Canon](../index.md)
