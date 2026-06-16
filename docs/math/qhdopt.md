# QHDOPT

Implements QHDOPT, a quantum solver that minimizes nonlinear functions by evolving Quantum Hamiltonian Descent (QHD) dynamics on quantum hardware.

QHD is the quantum analogue of gradient descent: instead of a single point moving downhill, a wavefunction $\Psi(t,x)$ evolves under a Schrodinger equation whose Hamiltonian combines a kinetic term (the Laplacian, which spreads and tunnels the state through barriers) with the objective $f$ acting as a potential. Measuring the state after evolution yields a low-energy configuration, i.e. an approximate minimizer. The two time-dependent coefficients act as a damping schedule that gradually shifts weight from the kinetic term toward the potential, annealing the dynamics from broad exploration to localization near a minimum.

$$
\begin{aligned}
i\,\frac{\partial}{\partial t}\Psi(t,x) &= \left[\, e^{\varphi_t}\left(-\tfrac{1}{2}\Delta\right) + e^{\chi_t} f(x) \,\right]\Psi(t,x), \\
\varphi_t &= -\log\!\left(1+\gamma t^2\right), \qquad \chi_t = \log\!\left(1+\gamma t^2\right).
\end{aligned}
$$

where $\Psi(t,x)$ is the quantum state over the search domain $\Omega$ (with $\Psi=0$ on $\partial\Omega$), $\Delta$ is the Laplacian giving the kinetic energy, $f(x)$ is the objective acting as the potential, and $\gamma>0$ controls the damping schedule that decays the kinetic coefficient $e^{\varphi_t}$ while growing the potential coefficient $e^{\chi_t}$.

Reference: Samuel Kushnir, Jiaqi Leng, Yuxiang Peng, Lei Fan, Xiaodi Wu, "QHDOPT: A Software for Nonlinear Optimization with Quantum Hamiltonian Descent", INFORMS Journal on Computing 2024. https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0587

---
[Back to the Canon](../index.md)
