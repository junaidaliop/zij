# Stochastic Quantum Hamiltonian Descent (SQHD)

Implements Stochastic Quantum Hamiltonian Descent (SQHD), a stochastic-gradient variant of Quantum Hamiltonian Descent for finite-sum objectives.

Quantum Hamiltonian Descent evolves a wavefunction $|\psi\rangle$ under a time-dependent Hamiltonian whose kinetic term drives exploration and whose potential term encodes the objective $f$. SQHD targets finite-sum problems $f(x) = \tfrac{1}{m}\sum_{j=1}^{m} f_j(x)$ by replacing the full potential with a single randomly sampled component each step, the quantum analogue of stochastic gradient descent. Each iteration is realized as a symmetric Trotter (Strang) splitting that alternates a half-step of free kinetic evolution, a full-step of evolution under the sampled potential, and another half-step of kinetic evolution.

The continuous Hamiltonian is $H(t) = e^{\psi(t)}\left(-\tfrac{1}{2}\Delta\right) + e^{\chi(t)}\hat{f}$, where $\hat{f} = \int f(x)\,|x\rangle\langle x|\,\mathrm{d}x$. Discretizing with step $\eta$ over $N$ epochs gives the per-step update:

$$
\begin{aligned}
a_j &= \exp\!\big(\psi((j+\tfrac{1}{2})\eta)\big), \qquad b_j = \exp\!\big(\chi((j+\tfrac{1}{2})\eta)\big), \\
\xi_j &\sim \mathrm{Uniform}\{1,\dots,m\}, \\
|\psi_{j+1}\rangle &= \exp\!\Big(-i\tfrac{\eta}{2}\,a_j\big(-\tfrac{1}{2}\Delta\big)\Big)\,
\exp\!\big(-i\,\eta\,b_j\,\hat{f}_{\xi_j}\big)\,
\exp\!\Big(-i\tfrac{\eta}{2}\,a_j\big(-\tfrac{1}{2}\Delta\big)\Big)\,|\psi_j\rangle .
\end{aligned}
$$

where $|\psi_j\rangle$ is the state after epoch $j$, $\Delta$ is the Laplacian (kinetic term), $\hat{f}_{\xi_j} = \int f_{\xi_j}(x)\,|x\rangle\langle x|\,\mathrm{d}x$ is the operator for the sampled component potential, $\eta$ is the discretization step, $\psi(t)$ and $\chi(t)$ are the QHD scheduling functions setting the relative kinetic and potential weights $a_j, b_j$, and the measured position of the final state $|\psi_N\rangle$ yields the candidate minimizer.

Reference: Sirui Peng, Shengminjie Chen, Xiaoming Sun, Hongyi Zhou, "Stochastic Quantum Hamiltonian Descent", arXiv 2025. https://arxiv.org/abs/2507.15424

---
[Back to the Canon](../index.md)
