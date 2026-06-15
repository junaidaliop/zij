# Universal AQC neural-network training

Implements universal adiabatic quantum neural-network training, learning weights by adiabatically evolving a qubit register into the ground state of a Hamiltonian that encodes the loss.

Each weight is discretized over $N$ qubits and promoted to a Pauli-spin operator, so the training loss becomes a problem Hamiltonian $\hat H = V(\hat w)$. Starting from the ground state of a trivial mixer $\hat H_0$, the system is evolved under a time-dependent interpolating Hamiltonian on a linear schedule $s(t)=t/t_{\mathrm{final}}$. The adiabatic theorem guarantees that, if the evolution is slow enough, the register stays in the instantaneous ground state and ends in the minimizer of the loss; measuring the qubits then reads off the trained weights. There is no iterative gradient step — optimization is the continuous-time Hamiltonian flow itself.

$$
\begin{aligned}
\hat H_A(t) &= \bigl[1 - s(t)\bigr]\,\hat H_0 + s(t)\,\hat H, \qquad s(t) = \frac{t}{t_{\mathrm{final}}}, \\
\hat H_0 &= -\sum_{\ell} X_\ell, \\
\hat w &= \sum_{\ell=1}^{N} 2^{-\ell}\,\frac{1 - Z_\ell}{2}, \\
\hat H &= V(\hat w), \qquad \mathcal{L} = \sum_a \bigl(Y(x_a) - y_a\bigr)^2.
\end{aligned}
$$

where $\hat H_A(t)$ is the interpolating Hamiltonian with schedule $s(t)$ running from $s(0)=0$ to $s(t_{\mathrm{final}})=1$, $\hat H_0$ is the transverse-field mixer whose ground state is the uniform superposition, $X_\ell$ and $Z_\ell$ are Pauli operators on qubit $\ell$, $\hat w$ is the $N$-qubit discretization of a weight into bins on $[0,1]$, $V(\cdot)$ encodes the loss as a polynomial in the weight operators, and $\mathcal{L}$ is the mean-squared error over data $(x_a, y_a)$ with network output $Y$. The trained weights are the eigenvalues recovered by measuring all $Z_\ell$ in the final ground state.

Reference: Steve Abel, Juan Carlos Criado, Michael Spannowsky, "Training neural networks with universal adiabatic quantum computing", Frontiers in Artificial Intelligence 2024. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1368569/full

---
[Back to the Canon](../README.md)
