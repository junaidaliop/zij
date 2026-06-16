# Rotosolve / Rotoselect

Implements Rotosolve / Rotoselect, gradient-free coordinate optimization for parameterized quantum circuits.

For a circuit built from single-qubit Pauli rotations $e^{-i\theta_d P/2}$, the expectation value of an observable $M$ as a function of one rotation angle $\theta_d$ (all other parameters frozen) is exactly sinusoidal. Rotosolve uses this to jump straight to the per-coordinate minimizer in closed form, requiring only three circuit evaluations per parameter rather than gradient estimates. Sweeping over all coordinates yields the optimizer. Rotoselect extends this by also choosing the rotation generator $P \in \{X, Y, Z\}$ that gives the lowest energy at each position.

Holding all other angles fixed, $\langle M \rangle_{\theta_d} = A \sin(\theta_d + B) + C$, whose unique minimum on a period sets the update. Evaluating at $\phi$, $\phi + \tfrac{\pi}{2}$, and $\phi - \tfrac{\pi}{2}$ gives:

$$
\begin{aligned}
\theta_d^{*} &= \phi - \frac{\pi}{2} - \mathrm{arctan2}\!\left(2\langle M\rangle_{\phi} - \langle M\rangle_{\phi+\pi/2} - \langle M\rangle_{\phi-\pi/2},\; \langle M\rangle_{\phi+\pi/2} - \langle M\rangle_{\phi-\pi/2}\right), \\
H_d &\leftarrow \mathrm{arg\,min}_{P \in \{X,Y,Z\}}\; \langle M \rangle_{\theta_d^{*}(P),\, P} \quad (\text{Rotoselect}).
\end{aligned}
$$

where $\theta_d$ is the angle of the $d$-th rotation gate, $\phi$ is a reference angle (commonly set to $0$, so the three evaluations are at $0, \pm\tfrac{\pi}{2}$), $\langle M \rangle_{\theta}$ is the observable expectation with that gate set to angle $\theta$, $\mathrm{arctan2}$ is the two-argument arctangent, $H_d = P$ is the chosen Pauli generator, and the minimizing energy at $\theta_d^{*}$ equals $C - |A|$ with $C = \tfrac{1}{2}(\langle M\rangle_{\phi+\pi/2} + \langle M\rangle_{\phi-\pi/2})$.

Reference: Mateusz Ostaszewski, Edward Grant, Marcello Benedetti, "Structure optimization for parameterized quantum circuits", Quantum 5, 391 (2021). https://quantum-journal.org/papers/q-2021-01-28-391/

---
[Back to the Canon](../index.md)
