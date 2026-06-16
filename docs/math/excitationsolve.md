# ExcitationSolve

Implements ExcitationSolve, a gradient-free, hyperparameter-free optimizer for excitation parameters in variational quantum eigensolvers.

Excitation generators $G$ used in unitary coupled-cluster ansätze satisfy $G^3 = G$, so their eigenvalues lie in $\{-1, 0, +1\}$. As a consequence, the energy expectation along any single parameter $\theta_j$ (holding all others fixed) is exactly a second-order trigonometric polynomial. ExcitationSolve reconstructs this one-dimensional landscape from a handful of energy evaluations, then jumps directly to its global minimizer along that coordinate, sweeping over parameters one at a time. The same quantum resources a gradient-based step would consume suffice to locate the exact per-parameter optimum.

For a single coordinate $\theta_j$, the energy takes the closed form

$$
\begin{aligned}
E(\theta_j) &= c + a_1 \cos\theta_j + b_1 \sin\theta_j + a_2 \cos 2\theta_j + b_2 \sin 2\theta_j, \\
\theta_j &\leftarrow \arg\min_{\theta \in [0, 2\pi)} E(\theta),
\end{aligned}
$$

where the five coefficients $c, a_1, b_1, a_2, b_2$ are fixed by sampling $E$ at five values $\theta_j^{(0)} + \tfrac{2\pi \ell}{5}$ for $\ell = 0,\dots,4$ (one reused from the previous step) and solving the resulting linear system; the global minimizer $\arg\min E$ is obtained analytically via the roots of $E'(\theta) = 0$ (a companion-matrix eigenvalue problem), and $\theta$ is the parameter, $E$ the measured energy expectation.

Reference: Jonas Jäger, Thierry Nicolas Kaldenbach, Max Haas, Erik Schultheis, "Fast gradient-free optimization of excitations in variational quantum eigensolvers", Communications Physics 2025. https://www.nature.com/articles/s42005-025-02375-9

---
[Back to the Canon](../index.md)
