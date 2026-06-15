# iCANS

Implements iCANS, a shot-frugal stochastic gradient optimizer for variational quantum algorithms that adapts the number of measurements per parameter.

iCANS performs gradient descent on a parameterized quantum circuit where each partial derivative is estimated by the parameter-shift rule from a finite number of measurement shots. To spend measurements efficiently, it allocates shots per parameter so as to maximize the expected gain per shot, using running estimates of the gradient and its sampling variance. Cheap, low-precision steps are taken early and the shot count grows smoothly as the optimization tightens.

$$
\begin{aligned}
g_\ell &= \frac{1}{2 s_\mathrm{tot}} \sum_{j=1}^{s_\mathrm{tot}} \left( \hat E^{+}_j - \hat E^{-}_j \right), \quad
S_\ell = \frac{1}{s_\mathrm{tot}-1} \sum_{j=1}^{s_\mathrm{tot}} \left[ \left( \tfrac{\hat E^{+}_j - \hat E^{-}_j}{2} \right)^2 - g_\ell^2 \right] \\
\theta_\ell &\leftarrow \theta_\ell - \alpha\, g_\ell \\
\chi'_\ell &\leftarrow \mu\, \chi'_\ell + (1-\mu)\, g_\ell, \qquad \xi'_\ell \leftarrow \mu\, \xi'_\ell + (1-\mu)\, S_\ell \\
\chi_\ell &= \frac{\chi'_\ell}{1-\mu^{k+1}}, \qquad \xi_\ell = \frac{\xi'_\ell}{1-\mu^{k+1}} \\
s_\ell &\leftarrow \left\lceil \frac{2 L \alpha}{2 - L \alpha} \cdot \frac{\xi_\ell}{\chi_\ell^2 + b\, \mu^{k}} \right\rceil
\end{aligned}
$$

where $\theta_\ell$ is the $\ell$-th circuit parameter, $\alpha$ the learning rate, $g_\ell$ the parameter-shift gradient estimate from $s_\mathrm{tot}$ shots with single-shot energies $\hat E^{\pm}_j$ at $\theta \pm \tfrac{\pi}{2}\hat e_\ell$, $S_\ell$ its sample variance, $\chi_\ell,\xi_\ell$ the bias-corrected running averages of gradient and variance with decay $\mu$, $L = \sum_i |c_i|$ the Lipschitz constant (sum of Hamiltonian coefficient magnitudes), $b$ a small positive regularizer, $k$ the iteration index, and $s_\ell$ the shots allotted to parameter $\ell$ (clipped to $[s_\mathrm{min}, s_\mathrm{max}]$). The coupled variant (iCANS2) instead picks the shot budget maximizing the expected gain per shot $\gamma_\ell = \tfrac{1}{s_\ell}\!\left[ (\alpha - \tfrac{L\alpha^2}{2})\chi_\ell^2 - \tfrac{L\alpha^2}{2 s_\ell}\xi_\ell \right]$.

Reference: Kübler, Arrasmith, Cincio, Coles, "An Adaptive Optimizer for Measurement-Frugal Variational Algorithms", Quantum 4, 263 (2020). https://quantum-journal.org/papers/q-2020-05-11-263/

---
[Back to the Canon](../README.md)
