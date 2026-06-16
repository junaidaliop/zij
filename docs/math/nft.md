# NFT

Implements NFT (Nakanishi-Fujii-Todo), sequential minimal optimization for variational quantum-classical hybrid circuits.

When all but one parameter $\theta_j$ are held fixed, the expectation-value cost of a parameterized quantum circuit is exactly a single sinusoid of period $2\pi$ in $\theta_j$. NFT exploits this: at each step it picks one coordinate, reconstructs that sinusoid from three cost evaluations, and jumps directly to its analytic minimizer. The method is gradient-free and hyperparameter-free, sweeping coordinates cyclically (or randomly) and recycling one evaluation per step.

For the active coordinate, write the restricted cost as $\mathcal{L}_j(\theta_j) = a_1 \cos(\theta_j - a_2) + a_3$. Measuring it at the current point and at the two shifts $\pm \tfrac{\pi}{2}$ gives $f_0, f_+, f_-$, which determine the sinusoid and hence its minimizer at $\theta_j = a_2 + \pi$:

$$
\begin{aligned}
f_0 &= \mathcal{L}_j(\theta_j^{(n-1)}), \quad f_\pm = \mathcal{L}_j\!\left(\theta_j^{(n-1)} \pm \tfrac{\pi}{2}\right), \\
\theta_j^{(n)} &= \theta_j^{(n-1)} + \arctan\!\left(\frac{f_+ - f_-}{2 f_0 - f_+ - f_-}\right) + \frac{\pi}{2}\left(1 + \mathrm{sign}\!\left(2 f_0 - f_+ - f_-\right)\right), \\
\theta_k^{(n)} &= \theta_k^{(n-1)} \quad (k \neq j).
\end{aligned}
$$

where $\theta_j^{(n)}$ is the updated value of the chosen coordinate after sweep $n$, $f_0,f_\pm$ are the three cost evaluations, the $\arctan$ recovers the phase $a_2 - \theta_j^{(n-1)}$, and the $\mathrm{sign}$ term selects the cosine minimum (adding $\pi$). The recovered minimum cost $a_3 - a_1$ is reused as $f_0$ for the next coordinate.

Reference: Ken M. Nakanishi, Keisuke Fujii, Synge Todo, "Sequential minimal optimization for quantum-classical hybrid algorithms", Phys. Rev. Research 2, 043158 (2020). https://link.aps.org/doi/10.1103/PhysRevResearch.2.043158

---
[Back to the Canon](../index.md)
