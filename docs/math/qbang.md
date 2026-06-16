# qBang

Implements qBang (quantum Broyden adaptive natural gradient), a natural-gradient optimizer for variational quantum algorithms that interweaves a Broyden-approximated metric with Adam-style momentum.

The quantum Fisher information matrix is treated as slowly varying, so instead of recomputing it each step, qBang maintains a metric $B_k$ updated by a rank-1 low-pass filter with rate $\varepsilon_k$, and propagates its inverse directly through the Sherman-Morrison identity. The descent direction is a bias-corrected, variance-normalized momentum vector (as in Adam), preconditioned by $B_k^{-1}$. This keeps each step at $\mathcal{O}(n_\theta)$ circuit evaluations while retaining natural-gradient geometry to navigate flat energy landscapes (barren plateaus).

$$
\begin{aligned}
m_k &= \beta_1 m_{k-1} + (1-\beta_1)\, g_k, \\
v_k &= \beta_2 v_{k-1} + (1-\beta_2)\, g_k \odot g_k, \\
\hat m_k &= \frac{m_k}{1-\beta_1^{k+1}}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^{k+1}}, \\
\{p_k\}_l &= \frac{\{\hat m_k\}_l}{\sqrt{\{\hat v_k\}_l} + \kappa}, \\
B_{k+1}^{-1} &= \left[ \mathbb{1} - \frac{\varepsilon_k\, B_k^{-1} g_k g_k^{\top}}{1-\varepsilon_k\,(1 - g_k^{\top} B_k^{-1} g_k)} \right] \frac{B_k^{-1}}{1-\varepsilon_k}, \\
\theta_{k+1} &= \theta_k - \frac{\eta\, B_k^{-1} p_k}{\big((k+2)-1\big)^{\varepsilon_0}}, \qquad \varepsilon_k = \frac{\varepsilon_0}{k+1}.
\end{aligned}
$$

where $\theta$ are the circuit parameters, $\eta$ the learning rate, $g_k = \nabla \mathcal{L}(\theta_k)$ the cost gradient, $m_k, v_k$ the first and second moment estimates with decays $\beta_1, \beta_2$, $\hat m_k, \hat v_k$ their bias-corrected forms, $\{p_k\}_l$ the per-coordinate normalized step ($\kappa$ a small stabilizer), $B_k$ the Broyden-filtered metric approximating the quantum Fisher information matrix, $\varepsilon_0$ the initial filter rate with schedule $\varepsilon_k$, and $\odot$ elementwise multiplication. The inverse-metric update is the Sherman-Morrison form of the rank-1 filter $B_{k+1} = (1-\varepsilon_k)B_k + \varepsilon_k\, g_k g_k^{\top}$.

Reference: David Fitzek, Robert S. Jonsson, Werner Dobrautz, Christian Schäfer, "Optimizing Variational Quantum Algorithms with qBang: Efficiently Interweaving Metric and Momentum to Navigate Flat Energy Landscapes", Quantum 8, 1313 (2024). https://arxiv.org/abs/2304.13882

---
[Back to the Canon](../index.md)
