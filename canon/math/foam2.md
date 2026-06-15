# FOAM

Implements FOAM, a Shampoo variant that adaptively tunes the preconditioner damping and the eigendecomposition refresh frequency to absorb staleness error.

Shampoo factors the per-parameter-matrix gradient $G_t \in \mathbb{R}^{m \times n}$ into left and right second-moment accumulators and preconditions with their inverse $p$-th roots. To avoid recomputing eigendecompositions every step, practitioners reuse stale factors $L_{t_0(t)}, R_{t_0(t)}$ from the last refresh step $t_0(t)$, which injects a staleness-oriented error into the update. FOAM counters this by growing the damping factor $\epsilon_t$ whenever an operator-error proxy $h_t$ exceeds a threshold $\tau$, and by triggering a fresh eigendecomposition (and resetting $\epsilon_t \to \epsilon_0$) once the projected damping exceeds a ceiling. Larger damping provably shrinks the sensitivity of the inverse root to stale statistics, trading a small bias for stability.

$$
\begin{aligned}
L_t &= \beta\, L_{t-1} + (1-\beta)\, G_t G_t^\top, \qquad R_t = \beta\, R_{t-1} + (1-\beta)\, G_t^\top G_t \\
h_t &= \frac{\alpha(\epsilon_{t-1})}{p}\, \big\| \hat{L}_t^{-1/2} (L_t - \hat{L}_t)\, \hat{L}_t^{-1/2} \big\|_F \\
\epsilon_t &= \max\!\big(\epsilon_0,\; \epsilon_{t-1}\, h_t / \tau \big) \\
\hat{L}_t &= \big(L_{t_0(t)} + \epsilon_t I_m\big)^{-1/p}, \qquad \hat{R}_t = \big(R_{t_0(t)} + \epsilon_t I_n\big)^{-1/p} \\
\theta_{t+1} &= \theta_t - \eta\, \hat{L}_t\, G_t\, \hat{R}_t
\end{aligned}
$$

where $\theta$ is a weight matrix, $\eta$ is the learning rate, $G_t$ the gradient, $\beta$ the accumulator decay, $p$ the root order (typically $4$), $L_t/R_t$ the left/right second-moment factors, $t_0(t)$ the most recent eigendecomposition refresh step, $\epsilon_0$ the base damping, $\tau$ the error threshold, $\alpha(\epsilon) = \|\hat{L}_t^{-1/p}\|_2 / \|\hat{L}_t^{-1/p}\|_F$ a normalization, and $h_t$ the relative operator-error proxy that drives both the adaptive damping and the refresh frequency.

Reference: Kyunghun Nam, Sumyeong Ahn, "FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo", ICML 2026. https://arxiv.org/abs/2606.02365

---
[Back to the Canon](../README.md)
