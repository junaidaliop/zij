# AGGC

Implements AGGC (Adaptive Group-wise Gradient Clipping), a gradient preprocessing scheme that stabilizes large language model training by clipping each parameter group's gradient into an adaptive band tracked by an exponential moving average.

Rather than a single global clip threshold, AGGC partitions parameters into groups and maintains a running estimate $S_t$ of each group's gradient norm. The clip band $[L_t, U_t]$ is derived from this estimate with time-dependent coefficients, so the bounds tighten or relax over the course of training. Group gradients whose norm leaves the band are rescaled back to it; gradients inside the band pass through unchanged. The rescaled gradient is then handed to the base optimizer (AdamW).

$$
\begin{aligned}
n_t &= \Big( \textstyle\sum_{i} \lVert g_{t,i} \rVert_2^2 \Big)^{1/2} \\
S_t &= \beta\, S_{t-1} + (1-\beta)\, n_t \\
L_t &= \max\!\big(\mathrm{min\_norm},\; \alpha^{\mathrm{low}}_t\, S_t\big), \qquad U_t = \alpha^{\mathrm{high}}_t\, S_t \\
c_t &=
\begin{cases}
U_t / (n_t + \epsilon), & n_t > U_t \\
L_t / (n_t + \epsilon), & n_t < L_t \\
1, & \text{otherwise}
\end{cases} \\
g_{t,i} &\leftarrow c_t\, g_{t,i}
\end{aligned}
$$

where $g_{t,i}$ is the gradient of the $i$-th parameter in a group at step $t$, $n_t$ the group gradient norm, $S_t$ its EMA with decay $\beta \in [0,1)$, $[L_t, U_t]$ the adaptive clip band built from time-dependent coefficients $\alpha^{\mathrm{low}}_t, \alpha^{\mathrm{high}}_t$ (linearly scheduled from initial to late values across a transition window), $\mathrm{min\_norm} \ge 0$ a floor on the lower bound, $c_t$ the per-group clip factor, and $\epsilon$ a small stability constant.

Reference: Zhiyuan Li, Yuan Wu, Yi Chang, "AGGC: Adaptive Group Gradient Clipping for Stabilizing Large Language Model Training", arXiv 2025. https://arxiv.org/abs/2601.11864

---
[Back to the Canon](../README.md)
