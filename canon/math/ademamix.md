# AdEMAMix

Implements AdEMAMix, an Adam variant mixing a fast and a slow gradient EMA.


$$
\begin{aligned}
m_{1,t} &= \beta_1 m_{1,t-1} + (1 - \beta_1)\, g_t \\
m_{2,t} &= \beta_3 m_{2,t-1} + (1 - \beta_3)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\theta_t &= \theta_{t-1} - \eta\left(
            \frac{m_{1,t} / (1 - \beta_1^t) + \alpha\, m_{2,t}}
                 {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
            + \lambda \theta_{t-1}\right)
\end{aligned}
$$

where $m_{1,t}$ is the fast EMA with decay $\beta_1$,
$m_{2,t}$ the slow EMA with decay $\beta_3$, $\alpha$
the coefficient mixing the two, and $\lambda$ the decoupled weight
decay. The slow EMA $m_{2,t}$ is not bias-corrected. When
`beta3_warmup` or `alpha_warmup` is set, $\beta_3$ and
$\alpha$ are ramped from $\beta_1$ and $0$ over that many
steps.

Reference: Matteo Pagliardini, Pierre Ablin, David Grangier,
"The AdEMAMix Optimizer: Better, Faster, Older", arXiv 2024.
https://arxiv.org/abs/2409.03137

---
[Back to the Canon](../README.md)
