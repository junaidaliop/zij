# MAdam

Implements MAdam, a metric-aware wrapper that preconditions a multi-objective solver's reconciled direction before handing it to Adam.

Multi-objective solvers (loss-balancing, gradient-balancing, Pareto-based) reconcile $C$ per-objective gradients $g_i^{(t)}$ into a single descent direction $d^{(t)} = \sum_i \lambda_i^{(t)} g_i^{(t)}$ under a time-varying preference $\lambda^{(t)}$, then delegate the step to Adam. MAdam observes that Adam's running second moment entangles the preference with gradient statistics (collapsing it into a history average) and imposes its own diagonal RMS metric in place of the Euclidean geometry the solver assumes. The fix is to whiten $d^{(t)}$ by the preference-conditioned curvature of the scalarized objective: on this whitened input Adam's second moment collapses to identity, so the realized step is governed by the intended metric.

The curvature is the diagonal second moment of the scalarized gradient at the current preference, $C_{\lambda^{(t)}} = \sum_{i,j} \lambda_i^{(t)} \lambda_j^{(t)} F_{ij}$, where the cross-objective Fisher blocks $F_{ij}$ are estimated online by an EMA sharing Adam's decay $\beta_2$. Preconditioning the direction by $M_{\lambda^{(t)}}^{-1}$ and feeding it to Adam yields, under the stationarity approximation $\hat v^{(t)} \approx 1$, the realized parameter update below.

$$
\begin{aligned}
d^{(t)} &= \sum_{i=1}^{C} \lambda_i^{(t)} g_i^{(t)} \\
F_{ij}^{(t)} &= \beta_2 F_{ij}^{(t-1)} + (1 - \beta_2)\, g_i^{(t)} \odot g_j^{(t)} \\
C_{\lambda^{(t)}} &= \sum_{i,j=1}^{C} \lambda_i^{(t)} \lambda_j^{(t)} F_{ij}^{(t)} \\
M_{\lambda^{(t)}} &= \mathrm{Diag}\!\left( \sqrt{C_{\lambda^{(t)}}} + \epsilon \right) \\
\theta^{(t+1)} &= \theta^{(t)} - \eta\, M_{\lambda^{(t)}}^{-1} d^{(t)}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_i^{(t)}$ the gradient of objective $i$, $\lambda^{(t)} \in \Delta^{C-1}$ the per-step preference vector on the probability simplex, $d^{(t)}$ the reconciled descent direction, $F_{ij}^{(t)}$ the EMA of the cross-objective Fisher interaction, $C_{\lambda^{(t)}}$ the preference-conditioned diagonal curvature, $M_{\lambda^{(t)}}$ the resulting diagonal metric, $\odot$ element-wise product, $\beta_2$ the second-moment decay shared with Adam, and $\epsilon$ a small stability constant. In practice the metric is rampup-blended toward the identity early in training, and the $F_{ij}$ EMAs are maintained via stochastic pair sampling to keep the cost at $O(1)$ backward passes per step; the whitened direction $M_{\lambda^{(t)}}^{-1} d^{(t)}$ is then passed through standard Adam.

Reference: Fengbei Liu, Rachit Saluja, Sunwoo Kwak, Ruibo Wang, Ruining Deng, Heejong Kim, Johannes C. Paetzold, Mert R. Sabuncu, "MAdam: Metric-Aware Multi-Objective Adam", arXiv 2026. https://arxiv.org/abs/2606.03904

---
[Back to the Canon](../README.md)
