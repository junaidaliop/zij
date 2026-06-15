# FedAdamW

Implements FedAdamW, a communication-efficient federated AdamW with a global-alignment correction and aggregated second moments.

FedAdamW runs AdamW locally on each client but adds a global-alignment term $\alpha \Delta^G_r$ to every local step, pulling client trajectories toward the estimated global direction and reducing client drift. To cut communication and stabilize adaptivity across rounds, the second-moment estimate is not reset to zero each round; instead each client starts from the server-aggregated mean $\bar v^r$, and only block-wise means of the second moment are sent back. The server reconstructs the global update $\Delta^G_r$ from the averaged local parameter changes and broadcasts it together with the new model and aggregated second moment.

For client $i$ at round $r$, initialize $x_i^{r,0} = x^r$, $m_i^{r,0} = 0$, $v_i^{r,0} = \bar v^r$, then for local steps $k = 1, \dots, K$:

$$
\begin{aligned}
g_t &= \nabla f_i(x_i^{r,k}; \xi_i), \\
m_i^{r,k} &= \beta_1 m_i^{r,k-1} + (1 - \beta_1) g_t, \\
v_i^{r,k} &= \beta_2 v_i^{r,k-1} + (1 - \beta_2)\, g_t \odot g_t, \\
\hat m_i^{r,k} &= \frac{m_i^{r,k}}{1 - \beta_1^{k}}, \qquad
\hat v_i^{r,k} = \frac{v_i^{r,k}}{1 - \beta_2^{t}}, \\
x_i^{r,k+1} &= x_i^{r,k} - \eta\!\left( \frac{\hat m_i^{r,k}}{\sqrt{\hat v_i^{r,k}} + \epsilon} + \alpha\, \Delta^G_r - \lambda\, x_i^{r,k} \right),
\end{aligned}
$$

with server aggregation over the $S$ participating clients:

$$
\begin{aligned}
\Delta^G_r &= -\frac{1}{S K \eta} \sum_{i=1}^{S} \left( x_i^{r,K} - x_i^{r,0} \right), \\
x^{r+1} &= x^r + \frac{1}{S} \sum_{i=1}^{S} \left( x_i^{r,K} - x_i^{r,0} \right), \qquad
\bar v^{r+1} = \frac{1}{S} \sum_{i=1}^{S} \bar v_i,
\end{aligned}
$$

where $x$ are the parameters, $\eta$ the learning rate, $g_t$ the stochastic gradient, $m$/$v$ the first/second moments with bias-corrected $\hat m$/$\hat v$, $\beta_1, \beta_2$ the decay rates, $t$ the global step index, $\epsilon$ the stability constant, $\lambda$ the decoupled weight decay, $\alpha$ the global-alignment coefficient, $\Delta^G_r$ the global update estimate, $\bar v_i$ the block-wise mean of client $i$'s second moment, $K$ the local steps, and $S$ the clients sampled per round.

Reference: Junkang Liu, Fanhua Shang, Kewen Zhu, Hongying Liu, Yuanyuan Liu, Jin Liu, "FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models", arXiv 2025. https://arxiv.org/abs/2510.27486

---
[Back to the Canon](../README.md)
