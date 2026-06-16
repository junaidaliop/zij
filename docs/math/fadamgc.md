# FAdamGC

Implements FAdamGC, federated Adam with gradient tracking for heterogeneous-data optimization.

In federated learning each client runs Adam locally on its own data, but client drift under heterogeneous (non-IID) data biases the aggregated update. FAdamGC corrects this by tracking gradients: a global tracking variable $y_t$ and per-client tracking variables $y_{i,t}$ are maintained, and each local gradient is centered toward the global direction before being fed into the Adam moment estimates. Each client then performs $K$ local Adam steps on the corrected gradient, and the server aggregates the resulting model and tracking deltas.

For client $i$ at round $t$, local step $k$:

$$
\begin{aligned}
\hat{g}_{i,t}^{(k)} &= g_{i,t}^{(k)} + y_t - y_{i,t} \\
m_{i,t}^{(k+1)} &= \beta_1 m_{i,t}^{(k)} + (1-\beta_1)\,\hat{g}_{i,t}^{(k)} \\
v_{i,t}^{(k+1)} &= \beta_2 v_{i,t}^{(k)} + (1-\beta_2)\,\hat{g}_{i,t}^{(k)} \odot \hat{g}_{i,t}^{(k)} \\
\hat{v}_{i,t}^{(k+1)} &= \max\!\big(\hat{v}_{i,t}^{(k)},\, v_{i,t}^{(k+1)}\big) \\
\Delta_{i,t}^{(k)} &= \frac{m_{i,t}^{(k+1)}}{\sqrt{\hat{v}_{i,t}^{(k+1)}} + \epsilon} \\
x_{i,t}^{(k+1)} &= x_{i,t}^{(k)} - \eta_l\, \Delta_{i,t}^{(k)} \\
y_{i,t+1} &= \tfrac{1}{K}\sum_{k=1}^{K} g_{i,t}^{(k)} \\
x_{t+1} &= x_t + \eta_g\, \tfrac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t}\big(x_{i,t}^{(K+1)} - x_t\big)
\end{aligned}
$$

where $x_t$ is the global model, $x_{i,t}^{(k)}$ the client iterate, $g_{i,t}^{(k)}$ the stochastic gradient, $\hat{g}_{i,t}^{(k)}$ the gradient-tracking-corrected gradient, $m,v$ the Adam moments with decays $\beta_1,\beta_2$, $\hat{v}$ the AMSGrad maximum, $\epsilon$ the stability constant, $\eta_l$ and $\eta_g$ the local and global learning rates, $K$ the number of local steps, $\mathcal{S}_t$ the set of participating clients, and $y_t,y_{i,t}$ the global and per-client gradient-tracking variables aggregated by the server.

Reference: Evan Chen, Jianing Zhang, Shiqiang Wang, Chaoyue Liu, Christopher Brinton, "Parameter Tracking in Federated Learning with Adaptive Optimization", arXiv 2025. https://arxiv.org/abs/2502.02727

---
[Back to the Canon](../index.md)
