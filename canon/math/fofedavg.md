# FOFedAvg

Implements FOFedAvg, federated averaging with a Caputo fractional-order local optimizer (FOSGD).

FOFedAvg keeps the standard FedAvg communication loop but replaces each client's local SGD step with a fractional-order update derived from a Caputo derivative of order $\alpha \in (0,1]$. A power-law factor of the most recent parameter change scales the gradient, compressing the past trajectory into a single memory-aware term that damps large local moves and reduces client drift in non-IID settings without extra communication or stored gradient histories. The server then aggregates client parameters by the usual data-weighted average. Setting $\alpha = 1$ recovers plain decaying-step SGD, since both $\Gamma(2-\alpha)$ and the power-law factor reduce to one.

$$
\begin{aligned}
\theta_t^{(k)} &\leftarrow \theta_t^{(k)} - \frac{\gamma_0}{\sqrt{t+1}} \cdot \frac{g_t^{(k)}}{\Gamma(2-\alpha)\,\bigl(\lVert \theta_t^{(k)} - \theta_{t-1}^{(k)} \rVert + \delta\bigr)^{1-\alpha}} \\
\theta_{t+1} &= \sum_{k \in S_t} \frac{n_k}{n}\, \theta_{t+1}^{(k)}
\end{aligned}
$$

where $\theta_t^{(k)}$ are the parameters of client $k$ at round $t$, $g_t^{(k)} = \nabla\ell(\theta_t^{(k)}; b)$ is the stochastic gradient on minibatch $b$, $\gamma_0$ is the initial learning rate, $\alpha \in (0,1]$ is the fractional order, $\Gamma$ is the Gamma function, $\delta > 0$ is a small stabilizing constant, $S_t$ is the set of clients sampled at round $t$, $n_k$ is the number of samples on client $k$, and $n = \sum_{k \in S_t} n_k$.

Reference: Mohammad Partohaghighi, Roummel Marcia, YangQuan Chen, "Fractional Order Federated Learning", arXiv 2026. https://arxiv.org/abs/2602.15380

---
[Back to the Canon](../README.md)
