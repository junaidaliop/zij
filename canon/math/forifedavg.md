# FO-RI-FedAvg

Implements FO-RI-FedAvg, a roughness-informed federated averaging method with a fractional-order coordinate-wise preconditioner.

Each client runs local SGD on a proximal-regularized objective, where the proximal pull toward the global weights $\theta_t$ is scaled by a roughness index $\mathcal{I}_k$ of the client's loss landscape. The gradient is then preconditioned element-wise by $p_t$, a fractional-order memory term built from the magnitude of the last local displacement raised to the power $1-\alpha$: large recent moves are damped and small ones amplified, with $\alpha=1$ recovering plain SGD ($p_t=\mathbf{1}$). After $H$ local steps the server forms the standard data-size-weighted average of the client weights.

$$
\begin{aligned}
p_t^{k} &= \frac{1}{\Gamma(2-\alpha)}\,\bigl(|\theta_t^{k}-\theta_{t-1}^{k}|+\delta\mathbf{1}\bigr)^{\odot(1-\alpha)} \\
g_t^{k} &= \nabla_\theta \ell(\theta_t^{k}; b_t^{k}) + \lambda_t\, r(\mathcal{I}_k)\,(\theta_t^{k}-\theta_t) \\
\theta_{t+1}^{k} &= \theta_t^{k} - \eta_t\,(g_t^{k}\odot p_t^{k}) \\
\theta_{t+1} &= \sum_{k\in S_t}\frac{n_k}{n_t}\,\theta_{t+1}^{k},\qquad r(\mathcal{I}_k)=\frac{\mathcal{I}_k}{\mathcal{I}_k+\tau_{\mathcal{I}}}
\end{aligned}
$$

where $\theta^k$ are client $k$'s parameters, $\theta_t$ the broadcast global weights, $\eta_t$ the local learning rate, $\lambda_t$ the base proximal strength, $\alpha\in(0,1]$ the fractional order, $\Gamma$ the Gamma function, $\delta>0$ a numerical stabilizer, $\odot$ element-wise operations, $\mathcal{I}_k$ the roughness index with saturation constant $\tau_{\mathcal{I}}$, $S_t$ the participating clients, and $n_k/n_t$ their data-proportional aggregation weights.

Reference: Mohammad Partohaghighi, Roummel Marcia, Bruce J. West, YangQuan Chen, "Fractional Order Federated Learning for Battery Electric Vehicle Energy Consumption Modeling", arXiv preprint 2026. https://arxiv.org/abs/2602.12567

---
[Back to the Canon](../README.md)
