# FedPAC

Implements FedPAC, a preconditioner alignment and correction framework for federated second-order optimization on non-IID data.

When clients run second-order optimizers locally, each learns a curvature-defined geometry (its preconditioner), and naively averaging models trained under incompatible geometries corrupts the global descent direction — a failure mode the paper calls preconditioner drift. FedPAC decouples parameter aggregation from geometry synchronization: it aligns local preconditioners into a shared global reference that warm-starts the next round, and corrects each local step by blending the local preconditioned gradient with an estimate of the global preconditioned direction. The preconditioner operator $\mathcal{P}$ is generic and is instantiated with SOAP, Sophia, or Muon.

For client $i$ at round $r$, local step $k$:

$$
\begin{aligned}
\tilde{g}_i^{r,k} &= \mathcal{P}_{\Theta_i^{r,k}}\!\left(g_i^{r,k}\right), \\
\theta_i^{r,k+1} &= \theta_i^{r,k} - \eta_l\left[(1-\beta)\,\tilde{g}_i^{r,k} + \beta\, g_G^{r}\right], \\
g_G^{r+1} &= -\frac{1}{SK\eta_l}\sum_{i=1}^{S}\left(\theta_i^{r,K} - \theta_i^{r,0}\right), \\
\theta^{r+1} &= \theta^{r} + \frac{1}{S}\sum_{i=1}^{S}\left(\theta_i^{r,K} - \theta_i^{r,0}\right), \\
\Theta^{r+1} &= \frac{1}{|\mathcal{S}_r|}\sum_{i \in \mathcal{S}_r}\Theta_i^{r,K}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_l$ the local learning rate, $g_i^{r,k}=\nabla F_i(\theta_i^{r,k};\xi_i^{r,k})$ the stochastic gradient, $\mathcal{P}_{\Theta}$ the preconditioner operator with state $\Theta$, $\tilde{g}$ the preconditioned gradient, $g_G$ the estimated global preconditioned direction, $\beta\in[0,1]$ the correction trade-off coefficient, $K$ the number of local steps, $S$ the number of participating clients, and $\mathcal{S}_r$ the client subset selected at round $r$.

Reference: Junkang Liu, Fanhua Shang, Hongying Liu, Jin Liu, Weixin An, Yuanyuan Liu, "Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data", ICML 2026. https://arxiv.org/abs/2602.19271

---
[Back to the Canon](../README.md)
