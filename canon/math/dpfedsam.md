# DP-FedSAM

Implements DP-FedSAM, a differentially private federated optimizer that flattens the loss landscape with Sharpness-Aware Minimization to absorb the performance cost of DP noise.

Per-sample gradient clipping and Gaussian noise injection in differentially private federated learning introduce a model inconsistency that sharpens the local loss landscape and degrades accuracy. DP-FedSAM runs SAM locally on each client so that updates land in flat minima that are robust to the clipping and noise, then clips and perturbs the aggregated client update before secure averaging on the server.

$$
\begin{aligned}
\delta(\theta^{t,k}) &= \rho\,\frac{g^{t,k}}{\lVert g^{t,k}\rVert_2}, \\
\theta^{t,k+1} &= \theta^{t,k} - \eta\,\nabla F_i\!\left(\theta^{t,k} + \delta(\theta^{t,k});\,\xi\right), \\
\Delta_i^{t} &= \theta_i^{t,K} - \theta^{t}, \\
\tilde{\Delta}_i^{t} &= \Delta_i^{t}\cdot\min\!\left(1,\ \frac{C}{\lVert \Delta_i^{t}\rVert_2}\right), \\
\hat{\Delta}_i^{t} &= \tilde{\Delta}_i^{t} + \mathcal{N}\!\left(0,\ \frac{\sigma^2 C^2}{m}\,\mathbf{I}_d\right), \\
\theta^{t+1} &= \theta^{t} + \frac{1}{m}\sum_{i\in W^t} \hat{\Delta}_i^{t}.
\end{aligned}
$$

where $\theta^{t,k}$ is client $i$'s local model at inner step $k$ of round $t$, $g^{t,k}=\nabla F_i(\theta^{t,k};\xi)$ is the local gradient, $\rho$ is the SAM perturbation radius, $\eta$ is the local learning rate, $\xi$ a sampled minibatch, $\Delta_i^t$ the accumulated local update after $K$ steps, $C$ the clipping threshold, $\sigma$ the noise multiplier, $m$ the number of sampled clients, $\mathbf{I}_d$ the $d$-dimensional identity, and $W^t$ the set of clients selected in round $t$.

Reference: Yifan Shi, Yingqi Liu, Kang Wei, Li Shen, Xueqian Wang, Dacheng Tao, "Make Landscape Flatter in Differentially Private Federated Learning", CVPR 2023. https://arxiv.org/abs/2303.11242

---
[Back to the Canon](../README.md)
