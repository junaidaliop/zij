# pFedSOP

Implements pFedSOP, a personalized federated optimizer that takes a regularized rank-one Fisher (Newton) step on a Gompertz-blended personal gradient.

Each round, client $i$ holds its previous local pseudo-gradient $\Delta_{i}$ and receives the aggregated global pseudo-gradient $\Delta$ from the server. The angle $\phi$ between them is normalized through a Gompertz function into a weight $\beta\in[0,1]$, which blends the two into a personalized gradient $\Delta^{p}_i$: when local and global directions agree the client keeps more of its own update, and when they disagree it leans on the global one. This personalized gradient also defines a regularized rank-one Fisher information matrix $F_i=\Delta^{p}_i{\Delta^{p}_i}^{\top}+\rho I$ used as a cheap Hessian surrogate; the Newton step $F_i^{-1}\Delta^{p}_i$ is computed in $O(d)$ by the Sherman–Morrison formula without ever forming $F_i$. After this personalization step the client runs $\mathcal{T}$ inner SGD iterations, and the resulting parameter drift becomes its new pseudo-gradient, which the server averages over the participating clients.

$$
\begin{aligned}
\mathrm{sim} &= \frac{\Delta_{i}\cdot\Delta}{\lVert\Delta_{i}\rVert\,\lVert\Delta\rVert}, \qquad \phi = \arccos(\mathrm{sim}) \\
\beta &= 1 - e^{-e^{-\lambda(\phi-1)}} \\
\Delta^{p}_{i} &= (1-\beta)\,\Delta_{i} + \beta\,\Delta \\
F_i &= \Delta^{p}_{i}{\Delta^{p}_{i}}^{\top} + \rho I \\
\overline{\Delta}_{i} &= F_i^{-1}\Delta^{p}_{i} = \frac{\Delta^{p}_{i}}{\rho} - \frac{\Delta^{p}_{i}\,({\Delta^{p}_{i}}^{\top}\Delta^{p}_{i})}{\rho^2 + \rho\,{\Delta^{p}_{i}}^{\top}\Delta^{p}_{i}} \\
\theta_{i} &\leftarrow \theta_{i} - \eta_1\,\overline{\Delta}_{i} \\
\Delta_{i} &\leftarrow \frac{\theta_{i} - \theta_{i}^{(\mathcal{T})}}{\eta_2}, \qquad \Delta = \frac{1}{K'}\sum_{j=1}^{K'}\Delta_{j}
\end{aligned}
$$

where $\theta_i$ are client $i$'s personalized parameters, $\Delta_i$ its local pseudo-gradient (parameter drift) and $\Delta$ the server-averaged global pseudo-gradient over the $K'$ participating clients, $\mathrm{sim}$ and $\phi$ the cosine similarity and angle between them, $\beta$ the Gompertz weight with sharpness $\lambda>0$, $\Delta^{p}_i$ the personalized gradient, $F_i$ the regularized rank-one Fisher matrix with regularizer $\rho>0$, $\overline{\Delta}_i$ the Sherman–Morrison Newton step, $\eta_1$ the personalization learning rate, $\eta_2$ the inner SGD learning rate, and $\theta_i^{(\mathcal{T})}$ the parameters after $\mathcal{T}$ inner SGD steps.

Reference: Mrinmay Sen, Chalavadi Krishna Mohan, "pFedSOP: Accelerating Training Of Personalized Federated Learning Using Second-Order Optimization", arXiv 2025. https://arxiv.org/abs/2506.07159

---
[Back to the Canon](../README.md)
