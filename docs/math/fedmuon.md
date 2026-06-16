# FedMuon

Implements FedMuon, federated learning with matrix-orthogonalized local updates and global direction alignment.

FedMuon brings Muon's orthogonalized-momentum step into the federated setting. Each client runs Muon locally: it accumulates a momentum matrix $M_t$ from its stochastic gradient and orthogonalizes it via SVD (in practice a Newton-Schulz iteration), taking the polar factor $U V^\top$ as the search direction. To curb client drift under data heterogeneity, the local step blends this orthogonalized direction with a shared global direction $\Delta^g$, mixed by a coefficient $\alpha$. After $K$ local steps, the server averages the clients' parameter changes to advance the global model, recomputes $\Delta^g$ from those changes, and aggregates client momenta to warm-start the next round.

For client $i$ at round $r$, local step $k$:

$$
\begin{aligned}
M_{i}^{r,k} &= \beta\, M_{i}^{r,k-1} + g_{i}^{r,k} \\
U_{i}^{r,k}, \Sigma_{i}^{r,k}, V_{i}^{r,k} &= \mathrm{SVD}\!\left(M_{i}^{r,k}\right) \\
x_{i}^{r,k+1} &= x_{i}^{r,k} - \eta\left[(1-\alpha)\, U_{i}^{r,k}\, V_{i}^{r,k\top} + \alpha\, \Delta^{g,r}\right] \\
\Delta^{g,r+1} &= -\frac{1}{S K \eta} \sum_{i=1}^{S}\left(x_{i}^{r,K} - x_{i}^{r,0}\right) \\
x^{r+1} &= x^{r} + \frac{1}{S} \sum_{i=1}^{S}\left(x_{i}^{r,K} - x_{i}^{r,0}\right) \\
\bar{M}^{r+1} &= \frac{1}{S} \sum_{i=1}^{S} M_{i}^{r,K}
\end{aligned}
$$

where $x$ is the (matrix-shaped) parameter, $g_{i}^{r,k}$ the client's stochastic gradient, $M$ the momentum matrix, $\beta$ its decay, $\eta$ the learning rate, and $U_{i}^{r,k} V_{i}^{r,k\top}$ the orthogonalized search direction (the polar factor of $M$). $\Delta^{g,r}$ is the shared global direction, $\alpha$ the mixing coefficient between local and global directions ($\alpha = 0.5$ in the experiments), $S$ the number of participating clients, $K$ the local steps per round, and $\bar{M}^{r}$ the aggregated momentum that initializes each client's $M_{i}^{r,0}$ at the start of round $r$.

Reference: Junkang Liu, Fanhua Shang, Junchao Zhou, Hongying Liu, Yuanyuan Liu, Jin Liu, "FedMuon: Accelerating Federated Learning with Matrix Orthogonalization", arXiv 2025. https://arxiv.org/abs/2510.27403

---
[Back to the Canon](../index.md)
