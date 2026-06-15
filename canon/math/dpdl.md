# DPDL

Implements DPDL, differentially private decentralized SGD that calibrates noisy cross-gradients by cosine similarity before a momentum update.

In decentralized learning over non-IID data, each agent $i$ shares gradients with its neighbors $\mathcal{N}_i$. DPDL protects them with the Gaussian mechanism: per-example gradients are clipped to norm $C$ and the batch sum is perturbed with noise before being sent. To counteract the resulting bias and the heterogeneity across agents, each received cross-gradient is weighted by a sigmoid of its cosine similarity to the agent's own self-gradient, so contributions aligned with the local descent direction are damped and divergent ones are up-weighted. The calibrated, aggregated gradient then drives a momentum-style update, after which models and velocities are mixed across neighbors via a doubly stochastic matrix $W$.

$$
\begin{aligned}
\hat{g}^{ji}_{t,b} &= g^{ji}_{t,b}\cdot\min\!\left\{1,\; C\,\|g^{ji}_{t,b}\|^{-1}\right\}, \\
\ddot{g}^{\,ji}_{t} &= \frac{1}{B}\Big(\textstyle\sum_{b}\hat{g}^{ji}_{t,b} + z^{ji}_{t}\Big), \qquad z^{ji}_{t}\sim\mathcal{N}(0,\sigma^{2}C^{2}I), \\
\mathcal{C}^{ij}_{t} &= \frac{1}{1+\exp\!\big(S(\ddot{g}^{\,ij}_{t},\hat{g}^{\,ii}_{t})\big)}, \qquad S(a,b)=\frac{\langle a,b\rangle}{\|a\|\,\|b\|}, \\
\tilde{g}^{\,ij}_{t} &= \frac{1}{\sqrt{w_{ij}}\,N}\,\ddot{g}^{\,ij}_{t} + \alpha\,w_{ij}\,\mathcal{C}^{ij}_{t}\,\hat{g}^{\,ii}_{t}, \qquad \tilde{g}^{\,i}_{t}=\sum_{j\in\mathcal{N}_i}\tilde{g}^{\,ij}_{t}, \\
\tilde{v}^{\,i}_{t} &= \beta\,v^{i}_{t-1} + \tilde{g}^{\,i}_{t}, \qquad \tilde{x}^{\,i}_{t} = x^{i}_{t-1} - \eta\,\tilde{v}^{\,i}_{t}, \\
v^{i}_{t} &= \sum_{j\in\mathcal{N}_i} w_{ij}\,\tilde{v}^{\,j}_{t}, \qquad x^{i}_{t} = \sum_{j\in\mathcal{N}_i} w_{ij}\,\tilde{x}^{\,j}_{t}.
\end{aligned}
$$

where $x^{i}_{t}$ is agent $i$'s model, $\eta$ the learning rate, $\beta$ the momentum coefficient, $g^{ji}_{t,b}$ the per-example cross-gradient of agent $j$'s data on $i$'s model in batch slot $b$, $C$ the clipping norm, $\sigma$ the noise multiplier, $B$ the batch size, $N$ the number of agents, $\hat{g}^{\,ii}_{t}$ the clipped self-gradient, $\mathcal{C}^{ij}_{t}$ the cosine-similarity calibration weight, $\alpha$ a scaling factor, and $w_{ij}$ the entries of the doubly stochastic mixing matrix $W$ over neighbors $\mathcal{N}_i$.

Reference: Yunsheng Yuan, Xue Xiao, Lina Wang, Feng Li, "DPDL: Towards Differential Privacy Preservation in Decentralized Stochastic Learning on Non-IID Data", arXiv 2026. https://arxiv.org/abs/2606.04399

---
[Back to the Canon](../README.md)
