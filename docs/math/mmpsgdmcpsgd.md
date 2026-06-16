# MM-PSGD / MC-PSGD

Implements MM-PSGD and MC-PSGD, periodic-averaging SGD variants for distributed learning over block-cyclic data.

In block-cyclic federated training the data distribution shifts cyclically across blocks of clients, so a single global model is suboptimal for any given block. Both methods run periodic stochastic gradient descent: each of $N$ clients takes $I$ local SGD steps, after which the models are averaged into a global model $\bar{x}_t$ and broadcast back. Instead of returning one model, each method maintains a block-specific predictor $\tilde{x}_m$ formed as the running average of the global models produced during the rounds belonging to block $m$.

MM-PSGD (Multiple-Model PSGD) keeps a single chain and one predictor per block. MC-PSGD (Multiple-Chain PSGD) additionally runs a second, block-separate chain $y$ with its own learning rate $\eta$ that trains only on the current block's data, and at each communication round it selects whichever interim model ($\bar{x}_t$ or $\bar{y}_t$) has the smaller average loss to update the predictor.

$$
\begin{aligned}
x_{t+1}^{i} &= x_t^{i} - \gamma\, g_t^{i}, \qquad y_{t+1}^{i} = y_t^{i} - \eta\, G_t^{i} \\
\bar{x}_t &= \frac{1}{N}\sum_{i=1}^{N} x_t^{i}, \qquad \bar{y}_t = \frac{1}{N}\sum_{i=1}^{N} y_t^{i} \quad (\text{every } I \text{ steps}) \\
\bar{u}_t &= \arg\min_{u \in \{\bar{x}_t,\, \bar{y}_t\}} \tfrac{1}{N}\sum_{i=1}^{N} \ell^{i}(u) \\
\tilde{x}_m &\leftarrow \frac{r}{r+1}\,\tilde{x}_m + \frac{1}{r+1}\,\bar{u}_t, \qquad x_t^{i} \leftarrow \bar{x}_t
\end{aligned}
$$

where $x_t^i$ is client $i$'s model on the block-mixed chain and $y_t^i$ on the block-separate chain, $g_t^i,G_t^i$ are their stochastic gradients, $\gamma,\eta$ are the two learning rates, $I$ is the local-step period between communications, $\bar{x}_t,\bar{y}_t$ are the averaged global models, $\ell^i$ is client $i$'s local loss, $\tilde{x}_m$ is the predictor for block $m$, and $r$ is the number of completed rounds in block $m$. MM-PSGD is the special case using only the $x$ chain, so $\bar{u}_t=\bar{x}_t$.

Reference: Yucheng Ding, Chaoyue Niu, Yikai Yan, Zhenzhe Zheng, Fan Wu, Guihai Chen, Shaojie Tang, Rongfei Jia, "Distributed Optimization over Block-Cyclic Data", ICML 2020. https://arxiv.org/abs/2002.07454

---
[Back to the Canon](../index.md)
