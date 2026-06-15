# DiLoCo

Implements DiLoCo, distributed low-communication training that alternates many local inner steps with an outer momentum step on the averaged parameter drift.

DiLoCo trains a model on $k$ poorly connected workers that each hold a data shard. Starting from the shared parameters, every worker runs $H$ steps of an inner optimizer (AdamW in the paper) on its own data without communicating. After the inner phase, the per-worker parameter drifts are averaged into an outer gradient $\Delta^{(t)}$, and a single step of an outer optimizer updates the global parameters. With Nesterov momentum as the outer optimizer, communication happens only once every $H$ steps yet matches fully synchronous data-parallel training.

$$
\begin{aligned}
\theta_i^{(t)} &\leftarrow \theta^{(t-1)} \quad\text{then for } h = 1,\dots,H:\ \theta_i^{(t)} \leftarrow \mathrm{InnerOpt}\!\left(\theta_i^{(t)},\, \nabla \mathcal{L}(x;\theta_i^{(t)})\right),\quad x \sim \mathcal{D}_i \\
\Delta^{(t)} &\leftarrow \frac{1}{k}\sum_{i=1}^{k}\left(\theta^{(t-1)} - \theta_i^{(t)}\right) \\
m_t &\leftarrow \beta\, m_{t-1} + \Delta^{(t)} \\
\theta^{(t)} &\leftarrow \theta^{(t-1)} - \gamma\left(\beta\, m_t + \Delta^{(t)}\right)
\end{aligned}
$$

where $\theta^{(t)}$ are the global parameters after outer step $t$, $\theta_i^{(t)}$ is worker $i$'s local copy, $\mathcal{D}_i$ is worker $i$'s data shard, $\mathrm{InnerOpt}$ is the inner optimizer (AdamW), $H$ is the number of inner steps, $k$ is the number of workers, $\Delta^{(t)}$ is the averaged outer gradient (the mean parameter drift), $\gamma$ is the outer learning rate, $\beta$ is the outer momentum, and $m_t$ is the outer momentum buffer. The outer optimizer is Nesterov momentum; the paper specifies it but does not write the buffer equations inline, so the last two lines give the standard Nesterov form (Sutskever et al., 2013) applied to $\Delta^{(t)}$.

Reference: Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen, "DiLoCo: Distributed Low-Communication Training of Language Models", arXiv 2023. https://arxiv.org/abs/2311.08105

---
[Back to the Canon](../README.md)
