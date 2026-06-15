# 0/1 Adam

Implements 0/1 Adam, a communication-efficient distributed Adam that freezes the variance and 1-bit-compresses the momentum.

0/1 Adam targets the communication cost of distributed Adam. Its two ideas are *adaptive variance freezing* — the second moment $v_t$ is refreshed only on a chosen schedule $\mathcal{T}_v$ and held stale otherwise — and *1-bit synchronization* — each node accumulates its locally scaled momentum into an auxiliary buffer $u_t$ and only synchronizes it across nodes at the steps in $\mathcal{T}_u$, where it is exchanged through 1-bit-compressed AllReduce.

Between synchronization points every node runs cheap local Adam-style steps using the frozen variance, so the only communication is a 1-bit reduction of the buffer (and a full-precision variance refresh on the sparse schedule). On node $i$ the per-step update is:

$$
\begin{aligned}
m_{t+1/2}^{(i)} &= \beta_1\, m_t^{(i)} + (1-\beta_1)\, g_t^{(i)} \\
\theta_{t+1/2}^{(i)} &= \theta_t^{(i)} - \gamma_t\, \frac{m_t^{(i)}}{\sqrt{v_t} + \epsilon} \\
u_{t+1/2}^{(i)} &= u_t^{(i)} + \gamma_t\, m_t^{(i)} \\
v_{t+1} &= \begin{cases} \beta_2\, v_t + (1-\beta_2)\, \bar{g}_t^{\,2} & t \in \mathcal{T}_v \\ v_t & t \notin \mathcal{T}_v \end{cases}
\end{aligned}
$$

where $\theta$ are the model parameters, $\gamma_t$ the learning rate, $g_t^{(i)}$ the local gradient, $\bar{g}_t$ the AllReduced gradient, $m_t$/$v_t$ the first and second moments, $u_t$ the auxiliary momentum buffer, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ the stability constant. At each $t \in \mathcal{T}_u$ the buffer is averaged across nodes via 1-bit AllReduce into $\bar{u}_{t+1/2}$, the momentum is re-estimated as $m_{t+1}^{(i)} = \bar{u}_{t+1/2} / \sum_{h=t'}^{t}\gamma_h$, the parameters are corrected to $\theta_{t+1}^{(i)} = \theta_{t'}^{(i)} - \bar{u}_{t+1/2}/(\sqrt{v_t}+\epsilon)$ from the last sync step $t'$, and the buffer is reset.

Reference: Yucheng Lu, Conglong Li, Minjia Zhang, Christopher De Sa, Yuxiong He, "Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam", arXiv 2022. https://arxiv.org/abs/2202.06009

---
[Back to the Canon](../README.md)
