# OpenDiLoCo

Implements OpenDiLoCo, an open reproduction of DiLoCo for low-communication distributed training.

OpenDiLoCo splits optimization into an inner and an outer loop. Each of the $N$ workers holds a replica of the model and runs $H$ local steps of an inner optimizer (AdamW) on its own data shard without any communication. After $H$ steps, every worker forms a pseudo-gradient equal to how far its replica drifted from the shared starting point. These pseudo-gradients are averaged across workers (an all-reduce, which can be done in FP16) and fed to an outer optimizer — SGD with Nesterov momentum — which produces the next shared parameters. Communication happens only once every $H$ inner steps, cutting bandwidth by orders of magnitude.

$$
\begin{aligned}
\theta_i^{(t,0)} &= \theta^{(t-1)} \\
\theta_i^{(t,h+1)} &= \mathrm{AdamW}\big(\theta_i^{(t,h)},\, g_i^{(t,h)}\big), \quad h = 0,\dots,H-1 \\
\Delta^{(t)} &= \frac{1}{N}\sum_{i=1}^{N}\big(\theta^{(t-1)} - \theta_i^{(t,H)}\big) \\
m^{(t)} &= \beta\, m^{(t-1)} + \Delta^{(t)} \\
\theta^{(t)} &= \theta^{(t-1)} - \eta\big(\Delta^{(t)} + \beta\, m^{(t)}\big)
\end{aligned}
$$

where $\theta^{(t)}$ are the shared (outer) parameters at outer step $t$, $\theta_i^{(t,h)}$ is worker $i$'s replica after $h$ inner steps, $g_i^{(t,h)}$ is its local loss gradient, $H$ is the number of inner steps, $N$ is the number of workers, $\Delta^{(t)}$ is the averaged pseudo-gradient, $m^{(t)}$ is the outer Nesterov momentum buffer, $\beta$ is the outer momentum ($\beta = 0.9$), and $\eta$ is the outer learning rate ($\eta = 0.7$).

Reference: Jaghouar, Ong, Hagemann, "OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training", 2024. https://arxiv.org/abs/2407.07852

---
[Back to the Canon](../README.md)
