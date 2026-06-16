# Decoupled DiLoCo

Implements Decoupled DiLoCo, an asynchronous DiLoCo variant that decouples learners from the synchronizer with quorum-based, token-weighted fragment merging for fault-tolerant distributed pre-training.

DiLoCo splits training across independent learners that each run many local inner-optimizer steps and only periodically exchange parameters; a central syncer turns the accumulated drift into a pseudo-gradient and applies an outer optimizer (SGD with Nesterov momentum). Decoupled DiLoCo removes the lock-step barrier: learners advance on their own clocks, and every synchronization period $H$ the syncer merges whatever fragments have arrived from the set $\mathcal{M}_t$ of available learners, requiring only a minimum quorum. Each learner's contribution is weighted by a token-based score (tokens processed times tokens-per-step efficiency), so the global model keeps progressing despite failed or straggling workers.

$$
\begin{aligned}
\theta_m^{(t_m)} &\leftarrow \mathrm{InnerOpt}\!\left(\theta_m^{(t_m-1)},\, \nabla \mathcal{L}\right) \\
w_{m,p} &= c_{m,p}^{\mathrm{tokens}} \cdot \frac{c_{m,p}^{\mathrm{tokens}}}{c_{m,p}^{\mathrm{steps}}} \\
\Delta_p^{(t)} &= \frac{\sum_{m \in \mathcal{M}_t} w_{m,p}\left(\Theta_p^{(t-H)} - \theta_{m,p}^{(t)}\right)}{\sum_{m \in \mathcal{M}_t} w_{m,p}} \\
\Theta_p^{(t)} &\leftarrow \mathrm{OuterOpt}\!\left(\Theta_p^{(t-H)},\, \Delta_p^{(t)}\right)
\end{aligned}
$$

where $\theta_m$ are the local parameters of learner $m$ stepped by $\mathrm{InnerOpt}$ (e.g. AdamW) on its data shard, $\Theta_p$ are the global parameters of fragment $p$ held by the syncer, $H$ is the synchronization period, $\mathcal{M}_t$ is the quorum of learners whose fragment $p$ is available at step $t$, $\Delta_p^{(t)}$ is the merged outer pseudo-gradient, $w_{m,p}$ is the token-weighted contribution of learner $m$ to fragment $p$ with $c_{m,p}^{\mathrm{tokens}}$ tokens processed and $c_{m,p}^{\mathrm{steps}}$ inner steps taken, and $\mathrm{OuterOpt}$ is SGD with Nesterov momentum applied to $\Delta_p^{(t)}$.

Reference: Arthur Douillard, Keith Rush, Yani Donchev, Zachary Charles, Nova Fallen, Ayush Dubey, Ionel Gog, Josef Dean, Blake Woodworth, Zachary Garrett, Nate Keating, Jenny Bishop, Henry Prior, Edouard Yvinec, Arthur Szlam, Marc'Aurelio Ranzato, Jeff Dean, "Decoupled DiLoCo for Resilient Distributed Pre-training", arXiv 2026. https://arxiv.org/abs/2604.21428

---
[Back to the Canon](../index.md)
