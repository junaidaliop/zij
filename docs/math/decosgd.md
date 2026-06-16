# DeCo-SGD

Implements DeCo-SGD, distributed SGD that jointly tunes gradient-compression ratio and update staleness for communication-efficient training.

DeCo-SGD builds on delayed-aggregation error-feedback SGD (DD-EF-SGD): each of the $n$ workers compresses a stale gradient together with its accumulated compression residual, the residual is updated with the part that was dropped, and the global parameters move along the averaged compressed messages. A controller periodically re-selects the staleness $\tau$ and the compression ratio $\delta$ from monitored bandwidth and compute conditions, trading communication volume against convergence speed.

$$
\begin{aligned}
\Delta_t^i &= C_\delta\!\left(e_{t-\tau}^i + g_{t-\tau}^i\right) \\
e_{t+1-\tau}^i &= e_{t-\tau}^i + g_{t-\tau}^i - \Delta_t^i \\
\theta_{t+1} &= \theta_t - \frac{\gamma}{n}\sum_{i=1}^{n}\Delta_t^i
\end{aligned}
$$

where $\theta_t$ are the global parameters, $\gamma$ is the learning rate, $g_{t-\tau}^i$ is worker $i$'s gradient delayed by staleness $\tau$, $e_t^i$ is its error-feedback residual, $C_\delta(\cdot)$ is a compression operator with ratio $\delta \in (0,1]$, and $\Delta_t^i$ is the compressed message averaged over the $n$ workers.

Reference: Rongwei Lu, Jingyan Jiang, Chunyang Li, Haotian Dong, Xingguang Wei, Delin Cai, Zhi Wang, "DeCo-SGD: Joint Optimization of Delay Staleness and Gradient Compression Ratio for Distributed SGD", arXiv preprint 2025. https://arxiv.org/abs/2507.17346

---
[Back to the Canon](../index.md)
