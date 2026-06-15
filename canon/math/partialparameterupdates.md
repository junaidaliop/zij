# Partial Parameter Updates

Implements Partial Parameter Updates, a memory- and compute-efficient distributed scheme where each worker trains only a fixed slice of the model.

The method follows the local-SGD / DiLoCo structure: $K$ workers run $H$ inner steps in parallel, then a single outer step aggregates their progress. The twist is that worker $k$ is assigned a fixed index set $I_k^{\text{train}}$ and computes gradients, allocates optimizer state, and updates parameters only within that set, freezing the rest. This shrinks the backward pass and optimizer memory while the union of the worker slices still covers the whole model. After the inner phase, each worker reports its parameter change as a pseudo-gradient, the changes are averaged per coordinate over only the workers that touched that coordinate, and an outer optimizer applies the result.

$$
\begin{aligned}
g_k^{(t,h)}[i] &= \begin{cases} \nabla_{\theta[i]}\, \mathcal{L}\!\left(\theta_k^{(t,h)}; X_k^{(t,h)}\right), & i \in I_k^{\text{train}} \\ 0, & \text{otherwise} \end{cases} \\
\theta_k^{(t,h+1)}\!\left[I_k^{\text{train}}\right] &= \mathrm{InnerOpt}\!\left(\theta_k^{(t,h)}\!\left[I_k^{\text{train}}\right],\, g_k^{(t,h)}\!\left[I_k^{\text{train}}\right]\right) \\
\Delta_k^{(t)}[i] &= \begin{cases} \theta_k^{(t,H)}[i] - \theta^{(t)}[i], & i \in I_k^{\text{train}} \\ 0, & \text{otherwise} \end{cases} \\
\Delta^{(t)}[i] &= \frac{1}{m[i]} \sum_{k=1}^{K} \Delta_k^{(t)}[i] \\
\theta^{(t+1)} &= \mathrm{OuterOpt}\!\left(\theta^{(t)},\, -\Delta^{(t)}\right)
\end{aligned}
$$

where $t$ indexes outer rounds, $h = 0,\dots,H-1$ inner steps, $k$ the worker, and $i$ a parameter coordinate; $I_k^{\text{train}}$ is the fixed slice worker $k$ trains; $m[i] = |\{k : i \in I_k^{\text{train}}\}|$ counts how many workers update coordinate $i$; $\Delta_k^{(t)}$ is worker $k$'s parameter change after $H$ steps; and $\mathrm{InnerOpt}$, $\mathrm{OuterOpt}$ are the local and global optimizers (AdamW and Nesterov SGD in the experiments), the latter consuming $-\Delta^{(t)}$ as a pseudo-gradient.

Reference: Filippova, Katharopoulos, Grangier, Collobert, "Partial Parameter Updates for Efficient Distributed Training", arXiv 2025. https://arxiv.org/abs/2509.22418

---
[Back to the Canon](../README.md)
