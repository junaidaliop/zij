# AdaBFE

Implements AdaBFE, a learning-rate-free optimizer that adapts a per-dimension step size by measuring the angle between successive gradients.

AdaBFE (Adaptive Binary Forward Exploration) probes each dimension independently. It takes a tentative step, recomputes the gradient there, and forms an angular change metric between the old and new gradient. If the gradients have turned too sharply (metric at or above a threshold), the step overshot and the learning rate is halved; otherwise the step was conservative and the rate is doubled. A binary search on $\eta_i$ thus drives the per-coordinate step toward the largest move that keeps the gradient direction stable, removing the need to tune a fixed learning rate.

$$
\begin{aligned}
g_{t,i} &= \frac{\partial f(\theta)}{\partial \theta_{t,i}} \\
\theta^{*}_{t,i} &= \theta_{t,i} - \eta_i\, g_{t,i} \\
g^{*}_{t,i} &= \frac{\partial f(\theta)}{\partial \theta^{*}_{t,i}} \\
\epsilon_{c,i} &= \arctan\!\left(\left|\frac{g^{*}_{t,i} - g_{t,i}}{1 + g^{*}_{t,i}\, g_{t,i}}\right|\right) \\
\eta_i &\leftarrow \begin{cases} \tfrac{1}{2}\,\eta_i & \text{if } \epsilon_{c,i} \ge \epsilon_{v,i} \\ 2\,\eta_i & \text{if } \epsilon_{c,i} < \epsilon_{v,i} \end{cases} \\
\theta_{t+1,i} &= \theta^{*}_{t,i}
\end{aligned}
$$

where $\theta_{t,i}$ is the $i$-th coordinate of the parameters at step $t$, $\eta_i$ its per-dimension learning rate, $g_{t,i}$ and $g^{*}_{t,i}$ the gradients before and after the tentative step, $\epsilon_{c,i}$ the measured angle of gradient change, and $\epsilon_{v,i}$ a target threshold angle. The arctan expression is the angle between the two consecutive gradient values.

Reference: Xin Cao, "BFE and AdaBFE: A New Approach in Learning Rate Automation for Stochastic Optimization", arXiv 2022. https://arxiv.org/abs/2207.02763

---
[Back to the Canon](../index.md)
