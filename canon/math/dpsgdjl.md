# DP-SGD-JL

Implements DP-SGD-JL, differentially private SGD that approximates per-example gradient norms with Johnson-Lindenstrauss projections.

Standard DP-SGD must clip each example's gradient to bound its sensitivity, which requires materializing per-example gradients and is slow and memory hungry. DP-SGD-JL avoids this by estimating each per-example gradient norm from a handful of random Jacobian-vector products. Drawing $r$ Gaussian probe vectors $v_1,\dots,v_r$, the projection $P_{ij}=\langle\nabla_\theta\mathcal{L}(\theta;X_i),v_j\rangle$ gives an unbiased norm estimate $M_i$, so the clip factor $\min\{1,C/M_i\}$ can be applied as a scalar weight on each example's loss before a single batch backward pass. Gaussian noise is then added to the clipped, averaged gradient.

$$
\begin{aligned}
P_{ij} &= \langle \nabla_\theta \mathcal{L}(\theta_{t-1};X_i),\, v_j \rangle, \quad v_j \sim \mathcal{N}(0, I_d), \\
M_i &= \sqrt{\tfrac{1}{r}\sum_{j=1}^{r} P_{ij}^2}, \\
\tilde{\mathcal{L}}(\theta) &= \tfrac{1}{B}\sum_{i\in B}\min\Big\{1,\ \tfrac{C}{M_i}\Big\}\,\mathcal{L}(\theta;X_i), \\
\tilde{g}_t &= \nabla_\theta \tilde{\mathcal{L}}(\theta_{t-1}) + \tfrac{\sigma C}{B}\,\mathcal{N}(0, I_d), \\
\theta_t &= \theta_{t-1} - \eta_t\, \tilde{g}_t.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $\mathcal{L}(\theta;X_i)$ the loss on example $X_i$, $r$ the number of JL probe vectors, $M_i$ the estimated per-example gradient norm, $C$ the clipping threshold, $B$ the batch size, $\sigma$ the noise multiplier, and $d$ the parameter dimension.

Reference: Zhiqi Bu, Sivakanth Gopi, Janardhan Kulkarni, Yin Tat Lee, Judy Hanwen Shen, Uthaipon Tantipongpipat, "Fast and Memory Efficient Differentially Private-SGD via JL Projections", NeurIPS 2021. https://arxiv.org/abs/2102.03013

---
[Back to the Canon](../README.md)
