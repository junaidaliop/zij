# DC-SGD

Implements DC-SGD, DP-SGD with a dynamically adjusted clipping threshold driven by a differentially private estimate of the gradient-norm distribution.

DC-SGD keeps the standard DP-SGD step but removes the brittle, manually tuned clipping bound $C$. Each round it builds a noisy histogram of per-sample gradient norms (sensitivity 1, perturbed with noise multiplier $\sigma_H$) and uses it to pick the next threshold. Two rules are offered: DC-SGD-P sets $C$ at the $p$-th percentile of the estimated norm distribution, while DC-SGD-E chooses, over a candidate grid, the $C$ minimizing an expected squared error that trades off injected-noise variance against clipping bias. The total privacy cost is unchanged because the overall multiplier $\sigma$ is split across the histogram and training queries.

$$
\begin{aligned}
\mathrm{Clip}(g_{t,i}, C_t) &= g_{t,i} \,/\, \max\!\Big(1, \tfrac{\lVert g_{t,i}\rVert_2}{C_t}\Big) \\
\theta_{t+1} &= \theta_t - \frac{\eta}{B}\Big( \sum_{i \in B_t} \mathrm{Clip}(g_{t,i}, C_t) + \mathcal{N}(0, \sigma_T^2 C_t^2 I) \Big) \\
\sigma_T &= \big(\sigma^{-2} - \sigma_H^{-2}\big)^{-1/2} \\
C_{t+1}^{\mathrm{P}} &= \mathrm{quantile}_p\big(\tilde{H}_t\big) \\
C_{t+1}^{\mathrm{E}} &= \arg\min_{C}\; \frac{\sigma_T^2\, C^2 d}{B^2} + \frac{1}{|B_t|}\sum_{j \in B_t}\max\!\big(\lVert g_{t,j}\rVert - C,\, 0\big)^2
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_{t,i}$ the per-sample gradient, $C_t$ the clipping threshold, $B$ the expected batch size and $|B_t|$ the actual one, $d$ the gradient dimension, $\sigma$ the overall noise multiplier split into the training multiplier $\sigma_T$ and the histogram multiplier $\sigma_H$, $\tilde{H}_t$ the differentially private gradient-norm histogram, and $\mathrm{quantile}_p$ the bin whose accumulated mass first exceeds the fraction $p$.

Reference: Chengkun Wei, Weixian Li, Chen Gong, Wenzhi Chen, "DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation", arXiv 2025. https://arxiv.org/abs/2503.22988

---
[Back to the Canon](../index.md)
