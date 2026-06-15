# DP-SGD

Implements DP-SGD, differentially private SGD via per-example gradient clipping and Gaussian noise.

To bound the influence of any single training example, each per-example gradient is clipped to a fixed $L_2$ norm $C$ before aggregation. Calibrated Gaussian noise with standard deviation $\sigma C$ is then added to the summed clipped gradients, and the result is averaged over the lot of size $L$. The descent step uses this noisy gradient, yielding a $(\varepsilon, \delta)$-differential privacy guarantee tracked across steps by the moments accountant.

$$
\begin{aligned}
g_t(x_i) &= \nabla_\theta \mathcal{L}(\theta_t, x_i) \\
\bar{g}_t(x_i) &= g_t(x_i) \Big/ \max\!\left(1, \frac{\lVert g_t(x_i) \rVert_2}{C}\right) \\
\tilde{g}_t &= \frac{1}{L}\left( \sum_i \bar{g}_t(x_i) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right) \\
\theta_{t+1} &= \theta_t - \eta_t\, \tilde{g}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t(x_i)$ the gradient on example $x_i$ in the lot, $C$ the clipping bound, $\sigma$ the noise multiplier, $L$ the lot size, and $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ the spherical Gaussian noise.

Reference: Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang, "Deep Learning with Differential Privacy", ACM CCS 2016. https://arxiv.org/abs/1607.00133

---
[Back to the Canon](../README.md)
