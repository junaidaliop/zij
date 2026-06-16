# Opacus

Implements Opacus, the PyTorch DP-SGD mechanism that trains with $(\varepsilon, \delta)$-differential privacy.

Opacus makes an existing optimizer differentially private by intercepting the per-sample gradients. For each example in the lot it clips the $\ell_2$ norm of that example's gradient to a bound $C$, sums the clipped gradients, adds isotropic Gaussian noise scaled by the noise multiplier $\sigma$ and the clip bound, and averages over the lot. The resulting privatized gradient is then handed to the base optimizer (e.g. SGD), so the update below is the gradient that replaces $g_t$ in any wrapped optimizer.

$$
\begin{aligned}
\bar g_t(x_i) &= \frac{g_t(x_i)}{\max\!\left(1, \dfrac{\lVert g_t(x_i) \rVert_2}{C}\right)} \\
\tilde g_t &= \frac{1}{B}\left( \sum_{i=1}^{B} \bar g_t(x_i) + \mathcal{N}\!\left(0,\, \sigma^2 C^2 I\right) \right) \\
\theta_{t} &= \theta_{t-1} - \eta\, \tilde g_t
\end{aligned}
$$

where $g_t(x_i)=\nabla_\theta \mathcal{L}(\theta_{t-1}, x_i)$ is the per-sample gradient, $C$ is the per-sample clipping bound, $B$ is the lot size, $\sigma$ is the noise multiplier (noise standard deviation per coordinate is $\sigma C$), $\eta$ is the learning rate, and the privacy budget $(\varepsilon, \delta)$ is tracked by a Renyi differential privacy accountant over the training steps.

Reference: Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen, Sayan Ghosh, Akash Bharadwaj, Jessica Zhao, Graham Cormode, Ilya Mironov, "Opacus: User-Friendly Differential Privacy Library in PyTorch", arXiv 2021. https://arxiv.org/abs/2109.12298

---
[Back to the Canon](../index.md)
