# ANSGD

Implements ANSGD, a noisy projected SGD for learning across data owners under joint differential privacy.

Parameters split into per-owner personalized components $x_j$ and a globally shared component $u$. At each step the optimizer samples one owner $j_t$ and one of its records $z_{i_t,j_t}$, takes a projected gradient step on the corresponding loss, and injects Gaussian noise into the shared coordinate only. Because the personalized blocks stay noise-free, privacy is spent solely on $u$, and the returned model is the iterate average.

$$
\begin{aligned}
b_t &\sim \mathcal{N}(0,\sigma^2 I_{\mathcal{U}}), \qquad \sigma^2 = \frac{L^2 T \log(1/\delta)}{\varepsilon^2 m^2 n^2}, \\
\big(x_{j_t}^{t+1}, u^{t+1}\big) &\leftarrow \Pi_{\mathcal{X}\times\mathcal{U}}\!\Big(\big(x_{j_t}^{t}, u^{t}\big) - \eta\,\big(\nabla h(x_{j_t}^{t}, u^{t}, z_{i_t,j_t}) + [\,0,\; b_t\,]\big)\Big), \\
(\bar{x}, \bar{u}) &= \frac{1}{T}\sum_{t=0}^{T-1}\big(x^{t}, u^{t}\big).
\end{aligned}
$$

where $\eta$ is the learning rate, $T$ the number of iterations, $h$ the $L$-Lipschitz loss, $g_t = \nabla h(\cdot)$ the sampled gradient, $\Pi_{\mathcal{X}\times\mathcal{U}}$ the Euclidean projection onto the convex parameter sets, $b_t$ the noise added to the shared block $u$, $(\varepsilon,\delta)$ the privacy budget, $m$ the number of owners, and $n$ the number of records per owner.

Reference: Yangsibo Huang, Haotian Jiang, Daogao Liu, Mohammad Mahdian, Jieming Mao, Vahab Mirrokni, "Learning across Data Owners with Joint Differential Privacy", arXiv 2023. https://arxiv.org/abs/2305.15723

---
[Back to the Canon](../README.md)
