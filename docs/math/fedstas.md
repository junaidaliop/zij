# FedSTaS

Implements FedSTaS, a privacy-preserving client- and data-level sampling scheme for federated learning.

FedSTaS stratifies clients by their compressed gradients, then re-allocates how many clients to draw from each stratum using optimal Neyman allocation, sampling more heavily from strata with higher gradient variance. Within each round, clients are drawn proportionally to their gradient norms, and a privacy-preserving estimate of the total participating data size is used to set per-observation subsampling probabilities. Selected clients run local SGD on their subsampled data, and the server forms an unbiased estimate of the full-participation update via a stratified average.

For stratum $h$ with $N_h$ clients and gradient variance $S_h$, the number of sampled clients $m_h$, the within-stratum client probabilities $p_t^k$, the local update, and the global aggregation are

$$
\begin{aligned}
m_h &= m \cdot \frac{N_h S_h}{\sum_{h=1}^{H} N_h S_h}, \\
p_t^k &= \frac{\lVert IS(G_t^k) \rVert}{\sum_{k=1}^{N_h} \lVert IS(G_t^k) \rVert}, \\
w_{t+i+1}^k &= w_{t+i}^k - \eta\, \nabla F_k\!\left(w_{t+i}^k, \xi_t^k\right), \\
w_{t+1} &= \frac{1}{N} \sum_{h=1}^{H} N_h \cdot \frac{1}{m_h} \sum_{k=1}^{m_h} w_{t+1}^k.
\end{aligned}
$$

where $H$ is the number of strata, $m$ the total client budget, $N = \sum_h N_h$ the total client count, $IS(G_t^k)$ the compressed (information-squeezed) gradient of client $k$ at round $t$, $\eta$ the local learning rate, $F_k$ client $k$'s local objective, $\xi_t^k$ its subsampled data batch drawn with per-observation probability $\eta/\tilde{n}$ (with $\tilde{n}$ the privately estimated participating data size), and $w_{t+1}^k$ the locally updated model returned to the server.

Reference: Jordan Slessor, Dezheng Kong, Xiaofen Tang, Zheng En Than, Linglong Kong, "FedSTaS: Client Stratification and Client Level Sampling for Efficient Federated Learning", arXiv preprint 2024. https://arxiv.org/abs/2412.14226

---
[Back to the Canon](../index.md)
