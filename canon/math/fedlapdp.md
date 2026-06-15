# FedLAP-DP

Implements FedLAP-DP, federated learning in which clients share differentially private synthetic samples that approximate their local loss landscapes.

Instead of transmitting raw gradients, each client $k$ synthesizes a small set $\mathcal{S}_k$ of samples by gradient matching: along a short local trajectory it forces the gradient produced by $\mathcal{S}_k$ to track the gradient produced by the real data $\mathcal{D}_k$, but only within a trust region of radius $r$ around the starting weights, where the local quadratic approximation of the loss stays accurate. Differential privacy is obtained by clipping each per-sample real gradient to norm $C$ and adding Gaussian noise before it drives the synthetic optimization. The server then collects every $\mathcal{S}_k$ and descends the reconstructed global loss landscape by ordinary gradient steps weighted by client data sizes, staying inside the smallest client trust region.

$$
\begin{aligned}
\tilde{g}^{\mathcal{D}}(x_k^i) &= g^{\mathcal{D}}(x_k^i)\cdot\min\!\left(1,\ \frac{C}{\lVert g^{\mathcal{D}}(x_k^i)\rVert_2}\right) \\
\tilde{\nabla}\mathcal{L}(w_k^{m,t},\mathcal{D}_k) &= \frac{1}{B}\sum_{i=1}^{B}\Big(\tilde{g}^{\mathcal{D}}(x_k^i) + \mathcal{N}(0,\sigma^2 C^2 I)\Big) \\
\mathcal{S}_k &= \arg\min_{\mathcal{S}_k}\ \sum_{t=1}^{T}\ \mathcal{L}_{\mathrm{dis}}\!\Big(\tilde{\nabla}\mathcal{L}(w_k^{m,t},\mathcal{D}_k),\ \nabla\mathcal{L}(w_k^{m,t},\mathcal{S}_k)\Big), \quad \lVert w_k^{m,t}-w_k^{m,1}\rVert < r \\
w_g^{m,t+1} &= w_g^{m,t} - \eta\sum_{k=1}^{K}\frac{N_k}{N}\,\nabla_w \mathcal{L}(w_g^{m,t},\mathcal{S}_k), \quad \lVert w_g^{m,t}-w_g^{m,1}\rVert \le \min_k r_k
\end{aligned}
$$

where $w_g$ are the global parameters, $w_k$ the client parameters, $\eta$ the learning rate, $g^{\mathcal{D}}(x_k^i)$ the gradient on real sample $x_k^i$, $\tilde{g}^{\mathcal{D}}$ its clipped version, $C$ the clipping bound, $\sigma$ the noise multiplier, $B$ the batch size, $\mathcal{S}_k$ the learned synthetic set, $\mathcal{L}_{\mathrm{dis}}$ the layer-wise cosine-distance gradient-matching loss, $r$ (and per-client $r_k$) the trust-region radius, $N_k$ the number of samples at client $k$ with $N=\sum_k N_k$, $m$ the communication round, and $t$ the local step index.

Reference: Hui-Po Wang, Dingfan Chen, Raouf Kerkouche, Mario Fritz, "FedLAP-DP: Federated Learning by Sharing Differentially Private Loss Approximations", PoPETs 2024. https://arxiv.org/abs/2302.01068

---
[Back to the Canon](../README.md)
