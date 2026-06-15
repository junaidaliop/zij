# DiZO

Implements DiZO, a divergence-driven zeroth-order fine-tuning method that rescales the cumulative update layer by layer.

DiZO targets the slow convergence of memory-efficient zeroth-order (ZO) fine-tuning, where the gradient is estimated by finite differences of forward passes and every layer receives a uniform-magnitude step. Its core idea is *divergence-driven layer adaptation*: after running ordinary ZO-SGD steps, the displacement of each layer from the pretrained weights is rescaled by a learned per-layer projection so that updates carry diverse magnitudes matched to each layer's needs, rather than the single global scale that standard ZO imposes.

The base optimization uses the SPSA-style two-sided estimator and a plain SGD step. Periodically, the cumulative displacement $\Delta\theta_t = \theta_t - \theta_0$ from the pretrained model $\theta_0$ is reshaped per layer $\ell$ by a projection scalar $\gamma_t^{(\ell)}$, which is itself optimized with ZO:

$$
\begin{aligned}
\widehat{\nabla}\mathcal{L}(\theta_t) &= \frac{1}{q}\sum_{i=1}^{q}\frac{\mathcal{L}(\theta_t+\epsilon u_i)-\mathcal{L}(\theta_t-\epsilon u_i)}{2\epsilon}\,u_i, \qquad u_i \sim \mathcal{N}(0,I) \\
\theta_t &= \theta_{t-1} - \eta\,\widehat{\nabla}\mathcal{L}(\theta_{t-1}) \\
\gamma_t^{*} &= \arg\min_{\gamma_t}\ \mathcal{L}\!\left(\theta_0 + \frac{\gamma_t}{\lVert\Delta\theta_t\rVert}\,\Delta\theta_t\right) \\
\theta_t^{(\ell)} &= \theta_0^{(\ell)} + \frac{\gamma_t^{(\ell)*}}{\lVert\Delta\theta_t^{(\ell)}\rVert}\,\Delta\theta_t^{(\ell)}, \qquad \frac{\gamma_t^{(\ell)}}{\lVert\Delta\theta_t^{(\ell)}\rVert}\in[1-\tau,\,1+\tau]
\end{aligned}
$$

where $\theta$ are the parameters, $\theta_0$ the pretrained weights, $\eta$ the learning rate, $\epsilon$ the perturbation scale, $u_i$ random Gaussian directions, $q$ the number of queries, $\ell$ the layer index, $\Delta\theta_t^{(\ell)}=\theta_t^{(\ell)}-\theta_0^{(\ell)}$ the per-layer displacement, and $\gamma_t^{(\ell)}$ the per-layer projection scalar (re-initialized to $\lVert\Delta\theta_t^{(\ell)}\rVert$ so the initial ratio is $1$, then optimized by the same ZO estimator and SGD step, with the ratio clipped to $[1-\tau,1+\tau]$).

Reference: Qitao Tan, Jun Liu, Zheng Zhan, Caiwen Ding, Yanzhi Wang, Jin Lu, Geng Yuan, "Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning", ICML 2025. https://arxiv.org/abs/2502.03304

---
[Back to the Canon](../README.md)
