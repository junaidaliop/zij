# LAGS-SGD

Implements LAGS-SGD, layer-wise adaptive gradient sparsification for communication-efficient distributed SGD.

In distributed data-parallel training, exchanging full gradients dominates the communication cost. LAGS-SGD sparsifies the gradient of each layer independently with a per-layer Top-$k$ operator, so layers with different sizes and communication-to-computation ratios use their own compression level, and the selection of a layer can be sent as soon as its backpropagation finishes (overlapping communication with computation).

To preserve accuracy under aggressive sparsification, each worker keeps a local residual (error feedback): the entries dropped by Top-$k$ are accumulated and re-injected in later iterations. The server averages the sparse contributions from all $P$ workers to update the parameters.

$$
\begin{aligned}
a_t^{p,(l)} &= \epsilon_{t-1}^{p,(l)} + \eta_{t-1}\, g_{t-1}^{p,(l)} \\
\epsilon_t^{p,(l)} &= a_t^{p,(l)} - \mathrm{TopK}\!\left(a_t^{p,(l)}, k^{(l)}\right) \\
\theta_t^{(l)} &= \theta_{t-1}^{(l)} - \frac{1}{P}\sum_{p=1}^{P}\mathrm{TopK}\!\left(a_t^{p,(l)}, k^{(l)}\right)
\end{aligned}
$$

where $l$ indexes layers and $p$ indexes the $P$ workers, $\theta^{(l)}$ are the layer-$l$ parameters, $\eta_t$ is the learning rate, $g_t^{p,(l)}$ is worker $p$'s stochastic gradient for layer $l$, $\epsilon_t^{p,(l)}$ is its accumulated residual, $a_t^{p,(l)}$ is the locally accumulated gradient, and $\mathrm{TopK}(x, k^{(l)})$ keeps the $k^{(l)}$ largest-magnitude entries of $x$ and zeros the rest.

Reference: Shaohuai Shi, Zhenheng Tang, Qiang Wang, Kaiyong Zhao, Xiaowen Chu, "Layer-wise Adaptive Gradient Sparsification for Distributed Deep Learning with Convergence Guarantees", arXiv 2019. https://arxiv.org/abs/1911.08727

---
[Back to the Canon](../index.md)
