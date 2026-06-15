# MicroAdam

Implements MicroAdam, a memory-efficient Adam variant that compresses gradients via top-$k$ sparsification with quantized error feedback.

MicroAdam reduces optimizer state by never storing dense first- and second-moment buffers. Instead it keeps a sliding window of the last $m$ sparse gradients (only their top-$k$ indices and values) and reconstructs the Adam moments on the fly from this window. To avoid losing the discarded coordinates, the residual after sparsification is fed back through a low-bit quantized error-feedback buffer $e_t$, which is dequantized and re-added to the next gradient. This yields provable convergence while cutting the memory footprint to a small fraction of standard Adam.

At step $t$ the dequantized error is added to the gradient, the top-$k$ components are extracted and stored in the window, and the new residual is requantized:

$$
\begin{aligned}
a_t &= g_t + Q^{-1}(e_t, \delta_t, \Delta_t), \\
(\mathcal{I}_t, \mathcal{V}_t) &= \mathrm{TopK}(|a_t|), \\
a_t[\mathcal{I}_t] &\leftarrow 0, \\
e_{t+1} &= Q(a_t, \delta_{t+1}, \Delta_{t+1}), \\
m_t &= \frac{1}{1-\beta_1^t} \sum_{i} \beta_1^{\,r_i}\, \mathcal{V}_i, \qquad
v_t = \frac{1}{1-\beta_2^t} \sum_{i} \beta_2^{\,r_i}\, \mathcal{V}_i^2, \\
\theta_{t+1} &= \theta_t - \gamma\, \frac{m_t}{\epsilon + \sqrt{v_t}},
\end{aligned}
$$

where $g_t$ is the gradient, $Q$ and $Q^{-1}$ are symmetric uniform $b$-bit quantization and dequantization with per-block range $[\delta,\Delta]$, $\mathrm{TopK}$ keeps the $k$ largest-magnitude coordinates, the sums run over the $\min(t,m)$ gradients in the sliding window with $r_i$ the age (in steps) of stored entry $i$, $\mathcal{V}_i$ are its sparse values placed at indices $\mathcal{I}_i$, $\gamma$ is the learning rate, $\beta_1,\beta_2$ are the moment decay rates, and $\epsilon$ is the stability constant.

Reference: Ionut-Vlad Modoranu, Mher Safaryan, Grigory Malinovsky, Eldar Kurtic, Thomas Robert, Peter Richtárik, Dan Alistarh, "MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence", NeurIPS 2024. https://arxiv.org/abs/2405.15593

---
[Back to the Canon](../README.md)
