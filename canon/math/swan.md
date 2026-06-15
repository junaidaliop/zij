# SWAN

Implements SWAN, a stateless optimizer that preprocesses the raw gradient with normalization and whitening in place of momentum and second-moment buffers.

SWAN (SGD with Whitening And Normalization) replaces Adam's stored statistics with two stateless operators applied to the current gradient matrix $G_t$ of a linear layer. GradNorm standardizes each row across the input dimension, mirroring the centering and variance control that momentum and the second moment provide over time. GradWhitening then multiplies by the inverse square root of the gradient Gram matrix (approximated with Newton-Schulz iterations), decorrelating the update directions. An optional rescaling restores the norm of the normalized gradient. Because no per-parameter state is kept, the optimizer's memory footprint matches plain SGD.

$$
\begin{aligned}
\tilde{G}_t &= \frac{G_t - \bar{g}\,\mathbf{1}_n^\top}{s\,\mathbf{1}_n^\top}, \quad \bar{g} = \frac{1}{n}\sum_{j=1}^{n} G_{t,:,j}, \quad s = \sqrt{\frac{1}{n}\sum_{j=1}^{n}\left(G_{t,:,j} - \bar{g}\right)^2} \\
\Delta W_t &= \left(\tilde{G}_t \tilde{G}_t^\top\right)^{-1/2} \tilde{G}_t \\
\Delta W_t &\leftarrow \frac{\lVert \tilde{G}_t \rVert}{\lVert \Delta W_t \rVert}\, \Delta W_t \\
\theta_{t+1} &= \theta_t - \eta\, \Delta W_t
\end{aligned}
$$

where $G_t$ is the gradient of an $m \times n$ weight matrix, $\bar{g}$ and $s$ are the row-wise mean and standard deviation across the $n$ input columns, $\mathbf{1}_n$ is the all-ones vector, $\tilde{G}_t$ is the normalized gradient, $(\tilde{G}_t \tilde{G}_t^\top)^{-1/2}$ is the matrix inverse square root computed via Newton-Schulz, the third line is the optional norm rescaling, and $\eta$ is the learning rate.

Reference: Chao Ma, Wenbo Gong, Meyer Scetbon, Edward Meeds, "SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training", arXiv 2024. https://arxiv.org/abs/2412.13148

---
[Back to the Canon](../README.md)
