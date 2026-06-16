# SlimAdam

Implements SlimAdam, a memory-efficient Adam that compresses the second moment by averaging it along selected tensor dimensions.

Adam stores a full per-coordinate second moment $v_t$, doubling the optimizer state. SlimAdam keeps the first moment and the Adam update unchanged, but replaces the second-moment accumulator with the mean of squared gradients taken over a chosen set of dimensions $K$ of the parameter tensor. The single compressed value is then broadcast back across those dimensions in the per-coordinate division, so $v_t$ shrinks by the size of the compressed axes. The dimensions $K$ are picked by a signal-to-noise criterion measured during a short warmup: an axis is compressed when its second-moment entries are well described by their mean (high SNR), giving up to ~98% second-moment memory savings with little quality loss.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \mathbb{E}_K[\, g_t^2 \,] \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moments, $\beta_1,\beta_2$ their decay rates, $\epsilon$ a stability constant, and $\mathbb{E}_K[\cdot]$ the mean over the compressed dimensions $K$ (fan-in, fan-out, or both), with the resulting value broadcast back across $K$ in the update.

Reference: Dayal Singh Kalra, John Kirchenbauer, Maissam Barkeshli, Tom Goldstein, "When Can You Get Away with Low Memory Adam?", arXiv 2025. https://arxiv.org/abs/2503.01843

---
[Back to the Canon](../index.md)
