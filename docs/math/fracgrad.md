# FracGrad

Implements FracGrad, a fractional-integral reweighting of accumulated microbatch gradients.

Gradient accumulation sums the gradients of $N$ sequential microbatches before a single parameter update, normally weighting each microbatch equally. FracGrad instead derives the per-microbatch weights from a discretized Riemann–Liouville fractional integral, yielding a power-law schedule that biases the accumulated gradient toward the most recent microbatches while retaining the contribution of earlier ones.

Within an accumulation window let $g_t^{(i)}$ be the gradient of the $i$-th microbatch ($i = 1, \dots, N$, with $i = N$ the most recent). The weighted accumulated gradient and parameter step are

$$
\begin{aligned}
w_i(\alpha) &= \frac{(N - i + 1)^{\alpha} - (N - i)^{\alpha}}{\sum_{j=1}^{N} \left[ (N - j + 1)^{\alpha} - (N - j)^{\alpha} \right]}, \\
g_t &= \sum_{i=1}^{N} w_i(\alpha)\, g_t^{(i)}, \\
\theta_{t+1} &= \theta_t - \eta\, g_t.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t^{(i)}$ the $i$-th microbatch gradient within an accumulation window of size $N$, $g_t$ the fractionally weighted accumulated gradient, and $\alpha \in (0, 1]$ the fractional order controlling recency bias. The weights sum to one; $\alpha = 1$ recovers uniform averaging, and smaller $\alpha$ skews weight toward recent microbatches. The accumulated $g_t$ may also be passed to any base optimizer in place of the plain mean.

Reference: Minhyeok Lee, "FracGrad: A Discretized Riemann–Liouville Fractional Integral Approach to Gradient Accumulation for Deep Learning", Fractal and Fractional 2025. https://doi.org/10.3390/fractalfract9110733

---
[Back to the Canon](../index.md)
