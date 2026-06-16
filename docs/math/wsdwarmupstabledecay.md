# WSD (Warmup-Stable-Decay)

Implements WSD (Warmup-Stable-Decay), a learning rate schedule that holds a constant rate for most of training and anneals only at the end.

WSD splits training into three stages: a warmup that ramps the rate linearly to a maximum $\eta$, a stable stage that keeps it fixed at $\eta$, and a decay stage that drives it down via a decreasing function $f$. Because the stable stage uses a constant rate, the schedule does not need the total step budget fixed in advance: a strong checkpoint exists at the end of the stable stage, and a short decay from any such point yields a fully annealed model. MiniCPM instantiates the decay with exponential annealing, e.g. $f(s-T) = 0.5^{(s-S)/T}$ over the decay window.

$$
\eta_s = \mathrm{WSD}(T; s) = \begin{cases}
\dfrac{s}{W}\,\eta, & s < W \\
\eta, & W < s < T \\
f(s - T)\,\eta, & T < s
\end{cases}
$$

where $s$ is the current step, $\eta$ the peak learning rate, $W$ the end of the warmup stage, $T$ the end of the stable stage, and $f$ a decreasing function with $0 < f(s-T) \le 1$ (here $S$ and $T$ parametrize the exponential decay rate).

Reference: Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", arXiv 2024. https://arxiv.org/abs/2404.06395

---
[Back to the Canon](../index.md)
