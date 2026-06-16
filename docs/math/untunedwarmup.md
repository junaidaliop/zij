# Untuned Warmup

Implements Untuned Warmup, a tuning-free learning-rate warmup for adaptive optimizers.

Adaptive methods such as Adam suffer from high variance in the early steps because the second-moment estimate $v_t$ is built from few samples. RAdam corrects this with a rectification term; this work shows that a simple warmup that ramps the learning rate from $0$ up to its full value achieves the same effect, and that the warmup length can be set directly from $\beta_2$ rather than tuned. Two schedules are given: an exponential variant with time constant $\tau = (1-\beta_2)^{-1}$ and a linear variant that reaches full rate after $\tau = 2(1-\beta_2)^{-1}$ steps. The base optimizer step is unchanged; only the effective step size is scaled by the warmup factor $\omega_t$.

$$
\begin{aligned}
\omega_t^{\text{expo}} &= 1 - \exp\!\big(-(1-\beta_2)\, t\big) \\
\omega_t^{\text{linear}} &= \min\!\Big\{1,\; \tfrac{1-\beta_2}{2}\, t\Big\} \\
\gamma_t &= \omega_t \cdot \gamma
\end{aligned}
$$

where $t$ is the iteration count, $\beta_2$ the second-moment decay rate, $\gamma$ the base learning rate, $\gamma_t$ the warmed-up learning rate applied at step $t$, and $\omega_t \in (0,1]$ the warmup factor (either the exponential or linear form).

Reference: Jerry Ma, Denis Yarats, "On the Adequacy of Untuned Warmup for Adaptive Optimization", AAAI 2021. https://arxiv.org/abs/1910.04209

---
[Back to the Canon](../index.md)
