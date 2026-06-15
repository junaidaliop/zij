# SM3

Implements SM3, the memory-efficient adaptive method of Anil et al.

This is the SM3-II variant. For a parameter tensor, the cover sets
$S_r$ are its slices along each axis, so a $d_1 \times d_2$
matrix keeps $d_1 + d_2$ accumulator entries instead of
$d_1 d_2$:


$$
\begin{aligned}
     \nu_t(j) &= \min_{r : S_r \ni j} \mu_{t-1}(r) + g_t(j)^2 \\
     \theta_{t+1}(j) &= \theta_t(j) - \eta \,
         \frac{g_t(j)}{\sqrt{\nu_t(j)}} \\
     \mu_t(r) &= \max_{j \in S_r} \nu_t(j)
\end{aligned}
$$


**Note:** The defaults follow the paper: with `beta=0` the accumulators upper bound the running sums of squared gradients. Setting `beta > 0` replaces the sums with exponential moving averages, and `momentum > 0` adds a moving average of the preconditioned update, at the cost of one extra buffer per parameter. Momentum is ignored for sparse gradients.

Reference: Rohan Anil, Vineet Gupta, Tomer Koren, Yoram Singer,
"Memory-Efficient Adaptive Optimization", NeurIPS 2019.
https://arxiv.org/abs/1901.11150

---
[Back to the Canon](../README.md)
