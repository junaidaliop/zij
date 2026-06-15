# Inverse square root

Implements Inverse Square Root, a learning rate schedule that warms up linearly then decays in proportion to the inverse square root of the step count.

Introduced as the Transformer learning rate schedule, it raises the rate linearly over the first $w$ warmup steps and afterward scales it by $1/\sqrt{t}$, so the rate peaks at the end of warmup and falls off as the inverse square root of the step number. The original schedule additionally scales by $d_{\mathrm{model}}^{-1/2}$, where $d_{\mathrm{model}}$ is the model embedding dimension.

$$
\eta_t = d_{\mathrm{model}}^{-1/2} \cdot \min\!\left( t^{-1/2},\; t \cdot w^{-3/2} \right)
$$

where $\eta_t$ is the learning rate at step $t$, $w$ is the number of warmup steps (4000 in the original work), and $d_{\mathrm{model}}$ is the model embedding dimension. For $t \le w$ the $t \cdot w^{-3/2}$ term dominates (linear warmup); for $t > w$ the $t^{-1/2}$ term dominates (inverse square root decay).

Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017. https://arxiv.org/abs/1706.03762

---
[Back to the Canon](../README.md)
