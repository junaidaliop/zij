# BlockOptimizer

Implements BAdam, block coordinate descent with Adam as the inner solver.

The parameters are partitioned into $D$ blocks
$\theta = (\theta_{\pi_1}, \ldots, \theta_{\pi_D})$. Only the
active block $\pi_i$ is trainable; it receives $K$ steps of
the base optimizer, Adam in the paper, before the next block becomes
active and the optimizer state is reset:


$$
\begin{aligned}
g_k &= \nabla_{\theta_{\pi_i}} \mathcal{L}(\theta) \\
m_k &= \beta_1 m_{k-1} + (1 - \beta_1) g_k \\
v_k &= \beta_2 v_{k-1} + (1 - \beta_2) g_k^2 \\
\theta_{\pi_i} &\leftarrow \theta_{\pi_i} - \eta\,
    \frac{m_k / (1 - \beta_1^k)}{\sqrt{v_k / (1 - \beta_2^k)}
    + \epsilon}
\end{aligned}
$$

with $m_0 = v_0 = 0$ at every block switch and all blocks other
than $\pi_i$ frozen. Only the active block carries optimizer state
and a float32 master copy, so the memory overhead is that of a single
block rather than the full model.

Reference: Qijun Luo, Hengxu Yu, Xiao Li,
"BAdam: A Memory Efficient Full Parameter Optimization Method for Large
Language Models", NeurIPS 2024.
https://arxiv.org/abs/2404.02827


**Note:** Pass `model.named_parameters()` as `params` so blocks can be inferred from transformer layer names, or set `block_prefix_list` explicitly; a plain parameter list falls back to one block per parameter. `base_optimizer` may be an optimizer class, constructed with the remaining keyword arguments, or an already constructed instance. The memory savings assume fp16/bf16 model weights.


---
[Back to the Canon](../README.md)
