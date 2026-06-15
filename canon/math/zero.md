# ZeRO

Implements ZeRO (Zero Redundancy Optimizer), a memory-partitioning scheme for data-parallel training rather than a parameter-update rule.

ZeRO is not an optimizer in the SGD/Adam sense: it does not define how $\theta$ moves in response to $g_t$. Instead it eliminates the memory redundancy of standard data parallelism, where every device holds a full copy of the model states. The actual parameter update is delegated to whatever base optimizer is used (typically Adam). ZeRO-DP partitions the three classes of training state across the $N_d$ data-parallel devices in three cumulative stages: optimizer states ($P_{os}$), then gradients ($P_{os+g}$), then parameters ($P_{os+g+p}$). Each stage trades a small amount of extra communication for a proportional reduction in per-device memory, so the per-device footprint shrinks toward $0$ as $N_d$ grows while the global computation stays mathematically identical to ordinary data parallelism.

For mixed-precision training with $\Psi$ parameters, each device under plain data parallelism stores $2\Psi$ bytes of fp16 parameters, $2\Psi$ bytes of fp16 gradients, and $K\Psi$ bytes of fp32 optimizer states (with $K=12$ for Adam: an fp32 parameter copy plus first and second moments), giving a baseline of $(2+2+K)\Psi = 16\Psi$ bytes. The three ZeRO stages reduce this per-device cost to:

$$
\begin{aligned}
M_{\text{baseline}} &= (2 + 2 + K)\,\Psi \\
M_{P_{os}} &= 2\Psi + 2\Psi + \frac{K\Psi}{N_d} \\
M_{P_{os+g}} &= 2\Psi + \frac{(2 + K)\Psi}{N_d} \\
M_{P_{os+g+p}} &= \frac{(2 + 2 + K)\Psi}{N_d}
\end{aligned}
$$

where $\Psi$ is the number of model parameters, $K$ is the optimizer-state memory multiplier ($K=12$ for mixed-precision Adam), and $N_d$ is the number of data-parallel devices; memory is measured in bytes per device.

Reference: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020. https://arxiv.org/abs/1910.02054

---
[Back to the Canon](../README.md)
