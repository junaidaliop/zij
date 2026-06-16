# SparseLoCo

Implements SparseLoCo, communication-efficient distributed pre-training with Top-k sparsification, quantization, and error feedback.

SparseLoCo is a DiLoCo-style method: each of the $R$ workers runs $H$ local AdamW steps, then forms a pseudo-gradient $\Delta_r$ equal to the drift of its parameters. Rather than transmit the dense pseudo-gradient, each worker accumulates it into an error-feedback buffer $e_r$ decayed by $\beta$, transmits only the chunk-wise Top-$k$ entries (further quantized by $Q$), and carries the residual forward. The decayed accumulator plays the role of DiLoCo's outer momentum, so the global step is a plain averaged descent on the sparse updates with no separate momentum state.

$$
\begin{aligned}
\theta_r^{(t)} &\leftarrow \mathrm{AdamW}^{(H)}\big(\theta^{(t-1)}\big) \\
\Delta_r^{(t)} &= \theta^{(t-1)} - \theta_r^{(t)} \\
e_r^{(t)} &\leftarrow \beta\, e_r^{(t)} + \Delta_r^{(t)} \\
\hat{\Delta}_r^{(t)} &= Q\big(\mathrm{Top\text{-}k}(e_r^{(t)})\big) \\
e_r^{(t+1)} &\leftarrow e_r^{(t)} - \hat{\Delta}_r^{(t)} \\
\Delta^{(t)} &= \frac{1}{R}\sum_{r=1}^{R} \hat{\Delta}_r^{(t)} \\
\theta^{(t+1)} &\leftarrow \theta^{(t)} - \alpha\, \Delta^{(t)}
\end{aligned}
$$

where $\theta$ are the shared parameters, $\theta_r^{(t)}$ the worker-$r$ copy after $H$ inner AdamW steps, $\Delta_r^{(t)}$ its pseudo-gradient, $e_r$ the per-worker error-feedback accumulator, $\beta$ its momentum/decay, $\mathrm{Top\text{-}k}$ the chunk-wise sparsifier keeping the $k$ largest-magnitude entries per chunk, $Q$ the quantizer, $\hat{\Delta}_r^{(t)}$ the transmitted sparse-quantized update, $R$ the number of workers, and $\alpha$ the outer learning rate.

Reference: Amir Sarfi, Benjamin Thérien, Joel Lidin, Eugene Belilovsky, "Communication Efficient LLM Pre-training with SparseLoCo", 2025. https://arxiv.org/abs/2508.15706

---
[Back to the Canon](../index.md)
