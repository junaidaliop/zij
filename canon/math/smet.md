# SMET

Implements SMET (Sparse Memory-Efficient Training), a dynamic-sparsity training scheme that runs Adam only over active parameters.

SMET pre-trains language models under dynamic sparse training: gradients and optimizer states are stored solely for the active (unmasked) parameters, cutting memory. Two stabilizers fix the instability that arises when the sparse connectivity is periodically updated. First, each newly regrown parameter gets a local timestep reset to $1$, which restores Adam's bias correction for that connection (a shared global timestep would otherwise inflate its first updates by roughly $3.16\times$) and is followed by a short linear learning-rate warm-up over $W$ steps. Second, the base learning rate is scaled by the network density to keep update magnitudes stable across sparsity levels.

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t_i}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t_i}} \\
\alpha_s &= \alpha_0 \cdot \frac{1}{\sqrt{d}} \\
\alpha_t &= \alpha_s \cdot \frac{t}{W}, \qquad t \in \{1,\ldots,W\} \\
\theta_t &= \theta_{t-1} - \alpha_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

where $\theta$ are the active parameters, $m_t,v_t$ the Adam moments updated with decays $\beta_1,\beta_2$, $t_i$ the per-parameter local timestep (reset to $1$ on regrowth), $\alpha_0$ the dense-training base learning rate, $d \in (0,1]$ the current density (fraction of active parameters), $\alpha_s$ the density-scaled target rate, $W$ the warm-up length, and $\epsilon$ a stability constant.

Reference: Qiao Xiao, Boqian Wu, Patrik Okanovic, Tomasz Sternal, Maurice van Keulen, Elena Mocanu, Mykola Pechenizkiy, Decebal Constantin Mocanu, Torsten Hoefler, "Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling", ICML 2026. https://arxiv.org/abs/2606.00888

---
[Back to the Canon](../README.md)
