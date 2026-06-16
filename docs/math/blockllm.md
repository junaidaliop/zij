# BlockLLM

Implements BlockLLM, memory-efficient fine-tuning that updates only a sparse set of coordinate blocks selected by Adam-processed gradient magnitude.

BlockLLM treats adaptation as block-coordinate descent over the network's parameters. It maintains Adam's first and second moments for the gradient, forms the preconditioned gradient $\tilde{g}_t$, and at each selection step picks the blocks (layers) whose $|\tilde{g}_t|$ is largest, biasing toward blocks that have been visited less often. Only the selected set $S$ receives updates; all other parameters are frozen, so optimizer state and gradients are stored for a small fraction of the model. The selected set is revised only when the loss stops improving relative to its recent moving average, which keeps the active block stable for many steps.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \\
\tilde{g}_t &= \frac{m_t}{\sqrt{v_t}+\epsilon}, \\
S &= \mathrm{top\text{-}k}_\ell \left( \frac{\lVert \tilde{g}_t^{(\ell)} \rVert}{f_\ell} \right), \\
\theta_{t+1}^{(\ell)} &=
\begin{cases}
\theta_t^{(\ell)} - \eta\, \tilde{g}_t^{(\ell)} & \ell \in S, \\
\theta_t^{(\ell)} & \ell \notin S.
\end{cases}
\end{aligned}
$$

where $\theta^{(\ell)}$ are the parameters of block (layer) $\ell$, $g_t$ is the gradient, $m_t,v_t$ are Adam moments with decays $\beta_1,\beta_2$, $\tilde{g}_t$ is the preconditioned gradient, $\eta$ is the learning rate, $\epsilon$ is the stability constant, $f_\ell$ is the visit frequency of block $\ell$, and $S$ is the selected block set, reselected when the loss $\phi_t$ exceeds its $m$-step moving average $\tfrac{1}{m}\sum_{i=t-m+1}^{t} H[i]$.

Reference: Amrutha Varshini Ramesh, Vignesh Ganapathiraman, Issam H. Laradji, Mark Schmidt, "BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks", arXiv 2024. https://arxiv.org/abs/2406.17296

---
[Back to the Canon](../index.md)
