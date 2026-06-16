# 8-bit Muon

Implements 8-bit Muon, a memory-efficient Muon that stores the momentum buffer in blockwise 8-bit quantized form.

Muon updates each 2D weight matrix by maintaining a momentum buffer, orthogonalizing it with a Newton-Schulz iteration, and stepping along the resulting orthogonal factor. 8-bit Muon keeps the same update but never holds the full-precision momentum: at each step the stored 8-bit state $Z^{(t-1)}$ is dequantized, the momentum is refreshed and used for the step, then re-quantized back to 8 bits. Quantization is applied per block (block size 2048) using a symmetric linear map normalized by the block's infinity norm, cutting optimizer-state memory by roughly 74%.

$$
\begin{aligned}
\tilde{m}_{t-1} &= \mathrm{Dequantize}_{\mathcal{S}}(Z^{(t-1)}) \\
m_t &= \beta\, \tilde{m}_{t-1} + g_t \\
O_t &= \mathrm{NS}(m_t) \\
\theta_t &= (1 - \gamma\lambda)\,\theta_{t-1} - 0.2\,\gamma\,\sqrt{\max(m,n)}\; O_t \\
Z^{(t)} &= \mathrm{Quantize}_{\mathcal{S}}(m_t), \qquad [Q(x)]_i = \frac{\lVert x \rVert_\infty}{127}\,\mathrm{round}\!\left(\frac{127\, x_i}{\lVert x \rVert_\infty}\right)
\end{aligned}
$$

where $g_t = \nabla f(\theta_{t-1})$ is the gradient, $\tilde{m}_t$ the dequantized momentum, $Z^{(t)}$ the stored 8-bit momentum with auxiliary scale state $\mathcal{S}$, $\beta$ the momentum decay, $\gamma$ the learning rate, $\lambda$ the weight decay, $m\times n$ the shape of the weight matrix, $\mathrm{NS}$ the Newton-Schulz iteration approximating the polar (orthogonal) factor of $m_t$, and $\lVert\cdot\rVert_\infty$ the (per-block) max-magnitude used to scale the linear quantizer $Q$ onto the signed 8-bit grid.

Reference: Aman Gupta, Rafael Celente, Abhishek Shivanna, D. T. Braithwaite, Gregory Dexter, Shao Tang, Hiroto Udagawa, Daniel Silva, Rohan Ramanath, S. Sathiya Keerthi, "Effective Quantization of Muon Optimizer States", arXiv 2025. https://arxiv.org/abs/2509.23106

---
[Back to the Canon](../index.md)
