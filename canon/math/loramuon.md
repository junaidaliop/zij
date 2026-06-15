# LoRA-Muon

Implements LoRA-Muon, Muon's spectral steepest descent specialized to the low-rank manifold of LoRA factors.

LoRA finetuning writes the adapted weight as $W = AB^\top$ with factors $A \in \mathbb{R}^{m\times r}$, $B \in \mathbb{R}^{n\times r}$. Applying factor-wise optimizers such as AdamW makes learning rates transfer poorly across rank and scale. LoRA-Muon instead solves the spectral-norm steepest-descent problem on the fixed-rank manifold $\mathcal{M}_r = \{W : \mathrm{rank}(W)=r\}$, so the update is the Muon update of the product $W$ projected onto the tangent space. The trust-region budget is split evenly between the two tangent components $\Delta A\, B^\top$ and $A\, \Delta B^\top$, each side whitened by the current Gram geometry $S_A = A^\top A$, $S_B = B^\top B$ before the matrix-sign step. A split weight-decay rule applies decay to the composed weight $W$ rather than to each factor, keeping step sizes matched to full-rank Muon.

$$
\begin{aligned}
g_t^A &= \nabla_A f(W_{\mathrm{pre}} + A_t B_t^\top), \quad g_t^B = \nabla_B f(W_{\mathrm{pre}} + A_t B_t^\top) \\
m_t^A &= \beta\, m_{t-1}^A + (1-\beta)\, g_t^A, \quad m_t^B = \beta\, m_{t-1}^B + (1-\beta)\, g_t^B \\
S_A &= A_t^\top A_t, \quad S_B = B_t^\top B_t, \quad R_A = S_A^{-1/2}, \quad R_B = S_B^{-1/2} \\
\Delta A_t &= -\tfrac{\eta_t}{2}\, \mathrm{msign}\!\big(m_t^A R_B\big)\, R_B, \quad \Delta B_t = -\tfrac{\eta_t}{2}\, \mathrm{msign}\!\big(m_t^B R_A\big)\, R_A \\
s_t &= \sqrt{1 - \lambda \eta_t} \\
A_{t+1} &= s_t A_t + s_t^{-1} \Delta A_t, \quad B_{t+1} = s_t B_t + s_t^{-1} \Delta B_t
\end{aligned}
$$

where $A,B$ are the LoRA factors, $W_{\mathrm{pre}}$ is the frozen base weight, $\eta_t$ is the learning rate, $\beta$ the momentum, $\lambda$ the weight decay, $m_t^A,m_t^B$ the factor first moments, $S_A,S_B$ the factor Gram matrices with inverse square roots $R_A,R_B$, and $\mathrm{msign}(X)=UV^\top$ for the SVD $X=U\Sigma V^\top$ (the spectral-norm linear minimization oracle, realized by Newton-Schulz iteration without an explicit SVD).

Reference: Franz Louis Cesista, Cédric Simal, Katherine Crowson, Stella Biderman, "LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold", arXiv 2026. https://arxiv.org/abs/2606.12921

---
[Back to the Canon](../README.md)
