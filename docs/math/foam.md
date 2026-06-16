# FOAM

Implements FOAM, a memory-efficient Adam variant that folds optimizer states into shared blocks and corrects them with a lightweight residual.

FOAM (Folded Optimizer with Approximate Moment) cuts optimizer-state memory by averaging adjacent gradient entries into blocks of size $2^l$ and storing the first and second moments only in this compressed space. A fold operator $A^{(l)}$ averages each block down to one value; the matching unfold operator $E^{(l)}$ broadcasts a compressed value back to every entry of its block.

To keep parameters within a block from receiving identical updates, FOAM adds a residual $R_t$ that captures the fold-unfold error of the current gradient. The compressed moments are unfolded and combined with this residual to form full-space moments, after which a standard Adam-style adaptive step is applied.

$$
\begin{aligned}
\tilde{g}_t &= g_t\, A^{(l)} \\
\tilde{m}_t &= \beta_1\, \tilde{m}_{t-1} + (1-\beta_1)\, \tilde{g}_t \\
\tilde{v}_t &= \beta_2\, \tilde{v}_{t-1} + (1-\beta_2)\, \tilde{g}_t^{\,2} \\
R_t &= g_t - \tilde{g}_t\, E^{(l)} \\
m_t &= \tilde{m}_t\, E^{(l)} + R_t \\
v_t &= \tilde{v}_t\, E^{(l)} + R_t^{\,2} \\
\theta_t &= \theta_{t-1} - \eta_t\, \alpha\, \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $\alpha$ a scaling coefficient, $g_t$ the gradient, $\tilde{m}_t$/$\tilde{v}_t$ the compressed first and second moments, $m_t$/$v_t$ their unfolded full-space counterparts, $R_t$ the fold-unfold residual, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ the stability constant. The fold operator $A^{(l)}_{i,j} = 2^{-l}$ when $(j-1)2^l < i \le j\,2^l$ (else $0$) averages each block of $2^l$ adjacent entries, and the unfold operator $E^{(l)}_{i,j} = 1$ when $(i-1)2^l < j \le i\,2^l$ (else $0$) replicates the compressed value across that block.

Reference: Ziqing Wen, Jiahuan Wang, Ping Luo, Dongsheng Li, Tao Sun, "FOAM: Blocked State Folding for Memory-Efficient LLM Training", 2025. https://arxiv.org/abs/2512.07112

---
[Back to the Canon](../index.md)
